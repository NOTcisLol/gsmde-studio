"""FASE A da destilacao: cacheia latente + embeddings + alvo do teacher em disco.

POR QUE DUAS FASES
    Teacher (5,2 GB) e student em treino (6,26 GB medidos) NAO cabem juntos nos
    8 GB. Rodar os dois alternando custaria mover ~5 GB pelo PCIe duas vezes por
    passo. Entao o teacher roda UMA vez, grava o que o student precisa, e sai de
    cena. Na fase B so o student ocupa a placa.

    E' a regra do usuario aplicada: offload (para disco, aqui), nunca spill.

O QUE VAI PARA O DISCO, POR AMOSTRA
    latente     4x64x64 fp16   ~32 KB   (VAE ja aplicado)
    ruido       idem           ~32 KB   (o mesmo que o teacher viu)
    t           escalar
    eps_teacher idem           ~32 KB   (o alvo da destilacao)
    emb, pooled 77x2048 + 1280 ~310 KB  (os dois text encoders do SDXL)

    ~0,4 MB por amostra. Com K timesteps por imagem, 8617 imagens x 3 = ~10 GB.

POR QUE K TIMESTEPS
    Um so' timestep por imagem faria o student ver sempre o mesmo nivel de ruido
    daquela imagem. Com K amostras espalhadas pela faixa, cada imagem ensina a
    trajetoria inteira — que e' o que a destilacao precisa transferir.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import torch


def refs_do_dataset(p: Path, limite: int = 0):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    r = d.get("refs") or []
    return r[:limite] if limite else r


def abre_imagem(ref, cache_tar: dict):
    """ref = [tar, id, ext, tags]. Le direto do tar, sem extrair para disco."""
    from PIL import Image
    tar, ident, ext, tags = ref[0], ref[1], ref[2], (ref[3] if len(ref) > 3 else "")
    tf = cache_tar.get(tar)
    if tf is None:
        tf = cache_tar[tar] = tarfile.open(tar, "r")
    for nome in (f"{ident}.{ext}", ident, f"./{ident}.{ext}"):
        try:
            f = tf.extractfile(nome)
            if f is not None:
                return Image.open(io.BytesIO(f.read())).convert("RGB"), tags
        except KeyError:
            continue
    return None, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--base", default=r"D:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--timesteps", type=int, default=3, help="amostras por imagem")
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--shard", type=int, default=500, help="amostras por arquivo")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vram_sensor import VramSensor
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler

    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    refs = refs_do_dataset(Path(a.dataset), a.limite)
    print(f"[cache] {len(refs)} imagens x {a.timesteps} timesteps = "
          f"{len(refs)*a.timesteps} amostras", flush=True)

    s = VramSensor()
    pipe = StableDiffusionXLPipeline.from_single_file(a.base, torch_dtype=torch.float16)
    dev = "cuda"
    # O VAE encoda em fp32 para nao gerar NaN; ele e' pequeno e sai da placa depois.
    pipe.vae.to(dev, torch.float32)
    pipe.text_encoder.to(dev); pipe.text_encoder_2.to(dev)
    pipe.unet.to(dev)
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    print(f"[cache] {s.resumo()}", flush=True)

    # ALTERNANCIA EM BLOCOS — mata o spill.
    #
    # VAE + 2 text encoders + UNet juntos na placa dao DESPEJO 0,73GB (medido no
    # smoke test) e a 9 s/imagem nada escala. Eles nao precisam coexistir: o VAE e
    # os TEs so trabalham na ENTRADA, o UNet so na saida.
    #
    # Entao o bloco processa BL imagens com VAE+TE na placa e o UNet na CPU, e
    # depois inverte. O custo de mover o UNet (5,2GB, duas vezes por bloco) se
    # dilui por BL imagens; com BL=64 sao ~0,16GB/imagem contra os ~5GB/imagem
    # que alternar por imagem custaria.
    BL = 64

    def _para(m, d):
        m.to(d)

    cache_tar, buf, n_shard, feitos, t0 = {}, [], 0, 0, time.time()
    for i, ref in enumerate(refs):
        try:
            img, tags = abre_imagem(ref, cache_tar)
            if img is None:
                continue
            img = img.resize((a.res, a.res))
            import numpy as np
            x = torch.from_numpy(np.array(img)).float().div(127.5).sub(1.0)
            x = x.permute(2, 0, 1)[None].to(dev, torch.float32)
            if i % BL == 0:                     # entra o bloco: VAE/TE sobem, UNet desce
                _para(pipe.unet, "cpu"); torch.cuda.empty_cache()
                _para(pipe.vae, dev); _para(pipe.text_encoder, dev)
                _para(pipe.text_encoder_2, dev)
                pend = []
            with torch.no_grad():
                lat = pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
                lat = lat.to(torch.float16)
                prompt = str(tags).replace("_", " ")[:300]
                pe, npe, pp, npp = pipe.encode_prompt(prompt=prompt, device=dev,
                                                      num_images_per_prompt=1,
                                                      do_classifier_free_guidance=False)
                for _ in range(a.timesteps):
                    t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
                    ruido = torch.randn_like(lat)
                    lat_ruid = sched.add_noise(lat, ruido, t)
                    pend.append({"lat": lat_ruid[0].cpu(), "t": int(t.item()),
                                 "ruido": ruido[0].cpu(),
                                 "pe": pe[0].cpu(), "pp": pp[0].cpu()})

            fim_bloco = (i % BL == BL - 1) or (i == len(refs) - 1)
            if fim_bloco and pend:             # sai o bloco: UNet sobe, VAE/TE descem
                _para(pipe.vae, "cpu"); _para(pipe.text_encoder, "cpu")
                _para(pipe.text_encoder_2, "cpu"); torch.cuda.empty_cache()
                _para(pipe.unet, dev)
                add_t = torch.tensor([[a.res, a.res, 0, 0, a.res, a.res]],
                                     device=dev, dtype=torch.float16)
                with torch.no_grad():
                    for am in pend:
                        eps = pipe.unet(am["lat"][None].to(dev), 
                                        torch.tensor([am["t"]], device=dev),
                                        encoder_hidden_states=am["pe"][None].to(dev),
                                        added_cond_kwargs={
                                            "text_embeds": am["pp"][None].to(dev),
                                            "time_ids": add_t}).sample
                        am["eps"] = eps[0].cpu()
                        buf.append(am); feitos += 1
                pend = []
            if len(buf) >= a.shard:
                torch.save(buf, saida / f"shard_{n_shard:05d}.pt")
                n_shard += 1
                buf = []
                el = time.time() - t0
                print(f"[cache] {feitos} amostras | {i+1}/{len(refs)} imgs | "
                      f"{el/60:.1f}min | ETA {(el/max(1,i+1)*(len(refs)-i-1))/60:.0f}min",
                      flush=True)
        except Exception as e:
            print(f"[cache] ref {i} pulada: {type(e).__name__}: {str(e)[:60]}", flush=True)
    if buf:
        torch.save(buf, saida / f"shard_{n_shard:05d}.pt")
        n_shard += 1
    print(f"[cache] FIM: {feitos} amostras em {n_shard} shards | "
          f"{(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
