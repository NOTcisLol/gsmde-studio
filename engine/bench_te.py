"""Mede qual (dispositivo, precisao) e' melhor para o TEXT ENCODER NESTA maquina.

POR QUE MEDIR EM VEZ DE CONSULTAR UMA TABELA
    Tabela "placa X prefere dtype Y" envelhece e, pior, nao enxerga o build. O
    caso desta maquina prova: o torch ROCm vem SEM MKL, entao fp32 na CPU cai
    num caminho ingenuo e fica ~5x mais lento que bf16 (que tem caminho nativo
    no oneDNN). Nenhuma tabela por modelo de GPU preveria isso — e' propriedade
    do build do torch, nao do hardware.

    Alem disso, o que interessa aqui nao e' o pico de FLOPs da placa: o TE roda
    UMA vez por geracao, com sequencia curta. O que decide e' latencia de um
    encode, incluindo o transito de subir e descer os pesos quando aplicavel.

SOBRE int8
    Nao entra como opcao porque nao e' um dtype no sentido de `.to()`: o
    PyTorch recusa (`nn.Module.to only accepts floating point or complex`).
    Rodar CLIP em int8 exige QUANTIZAR (substituir camadas, calibrar), o que e'
    outro recurso — nao uma precisao alternativa.

Uso:
    python bench_te.py                       # cpu e cuda, precisoes viaveis
    python bench_te.py --repete 3 --prompt "..."
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

BASE_PADRAO = next((p for p in (
    r"D:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors",
    r"G:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors") if Path(p).exists()), "")

PROMPT = ("1girl, anthro wolf, thick fur, detailed hands with jewelry, wet skin, "
          "stone wall background, dramatic lighting, cinematic, highly detailed, "
          "intricate texture, volumetric light, depth of field, masterpiece")
NEG = ("worst quality, low quality, bad anatomy, bad hands, extra digits, "
       "jpeg artifacts, blurry, lowres")

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _mede(pipe, dev, dt, centros, repete, sequencial):
    """Segundos por encode completo (todos os centros), ja incluindo o transito
    quando o modo e' sequencial."""
    import prompt_weighting as W
    tes = [pipe.text_encoder, pipe.text_encoder_2]
    tempos = []
    for _ in range(repete):
        W._CACHE_NEG.clear()          # mede o caso honesto, sem cache quente
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        if sequencial:
            for te in tes:
                te.to(dev, dt)
        for w in centros:
            W.get_center_weighted_embeddings(pipe, PROMPT, NEG, [w], device=dev)
        if sequencial:
            for te in tes:
                te.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tempos.append(time.time() - t0)
    return min(tempos)


def _embed(pipe, dev, dt, palavra):
    """Um embedding de referencia, p/ comparar precisoes."""
    import prompt_weighting as W
    for te in (pipe.text_encoder, pipe.text_encoder_2):
        te.to(dev, dt)
    pe, npe, pp, npp = W.get_center_weighted_embeddings(
        pipe, PROMPT, NEG, [palavra], device=dev)
    return pe.detach().float().cpu()


def divergencia(pipe, ref_dev="cpu"):
    """Quanto o embedding MUDA ao baixar a precisao, tomando fp32 como verdade.

    Responde a pergunta que importa: velocidade so' justifica perder precisao se
    a perda for pequena. Numero e' distancia de cosseno media — 0 = identico.
    Referencia de escala: mudancas de ~1e-4 sao ruido numerico; ~1e-2 ja mexe
    visivelmente no condicionamento.
    """
    import torch as _t
    ref = _embed(pipe, _t.device(ref_dev), _t.float32, "wolf")
    out = {}
    for nome, dt in (("fp16", _t.float16), ("bf16", _t.bfloat16)):
        try:
            e = _embed(pipe, _t.device(ref_dev), dt, "wolf")
            a, b = ref.flatten(), e.flatten()
            cos = _t.dot(a, b) / (a.norm() * b.norm())
            out[nome] = {"dist_cosseno": float(1 - cos),
                         "err_rel_medio": float((e - ref).abs().mean() / ref.abs().mean())}
        except Exception as ex:
            out[nome] = {"erro": str(ex)[:60]}
    for te in (pipe.text_encoder, pipe.text_encoder_2):
        te.to("cpu", _t.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_PADRAO)
    ap.add_argument("--centros", type=int, default=6)
    ap.add_argument("--repete", type=int, default=2)
    ap.add_argument("--saida", default=r"D:\GSMDE\trainer\config\te_bench.json")
    a = ap.parse_args()
    if not a.base:
        raise SystemExit("nao achei o checkpoint base; passe --base")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from diffusers import StableDiffusionXLPipeline as P

    print(f"carregando os text encoders de {Path(a.base).name} ...", flush=True)
    pipe = P.from_single_file(a.base, torch_dtype=torch.float32)
    # o UNet nao participa: liberar deixa a medida limpa e a VRAM livre
    pipe.unet = None
    pipe.vae = None
    centros = ["wolf", "fur", "hands", "jewelry", "skin", "stone",
               "lighting", "forest"][:a.centros]

    combos = [("cpu", "bf16", False), ("cpu", "fp32", False), ("cpu", "fp16", False)]
    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        combos += [("cuda", "fp16", True), ("cuda", "bf16", True),
                   ("cuda", "fp32", True)]
    else:
        nome = "(sem gpu)"

    print(f"gpu: {nome} | torch {torch.__version__}")
    print(f"{a.centros} centros, melhor de {a.repete}\n")
    print(f"{'dispositivo':12} {'precisao':9} {'modo':11} {'segundos':>9}")
    res = []
    for dev_s, dt_s, seq in combos:
        dev = torch.device(dev_s)
        dt = DTYPES[dt_s]
        try:
            if not seq:
                for te in (pipe.text_encoder, pipe.text_encoder_2):
                    te.to(dev, dt)
            t = _mede(pipe, dev, dt, centros, a.repete, seq)
            modo = "sequencial" if seq else "residente"
            print(f"{dev_s:12} {dt_s:9} {modo:11} {t:9.2f}")
            res.append({"dispositivo": dev_s, "precisao": dt_s,
                        "modo": modo, "segundos": round(t, 3)})
        except Exception as e:
            print(f"{dev_s:12} {dt_s:9} {'—':11} {'FALHOU':>9}  {str(e)[:60]}")
        finally:
            for te in (pipe.text_encoder, pipe.text_encoder_2):
                te.to("cpu", torch.float32)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if res:
        res.sort(key=lambda x: x["segundos"])
        melhor = res[0]
        print(f"\nMELHOR: {melhor['dispositivo']} / {melhor['precisao']} / "
              f"{melhor['modo']} ({melhor['segundos']}s)")
        pior = res[-1]
        print(f"contra o pior ({pior['dispositivo']}/{pior['precisao']}): "
              f"{pior['segundos'] / melhor['segundos']:.1f}x")
        print("\nQUANTO SE PERDE (fp32 = referencia):")
        div = divergencia(pipe)
        for nome, d in div.items():
            if "erro" in d:
                print(f"  {nome}: {d['erro']}")
            else:
                print(f"  {nome}: distancia de cosseno {d['dist_cosseno']:.2e} | "
                      f"erro relativo medio {d['err_rel_medio']:.2e}")
        Path(a.saida).write_text(json.dumps(
            {"gpu": nome, "torch": torch.__version__, "centros": a.centros,
             "medido_em": time.strftime("%Y-%m-%d %H:%M"), "resultados": res,
             "melhor": melhor, "divergencia": div}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"salvo em {a.saida}")


if __name__ == "__main__":
    main()
