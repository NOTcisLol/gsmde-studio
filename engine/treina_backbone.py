"""FASE B: treina o student de ~1,7B a partir do cache do teacher.

TRES VARIANTES (--alvo), a mesma arquitetura e o mesmo dado em todas:

    teacher   MSE(student_eps, eps_do_teacher)   -> destilacao
    raw       MSE(student_eps, ruido_verdadeiro) -> difusao padrao, sem professor
    merge     igual a 'teacher', mas o cache veio do teacher COM os LoRAs assados

PROTOCOLO DE FILA (pedido do usuario)
    3 passadas completas primeiro, sem ninguem sair. So depois disso a amostra
    que atingir loss < 0.01 e' APROVADA e deixa a fila; as demais voltam para o
    fim. Repete ate a fila esvaziar.

    O aquecimento existe porque nas primeiras passadas o loss ainda reflete o
    estado do modelo, nao a dificuldade da amostra: aprovar cedo deixaria sair
    justamente as faceis e o resto viraria uma fila de casos dificeis com o
    modelo ainda cru.

MEMORIA (regra do usuario: offload, nunca spill)
    Medido: 1,733B treina a 512px ocupando 6,26 GB, com 0,28 GB livres. Cabe, mas
    sem folga — por isso gradient checkpointing sempre ligado, Adafactor (fatorado;
    Adam precisaria de ~12 GB so de estado) e nada mais na placa. O teacher nao
    aparece aqui: ele ja fez o trabalho dele no cache.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import torch


def cfg_student():
    """~1,7B mantendo a LINGUAGEM do SDXL — 2 text encoders (cross_attention_dim
    2048), add_time_ids e o mesmo espaco latente. Sem isso o modelo nao poderia
    reusar VAE, prompts nem os centros."""
    return dict(
        sample_size=128, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(320, 640, 1280),
        layers_per_block=2,
        transformer_layers_per_block=(1, 2, 6),
        cross_attention_dim=2048,
        attention_head_dim=(5, 10, 20),
        addition_embed_type="text_time",
        addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=2816,
        norm_num_groups=32, use_linear_projection=True,
    )


def carrega_indice(cache: Path):
    """(shard, i) de cada amostra — o dado fica no disco ate a hora do uso."""
    idx = []
    for sh in sorted(cache.glob("shard_*.pt")):
        n = len(torch.load(sh, map_location="cpu", weights_only=False))
        idx += [(sh, i) for i in range(n)]
    return idx


class Shards:
    """Cache de um shard por vez: a fila embaralha, mas ler o mesmo shard em
    sequencia evita reabrir o arquivo a cada amostra."""

    def __init__(self):
        self.nome, self.dados = None, None

    def pega(self, sh, i):
        if self.nome != sh:
            self.dados = torch.load(sh, map_location="cpu", weights_only=False)
            self.nome = sh
        return self.dados[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--alvo", choices=["teacher", "raw", "merge"], default="teacher")
    ap.add_argument("--passadas-aquecimento", type=int, default=3)
    ap.add_argument("--loss-aprova", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--max-horas", type=float, default=0, help="0 = sem limite")
    ap.add_argument("--salva-cada", type=int, default=2000)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diffusers import UNet2DConditionModel
    from transformers.optimization import Adafactor

    saida = Path(a.saida); saida.mkdir(parents=True, exist_ok=True)
    idx = carrega_indice(Path(a.cache))
    print(f"[treino] alvo={a.alvo} | {len(idx)} amostras no cache", flush=True)
    if not idx:
        print("[treino] cache vazio — abortando", flush=True); return

    dev = "cuda"
    u = UNet2DConditionModel.from_config(cfg_student()).to(torch.float16)
    n = sum(p.numel() for p in u.parameters())
    u.enable_gradient_checkpointing(); u.train(); u.to(dev)
    print(f"[treino] student {n/1e9:.3f}B ({n*2/2**30:.2f} GB fp16)", flush=True)

    # Adafactor: estado fatorado (O(n+m) por matriz). Adam guardaria dois momentos
    # em fp32 = ~12 GB so de otimizador, o que nao existe nesta placa.
    opt = Adafactor(u.parameters(), lr=a.lr, scale_parameter=False,
                    relative_step=False, warmup_init=False)

    fila = deque(idx)
    random.seed(42); random.shuffle(fila)
    aprovadas, passada, vistas, passo = set(), 0, 0, 0
    shards = Shards()
    add_t = torch.tensor([[a.res, a.res, 0, 0, a.res, a.res]], device=dev,
                         dtype=torch.float16)
    hist, t0 = [], time.time()
    n_total = len(fila)

    while fila:
        if a.max_horas and (time.time() - t0) / 3600 >= a.max_horas:
            print(f"[treino] limite de {a.max_horas}h atingido", flush=True); break
        sh, i = fila.popleft()
        try:
            am = shards.pega(sh, i)
            lat = am["lat"][None].to(dev, torch.float16)
            pe = am["pe"][None].to(dev, torch.float16)
            pp = am["pp"][None].to(dev, torch.float16)
            t = torch.tensor([am["t"]], device=dev)
            alvo = (am["eps"] if a.alvo in ("teacher", "merge") else am["ruido"])
            alvo = alvo[None].to(dev, torch.float16)

            pred = u(lat, t, encoder_hidden_states=pe,
                     added_cond_kwargs={"text_embeds": pp, "time_ids": add_t}).sample
            loss = torch.nn.functional.mse_loss(pred.float(), alvo.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(u.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            l = float(loss.detach())
            passo += 1; vistas += 1
            hist.append(l)

            # FILA: so aprova depois do aquecimento
            if passada >= a.passadas_aquecimento and l < a.loss_aprova:
                aprovadas.add((str(sh), i))
            else:
                fila.append((sh, i))

            if vistas >= n_total:
                passada += 1; vistas = 0
                m = sum(hist[-n_total:]) / max(1, len(hist[-n_total:]))
                print(f"[treino] passada {passada} | fila {len(fila)} | "
                      f"aprovadas {len(aprovadas)} | loss medio {m:.4f} | "
                      f"{(time.time()-t0)/3600:.2f}h", flush=True)
            if passo % 200 == 0:
                m = sum(hist[-200:]) / len(hist[-200:])
                print(f"[treino] passo {passo} | loss(200) {m:.4f} | fila {len(fila)} | "
                      f"{(time.time()-t0)/60:.0f}min", flush=True)
            if passo % a.salva_cada == 0:
                torch.save({"state_dict": u.state_dict(), "cfg": cfg_student(),
                            "passo": passo, "alvo": a.alvo},
                           saida / f"student_{a.alvo}.pt")
                json.dump({"passo": passo, "passada": passada,
                           "aprovadas": len(aprovadas), "fila": len(fila),
                           "loss_ultimos200": sum(hist[-200:]) / len(hist[-200:])},
                          open(saida / f"estado_{a.alvo}.json", "w"), indent=2)
        except Exception as e:
            print(f"[treino] amostra pulada: {type(e).__name__}: {str(e)[:70]}", flush=True)

    torch.save({"state_dict": u.state_dict(), "cfg": cfg_student(),
                "passo": passo, "alvo": a.alvo}, saida / f"student_{a.alvo}.pt")
    print(f"[treino] FIM | {passo} passos | {passada} passadas | "
          f"{len(aprovadas)} aprovadas | {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
