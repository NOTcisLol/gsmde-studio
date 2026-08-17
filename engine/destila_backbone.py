"""Constroi o STUDENT do backbone encolhendo a FFN e preservando a atencao.

A DESCOBERTA QUE DEFINE A ESTRATEGIA (medido em 2026-07-28)
    Repartindo os 2,567B parametros do UNet SDXL por tipo de modulo:

        FFN (feed-forward) .... 1229,5M   47,9%   <- os centros NAO tocam
        atencao (attn1+attn2) .. 955,4M   37,2%   <- os centros VIVEM aqui
        resnets (conv) ......... 327,9M   12,8%
        resto ................... 54,7M    2,1%

    Os 36 centros treinados so tem chaves de `attn1`/`attn2` (verificado: 1120
    chaves, 560 de cada, zero em FFN ou conv). Ou seja: a metade mais gorda do
    modelo e' invisivel para eles.

    O design original propunha reduzir os CANAIS (320 -> 224). Isso encolhe tudo
    junto — inclusive a atencao — e muda as formas onde os LoRAs entram. Todos os
    36 centros parariam de encaixar: ~60 horas de re-treino, o maior risco listado
    no proprio documento.

    Encolhendo so a FFN, as formas da atencao ficam IDENTICAS e os centros
    continuam carregando sem tocar em nada. E' o corte que da o maior ganho pelo
    menor risco.

COMO O STUDENT NASCE
    Nao e' inicializacao aleatoria: o student HERDA os pesos do teacher. Atencao,
    resnets e embeddings sao copiados bit a bit; so as matrizes da FFN precisam
    ser reduzidas, e para elas usamos truncagem por norma — ficam as linhas de
    maior magnitude, que carregam mais sinal. Herdar e' o que separa "afinar por
    algumas horas" de "treinar do zero por semanas".

USO
    python destila_backbone.py --ff-mult 1.5 --saida D:\Models\gsmde\backbone_s1
    python destila_backbone.py --so-analise        # so mostra a conta, nao grava
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _plano(ff_mult_novo: float, ff_mult_orig: int = 4):
    """Quanto sobra do UNet ao encolher so a FFN. Numeros medidos nesta maquina."""
    total = 2567.0                       # M de params
    ff, attn, res, resto = 1229.5, 955.4, 327.9, 54.2
    ff_novo = ff * (ff_mult_novo / ff_mult_orig)
    novo = attn + res + resto + ff_novo
    return {
        "ff_orig_M": ff, "ff_novo_M": ff_novo,
        "total_orig_M": total, "total_novo_M": novo,
        "reducao_pct": (1 - novo / total) * 100,
        "gb_orig": total * 1e6 * 2 / 2**30,
        "gb_novo": novo * 1e6 * 2 / 2**30,
    }


def analise():
    print("=== quanto se ganha cortando SO a FFN (atencao intacta) ===\n")
    print(f"  {'ff_mult':>8} {'FFN':>10} {'UNet':>10} {'fp16':>9} {'reducao':>9}  LoRAs")
    for m in (4, 3, 2, 1.5, 1.0, 0.5):
        p = _plano(m)
        marca = "  <- alvo ~2,5GB" if 2.3 <= p["gb_novo"] <= 2.7 else ""
        print(f"  {m:>8} {p['ff_novo_M']:9.0f}M {p['total_novo_M']:9.0f}M "
              f"{p['gb_novo']:8.2f}G {p['reducao_pct']:8.1f}%  100% ok{marca}")
    print("\n  Para comparar, o caminho do design original (reduzir canais 320->224)")
    print("  encolheria ~40% MAS mudaria as formas da atencao: os 36 centros")
    print("  deixariam de encaixar (~60h de re-treino).")


def constroi(ff_mult: float, base: str, saida: str):
    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

    print(f"[destila] teacher: {base}", flush=True)
    pipe = StableDiffusionXLPipeline.from_single_file(
        base, torch_dtype=torch.float16, add_watermarker=False)
    teacher = pipe.unet
    cfg = dict(teacher.config)
    n_tea = sum(p.numel() for p in teacher.parameters())

    # A FFN do diffusers nao expoe "mult" na config: o tamanho interno e'
    # derivado (dim * 4) dentro do FeedForward. Entao o student e' construido com
    # a MESMA config e as matrizes da FFN sao cortadas depois, na mao.
    print(f"[destila] student: mesma arquitetura, FFN x{ff_mult} (era x4)", flush=True)
    student = UNet2DConditionModel.from_config(cfg)

    sd_t = teacher.state_dict()
    sd_s = {k: v.clone() for k, v in sd_t.items()}   # atencao/resnets: bit a bit
    herdadas = len(sd_s)

    # UM conjunto de indices POR FFN, aplicado aos tres tensores dela.
    #   net.0.proj.weight [2*inner, dim]  -> corta as linhas (valor e gate)
    #   net.0.proj.bias   [2*inner]       -> as MESMAS linhas
    #   net.2.weight      [dim, inner]    -> as MESMAS colunas
    # Escolher indices separados por tensor nao daria erro de forma nenhum e o
    # modelo sairia com os neuronios trocados — o pior tipo de bug: silencioso.
    ffs = sorted({k.rsplit(".net.", 1)[0] for k in sd_t if ".ff.net.0.proj.weight" in k})
    cortadas = 0
    for base_ff in ffs:
        kw = f"{base_ff}.net.0.proj.weight"
        kb = f"{base_ff}.net.0.proj.bias"
        k2 = f"{base_ff}.net.2.weight"
        v = sd_t[kw]
        idx, inner = _indices_geglu(v, ff_mult)
        sd_s[kw] = torch.cat([v[:inner][idx], v[inner:][idx]], 0).contiguous()
        cortadas += 1
        if kb in sd_t:
            b = sd_t[kb]
            sd_s[kb] = torch.cat([b[:inner][idx], b[inner:][idx]], 0).contiguous()
            cortadas += 1
        if k2 in sd_t:
            sd_s[k2] = sd_t[k2][:, idx].contiguous()   # mesmas colunas
            cortadas += 1
    herdadas -= cortadas

    Path(saida).mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd_s, "ff_mult": ff_mult, "config": cfg},
               str(Path(saida) / "student_raw.pt"))
    n_stu = sum(v.numel() for v in sd_s.values())
    print(f"[destila] {herdadas} tensores herdados intactos, {cortadas} cortados (FFN)")
    print(f"[destila] teacher {n_tea/1e9:.3f}B -> student {n_stu/1e9:.3f}B "
          f"({(1-n_stu/n_tea)*100:.1f}% menor, {n_stu*2/2**30:.2f} GB fp16)")
    print(f"[destila] gravado em {saida}")
    (Path(saida) / "plano.json").write_text(
        json.dumps({"ff_mult": ff_mult, "params_teacher": n_tea,
                    "params_student": n_stu, "base": base}, indent=2), encoding="utf-8")


def _indices_geglu(v, ff_mult: float, orig: int = 4):
    import torch
    """Quais neuronios da FFN sobrevivem. Devolve (indices, inner_original).

    O criterio e' a norma L2 combinada das duas metades do GEGLU: um neuronio so
    importa se VALOR e GATE forem relevantes juntos — olhar so a metade do valor
    manteria neuronios que o gate zera de qualquer jeito.

    Truncar pelas PRIMEIRAS linhas jogaria sinal fora ao acaso; ordenar por norma
    preserva as direcoes que mais contribuem, e o student nasce perto do teacher
    em vez de ter de reaprender do zero."""
    inner = v.shape[0] // 2
    novo = max(1, int(inner * ff_mult / orig))
    val, gate = v[:inner].float(), v[inner:].float()
    forca = val.norm(dim=1) * gate.norm(dim=1)
    idx = torch.argsort(forca, descending=True)[:novo]
    idx, _ = torch.sort(idx)              # ordem original: mantem a leitura estavel
    return idx, inner


def _corta_entrada(v, ff_mult: float, orig: int = 4):
    import torch
    """ff.net.2: [dim, inner] — reduz a ENTRADA, que e' a saida da GEGLU."""
    inner = v.shape[1]
    novo = max(1, int(inner * ff_mult / orig))
    idx = torch.argsort(v.float().norm(dim=0), descending=True)[:novo]
    idx, _ = torch.sort(idx)
    return v[:, idx].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff-mult", type=float, default=1.5)
    ap.add_argument("--base", default=r"D:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors")
    ap.add_argument("--saida", default=r"D:\Models\gsmde\backbone_student")
    ap.add_argument("--so-analise", action="store_true")
    a = ap.parse_args()
    analise()
    if not a.so_analise:
        print()
        constroi(a.ff_mult, a.base, a.saida)


if __name__ == "__main__":
    main()
