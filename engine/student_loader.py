"""Monta o UNet student (FFN reduzida) e prova que os centros ainda encaixam.

O diffusers nao expoe o multiplicador da FFN na config — ele e' derivado (dim*4)
dentro do FeedForward. Entao o student e' construido com a config do teacher e as
camadas da FFN sao TROCADAS por versoes menores antes de carregar os pesos. Sem
isso o load_state_dict falharia por forma incompativel.

A atencao nao e' tocada em momento nenhum: e' exatamente por isso que os LoRAs
dos 36 centros continuam encaixando.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def monta_student(caminho: str, dtype=torch.float16):
    """Le o student_raw.pt e devolve um UNet2DConditionModel com a FFN reduzida."""
    from diffusers import UNet2DConditionModel

    d = torch.load(str(Path(caminho) / "student_raw.pt"), map_location="cpu",
                   weights_only=False)
    sd, cfg = d["state_dict"], d["config"]
    u = UNet2DConditionModel.from_config(cfg)

    # Redimensiona cada FeedForward para bater com os pesos ja cortados.
    trocadas = 0
    for nome, mod in u.named_modules():
        proj = f"{nome}.net.0.proj.weight"
        saida = f"{nome}.net.2.weight"
        if proj in sd and saida in sd:
            dois_inner, dim = sd[proj].shape          # GEGLU: [2*inner, dim]
            inner = dois_inner // 2
            tem_bias_p = f"{nome}.net.0.proj.bias" in sd
            tem_bias_s = f"{nome}.net.2.bias" in sd
            mod.net[0].proj = nn.Linear(dim, dois_inner, bias=tem_bias_p)
            mod.net[2] = nn.Linear(inner, sd[saida].shape[0], bias=tem_bias_s)
            trocadas += 1

    faltando, sobrando = u.load_state_dict(sd, strict=False)
    if faltando or sobrando:
        raise RuntimeError(f"pesos nao bateram: faltando={len(faltando)} "
                           f"sobrando={len(sobrando)}")
    u = u.to(dtype)
    n = sum(p.numel() for p in u.parameters())
    print(f"[student] {trocadas} FFNs redimensionadas | {n/1e9:.3f}B params "
          f"({n * 2 / 2**30:.2f} GB fp16)", flush=True)
    return u


def testa_lora(u, centro: str = "person",
               raiz: str = r"D:\Models\gsmde\specialists") -> bool:
    """Carrega um centro no student. Se as formas da atencao mudassem, isto
    estouraria — e' a prova de que a estrategia (cortar FFN, poupar atencao)
    preserva os 36 centros."""
    from safetensors.torch import load_file
    cam = Path(raiz) / centro / "specialist.safetensors"
    if not cam.exists():
        print(f"[student] centro '{centro}' nao encontrado", flush=True)
        return False
    st = load_file(str(cam))
    try:
        u.load_lora_adapter(st, prefix="unet", adapter_name=centro)
        print(f"[student] centro '{centro}' CARREGOU: {len(st)} chaves encaixaram",
              flush=True)
        return True
    except Exception as e:
        print(f"[student] centro '{centro}' FALHOU: {type(e).__name__}: {e}",
              flush=True)
        return False


if __name__ == "__main__":
    import sys
    raiz = sys.argv[1] if len(sys.argv) > 1 else r"D:\Models\gsmde\backbone_student"
    u = monta_student(raiz)
    print("\n=== prova de compatibilidade dos centros ===")
    ok = 0
    for c in ("person", "expression", "biomes", "design", "maos"):
        ok += testa_lora(u, c)
    print(f"\n{ok}/5 centros encaixaram no student.")
