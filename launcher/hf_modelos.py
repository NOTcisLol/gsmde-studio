"""Baixar SDXL do Hugging Face e preparar o UNet como backbone substituto.

O QUE "SUBSTITUIR O BACKBONE SEM RETREINAR" QUER DIZER AQUI
    Os centros do GSMDE sao LoRAs que atacam `to_k/to_q/to_v/to_out.0` do UNet.
    Todo checkpoint SDXL compartilha a MESMA arquitetura de UNet, entao esses
    modulos existem em qualquer um deles e o adapter encaixa mecanicamente —
    por isso nao precisa retreinar.

    Mas encaixar nao e' o mesmo que funcionar bem. Os centros foram treinados
    contra os pesos do Illustrious; quanto mais longe o novo base estiver dele,
    mais os centros erram o alvo. Trocar por outro derivado de Illustrious ou
    NoobAI tende a ser transparente; trocar por SDXL 1.0 base ou por um Pony
    provavelmente degrada, porque o espaco latente que os centros aprenderam a
    corrigir e' outro. Isso NAO da' erro — da' resultado pior, que e' pior de
    diagnosticar. Por isso `inspecionar()` compara e avisa antes de baixar 6 GB.

DOIS FORMATOS NO HUB
    - checkpoint unico (.safetensors na raiz)  -> serve direto como GSMDE_BASE
    - formato diffusers (pastas unet/ vae/ ...) -> extraimos so' o UNet
    O segundo e' a razao do "extrai e converte": um UNet-only tem ~5 GB em vez
    dos ~6.5 GB do checkpoint inteiro, e mantem o VAE e os text encoders que
    voce ja confia, trocando so' a espinha.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import hf_auth

API = "https://huggingface.co/api"
DESTINO = Path(r"D:\Models\Stable-diffusion")

ALVOS_LORA = ("to_k", "to_q", "to_v", "to_out.0")

# Deteccao de SDXL POR FORMA, nao por nome de chave.
#
# Tentei primeiro casar o nome `add_embedding.linear_1.weight` e deu FALSO
# NEGATIVO no Illustrious: esse e' o nome do formato diffusers, e num checkpoint
# unico (formato LDM) a mesma camada se chama `label_emb.0.0.weight`. O detector
# diria "nao e' SDXL" sobre um SDXL — o pior tipo de erro aqui, porque manda o
# usuario descartar um modelo bom.
#
# A forma nao depende do formato:
#   - contexto do cross-attention (attn2.to_k) = 2048 no SDXL
#     (SD1.5 = 768, SD2 = 1024). Este e' o discriminador forte.
#   - micro-conditioning (tamanho+crop) = alguma matriz [*, 2816],
#     que so' o SDXL tem, com qualquer um dos dois nomes.
CTX_SDXL = 2048
LARGURA_MICRO = 2816


def _e_sdxl(unet: dict) -> tuple[bool, dict]:
    ctx = set()
    for k, v in unet.items():
        if "attn2.to_k" in k and getattr(v, "ndim", 0) == 2:
            ctx.add(int(v.shape[1]))
    micro = any(getattr(v, "ndim", 0) == 2 and v.shape[1] == LARGURA_MICRO
                for v in unet.values())
    return (CTX_SDXL in ctx and micro), {
        "ctx_cross_attn": sorted(ctx), "tem_micro_cond": micro,
        "familia": ("SDXL" if CTX_SDXL in ctx else
                    "SD2" if 1024 in ctx else
                    "SD1.x" if 768 in ctx else "desconhecida")}


def _req(url: str, token: str = "") -> dict:
    h = {"Accept": "application/json", "User-Agent": "GSMDE-Studio/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def buscar(termo: str = "sdxl", limite: int = 25) -> dict:
    """Procura modelos. Filtra por tarefa text-to-image p/ nao trazer lixo."""
    tok, _ = hf_auth.token_ativo()
    url = f"{API}/models?" + urllib.parse.urlencode({
        "search": termo, "filter": "text-to-image", "sort": "downloads",
        "direction": -1, "limit": min(limite, 100)})
    try:
        itens = _req(url, tok)
    except Exception as e:
        return {"ok": False, "erro": str(e)}
    return {"ok": True, "itens": [{
        "id": m.get("id"), "downloads": m.get("downloads"), "likes": m.get("likes"),
        "gated": bool(m.get("gated")), "tags": (m.get("tags") or [])[:8],
    } for m in itens]}


def inspecionar(repo_id: str) -> dict:
    """Diz o que ha no repo ANTES de baixar: formato, tamanho e se e' SDXL.

    Usa a arvore de arquivos da API (nao baixa peso nenhum)."""
    tok, _ = hf_auth.token_ativo()
    try:
        info = _req(f"{API}/models/{repo_id}", tok)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return {"ok": False, "erro": "repo restrito (gated) — aceite os termos "
                                         "na pagina dele e garanta o escopo gated-repos"}
        if e.code == 404:
            return {"ok": False, "erro": "repo nao encontrado"}
        return {"ok": False, "erro": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "erro": str(e)}

    arquivos = [s.get("rfilename", "") for s in (info.get("siblings") or [])]
    unicos = [f for f in arquivos
              if f.endswith(".safetensors") and "/" not in f]
    tem_diffusers = any(f.startswith("unet/") and f.endswith(".safetensors")
                        for f in arquivos)
    unet_diffusers = [f for f in arquivos if f.startswith("unet/")
                      and f.endswith(".safetensors")]

    # SDXL costuma se anunciar nas tags e no config; se nao houver pista, avisa
    # em vez de afirmar — a checagem definitiva so' e' possivel com o arquivo.
    tags = [t.lower() for t in (info.get("tags") or [])]
    pista = any("xl" in t for t in tags) or bool(re.search(r"xl", repo_id, re.I))

    return {"ok": True, "repo": repo_id, "gated": bool(info.get("gated")),
            "formato": ("diffusers" if tem_diffusers else
                        "checkpoint_unico" if unicos else "desconhecido"),
            "checkpoints": unicos[:10], "unet_diffusers": unet_diffusers,
            "provavel_sdxl": pista, "tags": tags[:10],
            "aviso": ("" if pista else
                      "nao achei pista de SDXL nas tags — confira antes de baixar")}


def baixar(repo_id: str, arquivo: str, destino: Path | None = None,
           progresso=None) -> dict:
    """Baixa UM arquivo do repo com barra de progresso (stream, sem carregar
    tudo na RAM). Retoma nao: se cair, refaz — mais simples e o Hub e' rapido."""
    tok, _ = hf_auth.token_ativo()
    destino = Path(destino or DESTINO)
    destino.mkdir(parents=True, exist_ok=True)
    alvo = destino / Path(arquivo).name
    url = f"https://huggingface.co/{repo_id}/resolve/main/{arquivo}"
    h = {"User-Agent": "GSMDE-Studio/1.0"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    tmp = alvo.with_suffix(alvo.suffix + ".parcial")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                    timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            feito = 0
            with tmp.open("wb") as f:
                while True:
                    bloco = r.read(1 << 20)
                    if not bloco:
                        break
                    f.write(bloco)
                    feito += len(bloco)
                    if progresso:
                        progresso(feito, total)
        tmp.replace(alvo)                      # so' vira o arquivo final inteiro
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"ok": False, "erro": str(e)}
    return {"ok": True, "caminho": str(alvo), "bytes": alvo.stat().st_size}


def extrair_unet(entrada: str, saida: str | None = None) -> dict:
    """Extrai SO' o UNet de um checkpoint SDXL e grava um arquivo separado.

    Aceita tanto checkpoint unico (prefixo `model.diffusion_model.`) quanto
    UNet ja isolado do formato diffusers. Nao converte arquitetura — SDXL e'
    SDXL; o trabalho aqui e' escolher as chaves certas e checar que os alvos
    dos LoRAs estao presentes, que e' o que garante que os centros encaixam."""
    from safetensors.torch import load_file, save_file

    ent = Path(entrada)
    if not ent.exists():
        return {"ok": False, "erro": f"nao achei {ent}"}
    try:
        pesos = load_file(str(ent))
    except Exception as e:
        return {"ok": False, "erro": f"falha ao ler safetensors: {e}"}

    PREFIXO = "model.diffusion_model."
    if any(k.startswith(PREFIXO) for k in pesos):
        unet = {k[len(PREFIXO):]: v for k, v in pesos.items() if k.startswith(PREFIXO)}
        origem = "checkpoint_unico"
    else:
        # ja e' um unet solto (diffusers) — mantem como esta
        unet = pesos
        origem = "unet_isolado"

    if not unet:
        return {"ok": False, "erro": "nenhum tensor de UNet encontrado"}

    e_sdxl, forma = _e_sdxl(unet)
    alvos = {a: sum(1 for k in unet if f".{a}." in k or k.endswith(f".{a}.weight"))
             for a in ALVOS_LORA}
    faltando = [a for a, n in alvos.items() if n == 0]

    params = sum(v.numel() for v in unet.values())
    sai = Path(saida) if saida else ent.with_name(ent.stem + "_unet.safetensors")
    try:
        save_file(unet, str(sai), metadata={
            "formato": "unet_sdxl", "origem": str(ent), "tipo_origem": origem,
            "n_tensores": str(len(unet)), "n_params": str(params)})
    except Exception as e:
        return {"ok": False, "erro": f"falha ao gravar: {e}"}

    return {"ok": True, "caminho": str(sai), "origem": origem,
            "e_sdxl": e_sdxl, "forma": forma, "n_tensores": len(unet),
            "params_bi": round(params / 1e9, 2),
            "gb": round(sai.stat().st_size / 1e9, 2),
            "alvos_lora": alvos, "faltando": faltando,
            "compativel": bool(e_sdxl and not faltando),
            "aviso": ("" if e_sdxl else
                      f"nao parece SDXL (familia detectada: {forma['familia']}, "
                      f"contexto {forma['ctx_cross_attn']}): os centros nao encaixam")}


def comparar_com_base(unet_novo: str, base_atual: str) -> dict:
    """Distancia grosseira entre o UNet novo e o base com que os centros foram
    treinados. Nao e' medida de qualidade — e' um sinal de quao longe voce esta
    indo. Divergencia alta nao quebra nada, mas os centros vao mirar torto."""
    import torch
    from safetensors.torch import load_file

    try:
        a = load_file(unet_novo)
        b_todo = load_file(base_atual)
    except Exception as e:
        return {"ok": False, "erro": str(e)}
    P = "model.diffusion_model."
    b = ({k[len(P):]: v for k, v in b_todo.items() if k.startswith(P)}
         if any(k.startswith(P) for k in b_todo) else b_todo)

    comuns = [k for k in a if k in b and a[k].shape == b[k].shape]
    if not comuns:
        return {"ok": False, "erro": "nenhuma chave em comum — arquiteturas diferentes"}
    # amostra as camadas de atencao, que sao as que os centros tocam
    alvo = [k for k in comuns if any(f".{t}." in k for t in ALVOS_LORA)][:400]
    difs = []
    for k in alvo:
        x, y = a[k].float().flatten(), b[k].float().flatten()
        n = x.norm() * y.norm()
        if n > 0:
            difs.append(float(1 - torch.dot(x, y) / n))
    med = sum(difs) / len(difs) if difs else None
    if med is None:
        veredito = "indeterminado"
    elif med < 0.05:
        veredito = "muito proximo — troca deve ser transparente"
    elif med < 0.20:
        veredito = "proximo — provavelmente ok, teste um centro antes"
    elif med < 0.45:
        veredito = "distante — espere degradacao nos centros"
    else:
        veredito = "muito distante — os centros vao mirar torto"
    return {"ok": True, "chaves_comuns": len(comuns), "amostradas": len(difs),
            "divergencia_media": round(med, 4) if med is not None else None,
            "veredito": veredito}
