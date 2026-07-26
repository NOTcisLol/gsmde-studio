"""Fonte unica dos caminhos do GSMDE.

O PROBLEMA QUE ISTO RESOLVE
    Antes havia tres resolucoes independentes: `_pesos_gsmde` no launcher,
    `_pesos` no motor e `MODELOS` no catalogo — cada uma com sua propria lista
    de candidatos hardcoded. Elas podiam DISCORDAR: a UI mostrando um caminho
    enquanto o motor lia de outro, sem nada avisando. Trocar de disco exigia
    editar quatro arquivos e torcer.

PRECEDENCIA (a primeira que existir vence)
    1. override do usuario, salvo em prefs.json  ->  ganha de tudo
    2. variavel de ambiente (GSMDE_BASE, GSMDE_SPEC, ...)
    3. deteccao automatica da raiz de modelos
    4. candidatos padrao

    O override e' o topo de proposito: se o usuario apontou um lugar, o
    programa nao tem por que achar que sabe mais.

DERIVACAO
    Quase tudo pende da raiz de modelos (estilo SD.Next: Stable-diffusion/,
    Lora/, VAE/, yolo/ ...). Definir a raiz ja acerta as filhas; cada uma pode
    ser sobrescrita individualmente quando o usuario tem algo fora do padrao.
"""
from __future__ import annotations

import json
import os
import string
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parent
PREFS = LAUNCHER / "prefs.json"

# subpastas que caracterizam uma raiz de modelos no padrao SD.Next/ComfyUI.
# Quanto mais dessas existirem, mais provavel que a pasta seja a raiz certa.
MARCAS = ["Stable-diffusion", "Lora", "VAE", "embeddings", "controlnet",
          "ESRGAN", "RealESRGAN", "text_encoders", "Text-encoder", "clip_vision",
          "ipadapter", "yolo", "adetailer", "upscale_models", "UNET"]

# nome -> (tipo, subcaminho relativo a raiz de modelos, papel)
# subcaminho None = nao deriva da raiz, tem candidatos proprios.
ESQUEMA = {
    "modelos_raiz":  ("pasta", None,               "raiz da biblioteca (estilo SD.Next)"),
    "base":          ("arquivo", "Stable-diffusion", "checkpoint SDXL usado como backbone"),
    "yolo":          ("pasta", "yolo",             "modelos de deteccao do detailer"),
    "loras":         ("pasta", "Lora",             "LoRAs — candidatas a centro"),
    "especialistas": ("pasta", None,               "centros treinados do GSMDE"),
    "centros_hf":    ("pasta", None,               "centros baixados do HuggingFace"),
    "centros_civitai": ("pasta", None,             "centros baixados do CivitAI"),
    "saidas":        ("pasta", None,               "imagens geradas"),
    "saidas_intermed": ("pasta", None,             "passos intermediarios / debug"),
    "treino_saidas": ("pasta", None,               "logs e relatorios de treino"),
}

PADROES = {
    "modelos_raiz": [r"D:\Models", r"G:\Models", r"C:\Models"],
    "base": [r"G:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors",
             r"D:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors"],
    # ORDEM IMPORTA: o canonico primeiro. A copia em G: continua no disco e,
    # estando antes, sequestrava a resolucao inteira.
    "especialistas": [r"D:\Models\gsmde\specialists",
                      r"G:\Trainer_v13\models\specialists",
                      r"D:\Trainer_v13\models\specialists"],
    "centros_hf": [r"D:\Trainer_v13\centros_hf"],
    "centros_civitai": [r"D:\Trainer_v13\centros_civitai"],
    "saidas": [str(LAUNCHER.parent / "outputs" / "gsmde")],
    "saidas_intermed": [str(LAUNCHER.parent / "outputs" / "gsmde" / "intermed")],
    "treino_saidas": [r"D:\Trainer_outputs"],
}

ENV = {"base": "GSMDE_BASE", "especialistas": "GSMDE_SPEC",
       "modelos_raiz": "GSMDE_MODELS", "saidas": "GSMDE_OUT",
       "yolo": "GSMDE_YOLO"}


# ---------------------------------------------------------------------------
def _prefs() -> dict:
    try:
        return json.loads(PREFS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def overrides() -> dict:
    return (_prefs().get("caminhos") or {})


def salvar_overrides(novos: dict) -> dict:
    """Grava so' o que o usuario definiu. Campo vazio = volta ao automatico,
    em vez de gravar string vazia e quebrar a resolucao."""
    p = _prefs()
    atual = p.get("caminhos") or {}
    for k, v in (novos or {}).items():
        if k not in ESQUEMA:
            continue
        v = (v or "").strip()
        if v:
            atual[k] = v
        else:
            atual.pop(k, None)
    p["caminhos"] = atual
    PREFS.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return resolver()


def pontuar_raiz(p: Path) -> int:
    """Quantas marcas de biblioteca SD.Next a pasta tem."""
    try:
        if not p.is_dir():
            return 0
        nomes = {x.name.lower() for x in p.iterdir() if x.is_dir()}
    except OSError:
        return 0
    return sum(1 for m in MARCAS if m.lower() in nomes)


def detectar_raizes(min_marcas: int = 3) -> list:
    """Procura pastas de modelos plausiveis nos discos locais.

    Varre so' a raiz de cada disco e um nivel abaixo — o suficiente p/ achar
    D:\\Models ou C:\\IA\\Models, sem sair varrendo o disco inteiro (o que
    demoraria minutos e acharia lixo)."""
    achados = []
    for letra in string.ascii_uppercase:
        disco = Path(f"{letra}:\\")
        if not disco.exists():
            continue
        candidatos = [disco / "Models", disco / "models", disco / "AI" / "Models"]
        try:
            for filho in disco.iterdir():
                if filho.is_dir() and not filho.name.startswith(("$", ".")):
                    candidatos.append(filho / "models")
                    candidatos.append(filho / "Models")
        except OSError:
            pass
        for c in candidatos:
            n = pontuar_raiz(c)
            if n >= min_marcas and not any(a["caminho"] == str(c) for a in achados):
                achados.append({"caminho": str(c), "marcas": n,
                                "gb": _gb_rapido(c)})
    achados.sort(key=lambda x: -x["marcas"])
    return achados


def _gb_rapido(p: Path) -> float:
    """Tamanho aproximado: so' pesos, sem descer em tudo."""
    try:
        tot = sum(f.stat().st_size for f in p.rglob("*.safetensors"))
        return round(tot / 1e9, 1)
    except OSError:
        return 0.0


def _primeiro_existente(cands) -> str:
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return str(cands[-1]) if cands else ""


def resolver() -> dict:
    """Estado atual de cada caminho, com a origem da decisao — para a UI poder
    dizer POR QUE esta usando aquele lugar, em vez de so' mostrar o caminho."""
    ov = overrides()
    raiz = None
    out = {}

    # a raiz vem primeiro: as filhas derivam dela
    for nome, (tipo, sub, papel) in ESQUEMA.items():
        origem, valor = "", ""
        if ov.get(nome):
            valor, origem = ov[nome], "definido por voce"
        elif ENV.get(nome) and os.environ.get(ENV[nome]):
            valor, origem = os.environ[ENV[nome]], f"variavel {ENV[nome]}"
        elif sub and raiz:
            cand = Path(raiz) / sub
            if tipo == "arquivo":
                # p/ arquivo a subpasta e' onde procurar; mantem o padrao se
                # o arquivo nomeado nao estiver la'
                valor = _primeiro_existente(
                    [p for p in PADROES.get(nome, []) if Path(p).parent == cand]
                    + PADROES.get(nome, []))
                origem = "padrao"
            else:
                valor, origem = str(cand), "derivado da raiz"
        else:
            valor = _primeiro_existente(PADROES.get(nome, []))
            origem = "detectado" if valor and Path(valor).exists() else "padrao"

        if nome == "modelos_raiz":
            if not (ov.get(nome) or os.environ.get(ENV.get(nome, ""), "")):
                achados = detectar_raizes()
                if achados:
                    valor, origem = achados[0]["caminho"], \
                        f"detectado ({achados[0]['marcas']} marcas)"
            raiz = valor

        p = Path(valor) if valor else None
        out[nome] = {
            "caminho": valor, "papel": papel, "tipo": tipo, "origem": origem,
            "existe": bool(p and p.exists()),
            "editavel": True,
            "disco": (str(p.drive).rstrip(":").upper() if p and p.drive else "?"),
        }
    return out


def diagnostico() -> dict:
    r = resolver()
    faltando = [k for k, v in r.items() if not v["existe"]]
    return {"caminhos": r, "faltando": faltando,
            "raizes_plausiveis": detectar_raizes(),
            "ok": not faltando}


if __name__ == "__main__":
    print(json.dumps(diagnostico(), ensure_ascii=False, indent=1))
