"""Catalogo de centros: le LoRAs baixados, descobre o DOMINIO de cada um e
gera os .json que o orquestrador do GSMDE ja sabe ler.

TRES ORIGENS, SEPARADAS EM DISCO
    models/specialists/    treinados aqui, para o GSMDE (c1..c7)
    centros_hf/            baixados do Hugging Face
    centros_civitai/       baixados do CivitAI
    A separacao nao e' organizacao — e' proveniencia. Um LoRA baixado foi
    treinado contra OUTRO base, com outra receita, e pode ate ser de outra
    familia. Misturar com os treinados aqui faria o orquestrador tratar como
    iguais coisas que nao sao, e impediria de responsabilizar um resultado
    ruim. Cada origem vira um cluster proprio, com seu broad_specialists_*.json.

DE ONDE SAI O DOMINIO
    1. ss_tag_frequency  — metadado do Kohya com a contagem de tags do dataset
       de treino. Medido nos 389 LoRAs do disco: ~72% tem. E' a melhor fonte,
       porque diz o que o modelo VIU, nao o que alguem escreveu na descricao.
    2. trainedWords      — vem do CivitAI (sidecar .json gravado no download).
    3. tags do model card — HF, mais fraco: e' texto livre.
    Sem nenhuma das tres, o centro entra como "dominio desconhecido" e o
    orquestrador so' o usa se o usuario pedir pelo nome.

TAG FREQUENTE != TAG QUE DEFINE
    Em quase todo LoRA de personagem as campeas sao `1girl`, `solo`,
    `looking at viewer` — presentes em metade do Danbooru e portanto inuteis
    para rotear. O que define o dominio e' a tag que e' comum AQUI e rara LA
    FORA. Isso e' TF-IDF, e a frequencia global sai do id2tags.parquet que ja
    usamos nos clusters. Sem esse cruzamento, o roteador mandaria todo prompt
    com "1girl" para todos os centros ao mesmo tempo.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\GSMDE\trainer")
CFG = ROOT / "config"
USO = ROOT / "config" / "centros_uso.json"

MODELOS = Path(r"D:\Models")

ORIGENS = {
    "hf": ROOT / "centros_hf",
    "civitai": ROOT / "centros_civitai",
    # A biblioteca pessoal: e' onde o usuario ja mantem tudo, nas categorias da
    # convencao SD.Next. Nao movemos nada daqui — so' lemos. Um LoRA continua
    # servindo ao SD.Next e vira centro do GSMDE ao mesmo tempo; duplicar 79 GB
    # para ter duas copias do mesmo peso nao faria sentido.
    "local": MODELOS / "Lora",
}

# Categorias da biblioteca que interessam ao GSMDE, e para que servem.
CATEGORIAS = {
    "Lora": "candidatos a centro",
    "Stable-diffusion": "checkpoints (backbone)",
    "UNET": "UNets soltos (backbone alternativo)",
    "VAE": "decodificadores",
    "Text-encoder": "encoders de texto",
    "text_encoders": "encoders de texto",
    "controlnet": "controle estrutural",
    "control": "controle estrutural",
    "ipadapter": "contexto global do i2i",
    "clip_vision": "visao do ip-adapter",
    "yolo": "alvos do detailer",
    "adetailer": "alvos do detailer",
    "embeddings": "embeddings textuais",
    "upscale_models": "upscalers",
    "ESRGAN": "upscalers",
    "RealESRGAN": "upscalers",
    "huggingface": "cache do hub",
    "Diffusers": "modelos em formato diffusers",
}

PESOS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")


def inventario(raiz: Path | None = None) -> dict:
    """Passa o olho na biblioteca inteira: o que existe, onde e quanto ocupa.

    So' LEITURA — nao move, nao renomeia, nao apaga. Serve para o GSMDE saber
    com que pecas pode contar (ip-adapter? yolo? quantos backbones?) em vez de
    ter caminho chutado no codigo."""
    raiz = Path(raiz or MODELOS)
    if not raiz.exists():
        return {"ok": False, "erro": f"nao achei {raiz}"}
    cats = {}
    for pasta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        if pasta.name.startswith((".", "_")):
            continue
        arqs = [f for f in pasta.rglob("*") if f.is_file() and f.suffix.lower() in PESOS]
        if not arqs:
            continue
        cats[pasta.name] = {
            "papel": CATEGORIAS.get(pasta.name, "—"),
            "n": len(arqs),
            "gb": round(sum(f.stat().st_size for f in arqs) / 1e9, 2),
            "exemplos": [f.name for f in arqs[:3]],
        }
    return {"ok": True, "raiz": str(raiz), "categorias": cats,
            "total_arquivos": sum(c["n"] for c in cats.values()),
            "total_gb": round(sum(c["gb"] for c in cats.values()), 1)}

# tags tao comuns que nao discriminam nada; ficam fora do dominio mesmo que
# sejam as mais frequentes do LoRA.
RUIDO = {
    "1girl", "1boy", "solo", "looking at viewer", "looking_at_viewer",
    "simple background", "simple_background", "white background",
    "white_background", "highres", "absurdres", "masterpiece", "best quality",
    "very aesthetic", "general", "sensitive", "long hair", "short hair",
    "black hair", "blush", "smile", "upper body", "upper_body", "standing",
    "closed mouth", "closed_mouth", "breasts", "bangs",
}


def _norm(t: str) -> str:
    return t.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Leitura do dominio
# ---------------------------------------------------------------------------
def ler_metadados(caminho: Path) -> dict:
    """Metadados do safetensors sem carregar tensor nenhum."""
    try:
        from safetensors import safe_open
        with safe_open(str(caminho), framework="pt") as f:
            return f.metadata() or {}
    except Exception:
        return {}


def ler_dominio(caminho: Path) -> dict:
    """Tags que o modelo conhece + de onde vieram."""
    caminho = Path(caminho)
    meta = ler_metadados(caminho)
    tags: Counter = Counter()
    fonte = ""

    bruto = meta.get("ss_tag_frequency")
    if bruto:
        try:
            for _pasta, dic in json.loads(bruto).items():
                if isinstance(dic, dict):
                    for t, n in dic.items():
                        tags[_norm(t)] += int(n)
            if tags:
                fonte = "ss_tag_frequency"
        except Exception:
            pass

    # sidecar gravado no download (trainedWords do CivitAI / tags do card HF)
    lado = caminho.with_suffix(".gsmde.json")
    extra = {}
    if lado.exists():
        try:
            extra = json.loads(lado.read_text(encoding="utf-8"))
        except Exception:
            extra = {}
        if not tags:
            for t in (extra.get("trainedWords") or extra.get("tags") or []):
                tags[_norm(t)] += 1
            if tags:
                fonte = "sidecar"

    return {
        "arquivo": caminho.name,
        "caminho": str(caminho),
        "gb": round(caminho.stat().st_size / 1e9, 3) if caminho.exists() else 0,
        "tags_brutas": dict(tags.most_common(60)),
        "fonte_dominio": fonte or "desconhecida",
        "rank": _int(meta.get("ss_network_dim")),
        "n_imagens_treino": _int(meta.get("ss_num_train_images")),
        "base_treino": meta.get("ss_sd_model_name") or extra.get("baseModel") or "",
        "trigger": extra.get("trainedWords") or [],
        "titulo": extra.get("nome") or meta.get("ss_output_name") or caminho.stem,
        "origem_remota": extra.get("origem") or "",
    }


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def freq_global(parquet: Path | None = None) -> dict:
    """Contagem global de cada tag no dump — o 'documento' do TF-IDF.
    Cacheada em disco: varrer 9M linhas por causa de um LoRA seria absurdo."""
    cache = CFG / "tag_freq_global.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    import pandas as pd
    p = parquet or (ROOT / "id2tags.parquet")
    cont: Counter = Counter()
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(p))
    for b in pf.iter_batches(batch_size=500_000, columns=["tag_string"]):
        for s in b.to_pandas()["tag_string"].to_numpy():
            if isinstance(s, str):
                cont.update(s.split())
    d = dict(cont)
    cache.write_text(json.dumps(d), encoding="utf-8")
    return d


def destacar(tags_brutas: dict, glob: dict, total_posts: int = 9_113_285,
             quantas: int = 12) -> list:
    """As tags que DEFINEM o centro: comuns aqui, raras la' fora (TF-IDF).

    Sem isto o dominio de qualquer LoRA de personagem seria
    ['1girl','solo','looking_at_viewer'] — identico ao de todos os outros, e o
    roteador nao teria como escolher."""
    if not tags_brutas:
        return []
    maior = max(tags_brutas.values()) or 1
    pontos = []
    for t, n in tags_brutas.items():
        if t in RUIDO or len(t) < 2:
            continue
        tf = n / maior
        # tag ausente do dump e' trigger word inventada: idf maximo, e' o que
        # melhor identifica o LoRA.
        g = glob.get(t, 0)
        idf = math.log(total_posts / (1 + g))
        pontos.append((tf * idf, t, n, g))
    pontos.sort(reverse=True)
    return [{"tag": t, "no_lora": n, "no_dump": g, "peso": round(p, 2)}
            for p, t, n, g in pontos[:quantas]]


# ---------------------------------------------------------------------------
# Uso: o que fica perto da raiz
# ---------------------------------------------------------------------------
MEIA_VIDA_DIAS = 21


def carregar_uso() -> dict:
    if USO.exists():
        try:
            return json.loads(USO.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def registrar_uso(nome: str) -> dict:
    """Chamado quando um centro e' efetivamente carregado numa geracao."""
    u = carregar_uso()
    e = u.setdefault(nome, {"n": 0, "ultimo": 0, "pontos": 0.0})
    agora = time.time()
    e["pontos"] = _decair(e["pontos"], e["ultimo"], agora) + 1.0
    e["n"] += 1
    e["ultimo"] = agora
    USO.parent.mkdir(parents=True, exist_ok=True)
    USO.write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8")
    return e


def _decair(pontos: float, ultimo: float, agora: float) -> float:
    """Decaimento exponencial: uso recente pesa mais que uso antigo.

    E' o que faz o ranking 'girar' — um centro que voce usou muito ano passado
    e nunca mais afunda sozinho, sem precisar de rotacao explicita."""
    if not ultimo or pontos <= 0:
        return 0.0
    dias = (agora - ultimo) / 86400.0
    return pontos * (0.5 ** (dias / MEIA_VIDA_DIAS))


def ranking(agora: float | None = None) -> list:
    agora = agora or time.time()
    u = carregar_uso()
    itens = [{"nome": k, "pontos": round(_decair(v["pontos"], v["ultimo"], agora), 3),
              "n": v["n"], "ultimo": v["ultimo"]} for k, v in u.items()]
    itens.sort(key=lambda x: -x["pontos"])
    return itens


def preload(orcamento_gb: float, tamanhos: dict) -> dict:
    """Quais centros manter residentes, do mais quente para o mais frio, ate
    encher o orcamento.

    NOTA HONESTA SOBRE O CUSTO: achar o centro no indice nunca foi o gargalo —
    sao centenas de entradas, qualquer estrutura resolve em microssegundos.
    O que custa e' ler 200-600 MB do disco e subir para a VRAM. Por isso o
    ganho real esta AQUI, em ja ter os quentes na memoria, e nao no formato da
    arvore de busca."""
    escolhidos, soma = [], 0.0
    for it in ranking():
        gb = tamanhos.get(it["nome"], 0)
        if not gb or soma + gb > orcamento_gb:
            continue
        escolhidos.append(it["nome"])
        soma += gb
    return {"preload": escolhidos, "gb": round(soma, 2),
            "orcamento_gb": orcamento_gb}


# ---------------------------------------------------------------------------
# Geracao dos .json que o orquestrador le
# ---------------------------------------------------------------------------
def indexar(origem: str, com_tfidf: bool = True) -> dict:
    """Varre a pasta da origem e devolve um centro por .safetensors."""
    pasta = ORIGENS.get(origem)
    if pasta is None:
        return {"ok": False, "erro": f"origem desconhecida: {origem}"}
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(pasta.rglob("*.safetensors"))
    glob = freq_global() if (com_tfidf and arquivos) else {}

    centros = []
    for a in arquivos:
        d = ler_dominio(a)
        d["dominio"] = destacar(d["tags_brutas"], glob) if glob else []
        d["nome"] = re.sub(r"[^a-z0-9_]+", "_", a.stem.lower()).strip("_")
        centros.append(d)
    return {"ok": True, "origem": origem, "pasta": str(pasta),
            "n": len(centros), "centros": centros}


def gerar_cluster(origem: str) -> dict:
    """Escreve broad_specialists_<origem>.json + specialists_<origem>/*.json,
    no mesmo formato dos clusters treinados, para o roteador enxergar sem
    precisar de codigo novo."""
    idx = indexar(origem)
    if not idx.get("ok"):
        return idx
    centros = idx["centros"]
    if not centros:
        return {"ok": True, "origem": origem, "n": 0,
                "aviso": f"nenhum .safetensors em {idx['pasta']}"}

    grupos, papeis, relatorio = {}, {}, {}
    dsdir = ROOT / f"specialists_{origem}"
    dsdir.mkdir(parents=True, exist_ok=True)

    for c in centros:
        tags = [d["tag"] for d in c["dominio"]]
        if not tags:
            tags = list(c["tags_brutas"])[:8]
        nome = c["nome"]
        grupos[nome] = tags
        papeis[nome] = (f"{c['titulo']} (baixado de {origem}"
                        + (f", base {c['base_treino']}" if c["base_treino"] else "")
                        + ")")
        relatorio[nome] = {"fonte_dominio": c["fonte_dominio"], "gb": c["gb"],
                           "rank": c["rank"], "trigger": c["trigger"],
                           "n_imagens_treino": c["n_imagens_treino"]}
        # ficha por centro, no formato que o resto da pipeline ja le
        (dsdir / f"{nome}.json").write_text(json.dumps({
            "especialista": nome, "macro": origem, "anchor_tag": nome,
            "anchor_tags": tags, "arquivo": c["caminho"],
            "externo": True, "origem": origem,
            "base_treino": c["base_treino"], "trigger": c["trigger"],
            "n_imagens": c["n_imagens_treino"], "rank_sugerido": c["rank"],
            "dominio": c["dominio"], "refs": [],
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    doc = {
        "cluster": origem,
        "papel": f"centros BAIXADOS de {origem} (nao treinados aqui)",
        # nivel 2: nem generalistas como person (1), nem detalhe fino (3).
        # Externos entram com prioridade media e cedem lugar aos treinados
        # quando o orcamento de VRAM aperta.
        "nivel_roteador": 2,
        "externo": True,
        "fonte": f"{idx['pasta']} (varredura automatica)",
        "gerado_em": time.strftime("%Y-%m-%d %H:%M"),
        "papeis": papeis, "grupos": grupos, "relatorio": relatorio,
    }
    saida = CFG / f"broad_specialists_{origem}.json"
    saida.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    sem_dominio = [c["nome"] for c in centros if not c["dominio"]]
    return {"ok": True, "origem": origem, "n": len(centros),
            "config": str(saida), "datasets": str(dsdir),
            "sem_dominio": sem_dominio,
            "com_dominio": len(centros) - len(sem_dominio)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="indexa centros baixados e gera os json")
    ap.add_argument("origem", nargs="?", default="todas",
                    choices=["hf", "civitai", "todas"])
    a = ap.parse_args()
    alvos = list(ORIGENS) if a.origem == "todas" else [a.origem]
    for o in alvos:
        r = gerar_cluster(o)
        print(json.dumps(r, ensure_ascii=False, indent=1))
