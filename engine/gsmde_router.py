"""
Roteador semantico do GSMDE (Secao 5 do doc) — o categorizador AO CONTRARIO.

O categorizador faz tag -> nicho, p/ montar dataset. Aqui invertemos: le o prompt,
acha as tags do lexico dentro dele e ativa os nichos correspondentes. Pedir ao
usuario que digite "person:girl,woman;scenary:park,trees" e faze-LO de roteador —
com 6 nichos ja e' chato; com 256 e' impossivel. O doc sempre disse que o
codificador de texto e' um "mecanismo de roteamento topologico", nao um vetor de
forca bruta: isto e' a Fase 3 do roteiro (13.5), na versao lexica.

O ACHADO QUE SIMPLIFICA TUDO
A palavra do prompt que casou com a tag JA E' a ancora espacial que a mascara da
Secao 7.2 precisa (token -> pixel). Entao rotear e mascarar sao o mesmo ato: o
roteador devolve "nome:palavras" pronto, sem inventar nada.

REGIONAL vs GLOBAL sai do macro, nao de regra ad-hoc: 'styles' e 'camera' nao tem
territorio (sao atributos da tela toda); o resto e' regiao. E' a distincao de 11.3,
que a medicao de 12.3.2 mostrou ser a mesma coisa que eixo-vs-vizinho.

VRAM manda no teto: cada nicho custa 709MB + uma paginacao por passo. Por isso
--max-centros, ordenado por forca do casamento.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

CFG = Path(r"D:\GSMDE\trainer\config")
# Migrado p/ a biblioteca em 2026-07-26; o caminho antigo fica como reserva.
SPEC = next((p for p in (Path(r"D:\Models\gsmde\specialists"),
                         Path(r"D:\GSMDE\ckpts\models\specialists")) if p.exists()),
            Path(r"D:\Models\gsmde\specialists"))

# Macros sem territorio: aplicam na tela toda, sem mascara (11.3).
MACROS_GLOBAIS = {"styles", "camera"}
NICHOS_GLOBAIS = {"style", "camera"}

# O prompt fala humano; o lexico fala danbooru. Ponte minima e explicita —
# so o que o usuario realmente escreve. Nao e' um tradutor generico.
SINONIMOS = {
    "girl": "1girl", "woman": "1girl", "female": "1girl", "lady": "1girl",
    "boy": "1boy", "man": "1boy", "male": "1boy", "guy": "1boy",
    "alone": "solo", "sky": "sky", "clouds": "cloud", "trees": "tree",
    "sea": "ocean", "beach": "beach", "flowers": "flower", "grass": "grass",
    "sun": "sunlight", "sunny": "day", "night": "night", "evening": "sunset",
    "smiling": "smile", "smiles": "smile", "happy": "smile", "sad": "crying",
    "sitting": "sitting", "standing": "standing", "walks": "walking",
    "running": "running", "lying": "lying", "portrait": "portrait",
    "closeup": "close-up", "fullbody": "full_body", "landscape": "scenery",
    "city": "city", "buildings": "building", "mountains": "mountain",
    "dog": "dog", "cat": "cat", "dogs": "dog", "cats": "cat", "bird": "bird",
    "birds": "bird", "horse": "horse", "wolf": "wolf", "fox": "fox",
    "volumetric": "light_rays", "backlit": "backlighting", "glow": "glowing",
    "lit": "sunlight", "bedroom": "bedroom", "kitchen": "kitchen",
    "bathroom": "bathroom", "classroom": "classroom", "street": "street",
    "forest": "forest", "desert": "desert", "snowy": "snow", "underwater": "underwater",
    "park": "tree", "garden": "flower", "notes": "sparkle", "note": "sparkle",
    # --- ponte p/ os clusters c6/c7 (texturas e anatomia fina) ---
    # Esses centros foram montados com tags danbooru precisas
    # (own_hands_together, shiny_skin, brick_wall) e ninguem escreve prompt
    # assim. Medido: "hands wearing jewelry, wet skin, stone wall" nao acionava
    # 'maos', 'pele' nem 'pedra' — os tres existiam e ficavam mudos. O valor
    # aqui so' precisa ser UMA tag do grupo: ela escolhe o centro, e a ancora da
    # mascara continua sendo a palavra que o usuario escreveu.
    "hand": "own_hands_together", "hands": "own_hands_together",
    "finger": "interlocked_fingers", "fingers": "interlocked_fingers",
    "palm": "own_hands_together",
    "foot": "feet", "toe": "toes", "barefeet": "barefoot",
    "nails": "fingernails", "nail": "nail_polish", "manicure": "nail_polish",
    "mouth": "open_mouth", "teeth": "teeth", "tooth": "teeth",
    "skin": "shiny_skin", "complexion": "shiny_skin", "pores": "shiny_skin",
    "fabric": "silk", "cloth": "silk", "textile": "silk", "jeans": "denim",
    "knit": "sweater", "woolen": "sweater",
    "soaked": "wet", "damp": "wet", "droplets": "water_drop",
    "stone": "rock", "bricks": "brick_wall", "brick": "brick_wall",
    "masonry": "brick_wall", "boulder": "rock", "cobblestone": "pavement",
    "fur": "furry", "anthro": "furry", "anthropomorphic": "furry",
    "pelt": "furry", "feather": "feathers", "scale": "scales",
    "makeup": "makeup", "lips": "lips",
    "hair": None,   # sozinho nao diz nada; so vale composto (long_hair etc.)
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9_ -]", " ", (s or "").lower())


GRUPOS = [CFG / "broad_specialists.json",        # cluster 1
          CFG / "broad_specialists_c2.json",     # cluster 2
          CFG / "broad_specialists_c3.json",     # cluster 3 (objetos/particulas/manuseio/efeitos)
          CFG / "broad_specialists_c4.json",     # cluster 4 (tema externo: arq/plantas/design/multi/acoes/veiculos)
          CFG / "broad_specialists_c6.json",     # cluster 6 (texturas e superficies) — nivel 3
          CFG / "broad_specialists_c7.json",     # cluster 7 (anatomia fina e adorno) — nivel 3
          # centros BAIXADOS (nao treinados aqui). Gerados por
          # centros_catalogo.py a partir de centros_hf/ e centros_civitai/.
          # Nivel 2: cedem lugar aos treinados quando a VRAM aperta.
          CFG / "broad_specialists_local.json",   # biblioteca do usuario (D:\Models\Lora)
          CFG / "broad_specialists_hf.json",
          CFG / "broad_specialists_civitai.json",
          CFG / "broad_specialists_rules.json"]  # do rules.json, quando existir
# c5 (multiverso Miku) tem grupos vazio de proposito: roteia por NOME de personagem,
# nao por tag de cena — mecanismo separado, fora do vocabulario lexico.


# HIERARQUIA DE ESPECIFICIDADE. 1 = generico (cobre muita tela por centro),
# 3 = detalhe (refina pouca area, mas faz o que o generico nao sabe).
# Serve p/ decidir QUEM SOBREVIVE ao teto de centros:
#   poucas vagas  -> generico primeiro: com 2 slots, 'person' rende mais que
#                    'hair_filament_texture', que so melhora o fio do cabelo;
#   com folga     -> detalhe primeiro: se 'person' ja entrou, o que agrega e' o fio.
# Sem isto o desempate era alfabetico, o que nao tem relacao nenhuma com utilidade.
NIVEL_POR_CLUSTER = {"c1": 1, "c2": 2, "c3": 2, "c4": 2, "c6": 3, "c7": 3}
NIVEL_PADRAO = 2


def niveis(groups=None):
    """nicho -> nivel de especificidade, lido dos JSONs de cluster ('nivel' por
    nicho, senao o padrao do cluster)."""
    out = {}
    for arq in (groups or GRUPOS):
        try:
            d = json.loads(Path(arq).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # o c1 (broad_specialists.json) NAO tem o campo 'cluster' — cai no nome do
        # arquivo, senao 'person' e 'scenary' virariam nivel 2 e a hierarquia
        # perderia justamente a ponta generica que ela existe p/ proteger.
        cl = str(d.get("cluster", "") or "")
        if not cl:
            nome = Path(arq).stem
            cl = "c1" if nome == "broad_specialists" else nome.split("_")[-1]
        base = NIVEL_POR_CLUSTER.get(cl, NIVEL_PADRAO)
        expl = d.get("nivel_por_nicho", {}) or {}
        for nicho in (d.get("grupos", {}) or {}):
            out[nicho] = int(expl.get(nicho, base))
    return out


def vocabulario(disponiveis, groups=None, macros=CFG / "macro_groups_enriched.json"):
    """tag -> LISTA de (nicho, macro), so p/ nichos treinados.

    Lista, nao tupla: com mais de um cluster a mesma tag pertence a varios nichos.
    'forest' e' 'scenary' no c1 (cenario em geral) e 'biomes' no c2 (bioma
    especifico) — as duas leituras sao legitimas e o prompt pode querer as duas.
    Guardar so a primeira era o que fazia o roteador ignorar o c2 inteiro.
    """
    voc = {}
    arquivos = [Path(x) for x in (groups if groups is not None else GRUPOS)]
    for arq in arquivos:
        if not arq.exists():
            continue
        g = json.loads(arq.read_text(encoding="utf-8")).get("grupos", {})
        for nicho, tags in g.items():
            if nicho not in disponiveis:
                continue
            for t in tags:
                t = t.strip().lower()
                voc.setdefault(t, [])
                if not any(n == nicho for n, _ in voc[t]):
                    voc[t].append((nicho, nicho))
    if Path(macros).exists():
        m = json.loads(Path(macros).read_text(encoding="utf-8"))
        for macro, tags in m.get("lexico_danbooru", {}).items():
            for t in tags:
                t = t.strip().lower()
                if t in disponiveis:               # nicho por-tag (os 244 futuros)
                    voc.setdefault(t, [])
                    if not any(n == t for n, _ in voc[t]):
                        voc[t].append((t, macro))
                elif t not in voc:                 # tag conhecida, nicho nao treinado
                    voc[t] = [(None, macro)]
    return voc


FICHAS_EXTERNAS = [Path(r"D:\GSMDE\trainer") / f"specialists_{o}"
                   for o in ("local", "hf", "civitai")]


def disponiveis(spec_root=SPEC):
    """Centros que existem em disco: os TREINADOS aqui (<spec>/<nome>/specialist)
    mais os BAIXADOS, cujas fichas apontam um .safetensors avulso.

    Sem incluir os externos, `vocabulario()` os descartava no filtro
    `if nicho not in disponiveis` — o cluster inteiro era gerado, aparecia no
    GRUPOS e mesmo assim nunca roteava nada."""
    out = set()
    if Path(spec_root).exists():
        out |= {p.name for p in Path(spec_root).iterdir()
                if p.is_dir() and (p / "specialist.safetensors").exists()}
    for pasta in FICHAS_EXTERNAS:
        if not pasta.is_dir():
            continue
        for f in pasta.glob("*.json"):
            if f.stem == "brain_registry":
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            arq = d.get("arquivo")
            if arq and Path(arq).exists():
                out.add(f.stem)
    return out


_EXT_CACHE = set()


def externos(groups=None):
    """Nomes dos centros BAIXADOS (clusters marcados com 'externo': true)."""
    if _EXT_CACHE:
        return _EXT_CACHE
    for arq in (groups or GRUPOS):
        try:
            d = json.loads(Path(arq).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("externo"):
            _EXT_CACHE.update(d.get("grupos", {}) or {})
    return _EXT_CACHE


def rotear(prompt, disp=None, max_centros=4, voc=None):
    """prompt -> centros + as palavras do PROPRIO prompt que os ancoram."""
    disp = disp if disp is not None else disponiveis()
    voc = voc if voc is not None else vocabulario(disp)
    p = _norm(prompt)
    palavras = [w for w in re.split(r"[\s,]+", p) if w]

    achados = {}      # nicho -> {"palavras": [...], "tags": [...], "macro": m}
    diag = []

    def bate(surface, tag):
        """A ancora e' a palavra DO PROMPT (a mascara casa tokens do prompt),
        nao a tag canonica — se o texto diz 'girl', procurar '1girl' nao acha nada.
        Uma tag pode ativar nichos de clusters diferentes: ativa todos, e o teto de
        --max-centros decide quem sobra."""
        alvos = voc.get(tag) or []
        vivos = [(n, m) for n, m in alvos if n and n in disp]
        if not vivos:
            mortos = [m for n, m in alvos if not n]
            if mortos:
                diag.append(f"  '{surface}' -> tag '{tag}' [{mortos[0]}] mas o nicho nao esta treinado")
            return
        for nicho, macro in vivos:
            d = achados.setdefault(nicho, {"palavras": [], "tags": [], "macro": macro})
            if surface not in d["palavras"]:
                d["palavras"].append(surface)
            if tag not in d["tags"]:
                d["tags"].append(tag)
        alvo_txt = ", ".join(f"'{n}'" for n, _ in vivos)
        diag.append(f"  '{surface}' -> tag '{tag}' -> nicho {alvo_txt}"
                    + ("   [2 clusters]" if len(vivos) > 1 else ""))

    # n-gramas de 3..1: pega "full body" antes de "body"
    n = len(palavras)
    usadas = set()
    for tam in (3, 2, 1):
        for i in range(n - tam + 1):
            if any(j in usadas for j in range(i, i + tam)):
                continue
            surface = " ".join(palavras[i:i + tam])
            for cand in (surface.replace(" ", "_"), surface.replace(" ", "-"), surface):
                if cand in voc:
                    bate(surface, cand)
                    usadas.update(range(i, i + tam))
                    break
            else:
                if tam == 1:
                    sin = SINONIMOS.get(surface)
                    if sin and sin in voc:
                        bate(surface, sin)
                        usadas.add(i)

    # Ordem = evidencia + HIERARQUIA, e o sentido da hierarquia depende do aperto:
    #   mais candidatos que vagas -> generico primeiro (sobrevive quem cobre mais);
    #   cabe todo mundo           -> detalhe primeiro (o generico ja esta garantido).
    lv = niveis()
    ext = externos()
    aperto = len(achados) > max_centros
    # TREINADO AQUI VEM SEMPRE ANTES DO BAIXADO — primeira chave da ordenacao.
    #
    # Medido: com os 389 LoRAs da biblioteca no vocabulario, o prompt
    # "hands wearing jewelry, wet skin, stone wall" escolhia seis LoRAs externas
    # casando em tags genericas ('jewelry', 'close up') e deixava de fora
    # 'maos', 'bijuteria', 'pele', 'molhado' e 'pedra' — os centros feitos
    # exatamente para aquilo. Dois motivos se somavam: os externos entram como
    # nivel 2 (padrao) e, sob aperto, a ordem por nivel ASCENDENTE os punha na
    # frente dos c6/c7 (nivel 3); e sao 389 contra 36, entao qualquer empate
    # tende para eles pela quantidade.
    #
    # Um LoRA baixado foi treinado contra OUTRO base, com outra receita, e nao
    # passou por curadoria nenhuma. Ele pode somar quando ha vaga sobrando, mas
    # nao tem por que deslocar um centro construido para a tarefa.
    ordem = sorted(achados.items(),
                   key=lambda x: (1 if x[0] in ext else 0,
                                  lv.get(x[0], NIVEL_PADRAO) if aperto
                                  else -lv.get(x[0], NIVEL_PADRAO),
                                  -len(x[1]["tags"]), x[0]))
    regional, globais = [], []
    for nicho, d in ordem:
        if nicho in NICHOS_GLOBAIS or d["macro"] in MACROS_GLOBAIS:
            globais.append(nicho)
        else:
            regional.append((nicho, d["palavras"]))

    # teto de VRAM: cada nicho sao 709MB + 1 paginacao por passo
    cortados = []
    if len(regional) > max_centros:
        cortados = [n for n, _ in regional[max_centros:]]
        regional = regional[:max_centros]

    spec = ";".join(f"{n}:{','.join(w)}" for n, w in regional)
    return {"centros": spec, "regional": regional, "globais": globais,
            "diag": diag, "cortados": cortados,
            "nao_roteado": [w for i, w in enumerate(palavras) if i not in usadas]}


def main():
    ap = argparse.ArgumentParser(description="Roteia um prompt p/ os nichos do GSMDE")
    ap.add_argument("prompt")
    ap.add_argument("--max-centros", type=int, default=4)
    args = ap.parse_args()
    disp = disponiveis()
    r = rotear(args.prompt, disp, args.max_centros)
    print(f"prompt : {args.prompt}")
    print(f"nichos treinados: {sorted(disp)}\n")
    print("casamentos:")
    print("\n".join(r["diag"]) or "  (nenhum)")
    print(f"\nREGIONAIS: {r['centros'] or '(nenhum — cai no base puro)'}")
    print(f"GLOBAIS  : {','.join(r['globais']) or '(nenhum)'}")
    if r["cortados"]:
        print(f"CORTADOS (teto de {args.max_centros}): {r['cortados']}")
    print(f"\nsem nicho (ficam com o base): {r['nao_roteado']}")


if __name__ == "__main__":
    main()
