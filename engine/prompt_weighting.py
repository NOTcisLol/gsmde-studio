"""Parser de enfase/atencao estilo A1111/SD.Next para o GSMDE (compartilhado pelos
3 encoders: T5 orquestrador + CLIP-L + CLIP-G).

Sintaxe suportada (aninhavel, exceto a quebra de linha/BREAK):
  (prompt)          -> peso *1.1 por nivel de parentese
  [prompt]          -> peso /1.1 por nivel de colchete
  (prompt:1.3)      -> peso especifico
  \\( \\) \\[ \\]    -> parentese/colchete literais (escape)
  {A | B | C}       -> seletor aleatorio (semeado); ANINHAVEL
  BREAK  ou  \\n     -> fecha o bloco de 75 tokens uteis e comeca outro (so CLIP)

Uso tipico:
  from prompt_weighting import resolve_wildcards, parse_prompt_attention, \\
       clean_text, get_weighted_sdxl_embeddings
  # roteador (lexical/T5) le clean_text(prompt, seed); CLIP usa embeddings pesados.
"""
from __future__ import annotations
import random
import re

import torch

# ------------------------------------------------------------------ 1. wildcards
_WILD = re.compile(r"\{([^{}]*)\}")   # innermost {...} sem chaves aninhadas dentro


def resolve_wildcards(text: str, rng: random.Random | None = None) -> str:
    """Resolve {A|B|C} escolhendo uma opcao (aninhavel: resolve de dentro p/ fora).
    rng semeado -> reproduzivel (mesma seed => mesma escolha)."""
    rng = rng or random
    guard = 0
    while "{" in text and guard < 100:
        guard += 1
        m = _WILD.search(text)
        if not m:
            break
        opts = m.group(1).split("|")
        text = text[:m.start()] + rng.choice(opts).strip() + text[m.end():]
    return text


# ------------------------------------------------------------------ 2. enfase A1111
_re_attention = re.compile(r"""
\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|]|[^\\()\[\]:]+|:
""", re.X)
_re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str):
    """text -> [[trecho, peso], ...]. Aninhamento via pilha de multiplicadores.
    'BREAK' vira um marcador ['BREAK', -1]."""
    res, round_b, square_b = [], [], []
    RB, SB = 1.1, 1 / 1.1

    def mul(start, m):
        for i in range(start, len(res)):
            res[i][1] *= m

    for mm in _re_attention.finditer(text):
        t, w = mm.group(0), mm.group(1)
        if t.startswith("\\"):
            res.append([t[1:], 1.0])
        elif t == "(":
            round_b.append(len(res))
        elif t == "[":
            square_b.append(len(res))
        elif w is not None and round_b:
            mul(round_b.pop(), float(w))
        elif t == ")" and round_b:
            mul(round_b.pop(), RB)
        elif t == "]" and square_b:
            mul(square_b.pop(), SB)
        else:
            for i, part in enumerate(_re_break.split(t)):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])
    for p in round_b:
        mul(p, RB)
    for p in square_b:
        mul(p, SB)
    if not res:
        res = [["", 1.0]]
    # funde vizinhos de mesmo peso
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0] != "BREAK" and res[i + 1][0] != "BREAK":
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def clean_text(text: str, rng: random.Random | None = None) -> str:
    """Resolve wildcards e REMOVE a sintaxe de peso -> texto plano p/ o roteador
    (lexical ou T5). '(steampunk:1.3)' vira 'steampunk'; BREAK/\\n vira espaco."""
    t = resolve_wildcards(text, rng)
    parts = parse_prompt_attention(t.replace("\n", " BREAK "))
    return " ".join(p[0] for p in parts if p[0] != "BREAK").strip()


# ------------------------------------------------------------------ 3. tokens+pesos p/ CLIP
def _tokens_and_weights(pairs, tokenizer, focus_ids=None, focus_w=1.0, context_w=1.0):
    """[[texto,peso]] -> ([ids...], [pesos...]) com bos/eos, blocos de 75 nos BREAK.
    focus_ids: se dado, cada token recebe *focus_w se for foco do centro, senao
    *context_w (o peso do prompt global CAI no centro, o foco SOBE; backbone fica 1.0).
    bos/eos/pad ficam em 1.0 (estrutura)."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    id_chunks, w_chunks = [[]], [[]]
    for text, weight in pairs:
        if text == "BREAK":
            id_chunks.append([]); w_chunks.append([]); continue
        # verbose=False: o transformers avisa "sequence length is longer than 77 —
        # will result in indexing errors" toda vez que ve um texto longo sem
        # truncation. Aqui NAO ha erro nenhum: e' exatamente o passo anterior ao
        # corte em blocos de 75. O aviso so assustava.
        ids = tokenizer(text, truncation=False, add_special_tokens=False,
                        verbose=False).input_ids
        for tid in ids:
            if len(id_chunks[-1]) >= 75:            # bloco cheio -> novo bloco
                id_chunks.append([]); w_chunks.append([])
            w = weight
            if focus_ids is not None:
                w *= focus_w if tid in focus_ids else context_w
            id_chunks[-1].append(tid); w_chunks[-1].append(w)
    # embrulha cada bloco com bos/eos e pad ate 77
    for ids, ws in zip(id_chunks, w_chunks):
        ids.insert(0, bos); ws.insert(0, 1.0)
        ids.append(eos); ws.append(1.0)
        while len(ids) < 77:
            ids.append(eos); ws.append(1.0)
    return id_chunks, w_chunks, bos, eos


def _pares(texto, literal=False):
    """pares [texto,peso] p/ _tokens_and_weights.

    literal=True ignora a sintaxe de peso e trata '(x:1.3)' como TEXTO. Serve p/ o
    caso 'usuario desligou os pesos mas o prompt passa de 75 tokens': ele ainda
    precisa de chunking, e reinterpretar os parenteses como enfase mudaria o
    sentido do prompt sem ele ter pedido."""
    if not literal:
        return parse_prompt_attention(texto)
    out = []
    for i, parte in enumerate(re.split(r"\bBREAK\b", texto)):
        if i:
            out.append(["BREAK", 1.0])
        if parte.strip():
            out.append([parte.strip(), 1.0])
    return out or [["", 1.0]]


def conta_tokens(tokenizer, texto, literal=False):
    """Quantos tokens CLIP o texto ocupa (sem bos/eos). >75 = precisa de chunking."""
    if not texto:
        return 0
    n = 0
    for t, _ in _pares(str(texto).replace("\n", " BREAK "), literal):
        if t != "BREAK" and t:
            n += len(tokenizer(t, truncation=False, add_special_tokens=False,
                               verbose=False).input_ids)
    return n


def ids_em_blocos(tokenizer, texto, literal=False):
    """Os ids NA MESMA ordem que o encoder viu (blocos de 75 embrulhados em
    bos/eos e completados ate 77). E' o mapa que o center_tokens precisa: com
    chunking, o token 80 do prompt NAO fica no indice 80 da sequencia — cada bloco
    gasta 2 posicoes com bos/eos. Indexar pela tokenizacao plana apontaria as
    mascaras p/ palavras erradas."""
    ids, _, _, _ = _tokens_and_weights(
        _pares(str(texto).replace("\n", " BREAK "), literal), tokenizer)
    return [i for bloco in ids for i in bloco]


def _encode_chunks_sdxl(pipe, id_chunks, w_chunks, device):
    """Codifica os blocos nos DOIS CLIP e aplica os pesos (metodo webui: multiplica e
    restaura a media do bloco). Retorna (prompt_embeds[1, 77*n, 2048], pooled[1,1280])."""
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    outs, pooled = [], None
    for ids, ws in zip(id_chunks, w_chunks):
        tid = torch.tensor([ids], device=device)
        wt = torch.tensor([ws], device=device, dtype=torch.float32)
        h1 = te1(tid, output_hidden_states=True).hidden_states[-2]        # [1,77,768]
        o2 = te2(tid, output_hidden_states=True)
        h2 = o2.hidden_states[-2]                                         # [1,77,1280]
        if pooled is None:
            pooled = o2.text_embeds                                       # pooled do 1o bloco
        h = torch.cat([h1, h2], dim=-1)                                   # [1,77,2048]
        # aplica peso por token, preservando a media do bloco (evita clareamento/escurecimento)
        orig_mean = h.mean(dim=(1, 2), keepdim=True)
        h = h * wt.unsqueeze(-1).to(h.dtype)
        new_mean = h.mean(dim=(1, 2), keepdim=True)
        h = h * (orig_mean / (new_mean + 1e-6))
        outs.append(h)
    return torch.cat(outs, dim=1), pooled


def get_weighted_sdxl_embeddings(pipe, prompt, neg, rng: random.Random | None = None,
                                 device=None, literal=False):
    """Substitui pipe.encode_prompt com suporte a enfase/wildcard/BREAK.
    Retorna (pe, npe, pp, npp) com o MESMO nº de blocos (pad do menor com bloco vazio).
    Aninhamento total (exceto BREAK). Wildcards resolvidos ANTES (semeado)."""
    device = device or torch.device("cpu")
    p = resolve_wildcards(prompt, rng).replace("\n", " BREAK ")
    n = resolve_wildcards(neg or "", rng).replace("\n", " BREAK ")
    pid, pw, _, _ = _tokens_and_weights(_pares(p, literal), pipe.tokenizer)
    nid, nw, bos, eos = _tokens_and_weights(_pares(n, literal), pipe.tokenizer)
    # iguala nº de blocos (CFG exige pos/neg do mesmo comprimento)
    empty = [bos] + [eos] * 76
    while len(pid) < len(nid):
        pid.append(list(empty)); pw.append([1.0] * 77)
    while len(nid) < len(pid):
        nid.append(list(empty)); nw.append([1.0] * 77)
    with torch.no_grad():
        pe, pp = _encode_chunks_sdxl(pipe, pid, pw, device)
        # O NEGATIVO NAO DEPENDE DO CENTRO. Esta funcao e' chamada uma vez por
        # centro e so' o POSITIVO muda (foco/contexto); o `nid/nw` sai identico
        # em todas. Medido: 15 centros gastaram 211s no encode, e metade disso
        # era recalcular o mesmo negativo 15 vezes na CPU.
        #
        # A chave inclui o formato do positivo porque o negativo e' PADDED ate
        # o comprimento dele (o while acima) — mesmo texto com contagem de
        # blocos diferente da tensor diferente.
        chave = (id(pipe), device_key(device), tuple(map(tuple, nid)), len(pid))
        cache = _CACHE_NEG.get(chave)
        if cache is None:
            cache = _encode_chunks_sdxl(pipe, nid, nw, device)
            if len(_CACHE_NEG) > 8:      # so' o prompt da vez importa
                _CACHE_NEG.clear()
            _CACHE_NEG[chave] = cache
        npe, npp = cache
    return pe, npe, pp, npp


_CACHE_NEG: dict = {}


def device_key(d):
    return str(d)


def get_center_weighted_embeddings(pipe, full_prompt, neg, focus_words,
                                    focus_w=1.2, context_w=0.8,
                                    rng: random.Random | None = None, device=None):
    """Condicionamento POR CENTRO (ideia do usuario): o centro le o prompt INTEIRO
    (contexto), mas com o peso do global ABAIXADO (context_w<1) e o proprio FOCO
    ERGUIDO (focus_w>1). Assim o especialista prioriza o que e' dele sem se
    sobre-comprometer com os conceitos dos outros centros (o que fritava). O
    backbone/global continua em 1.0 (fora daqui). Pesos do usuario '(x:v)' sao
    PRESERVADOS e multiplicam por cima. Retorna (pe, npe, pp, npp)."""
    device = device or torch.device("cpu")
    focus_ids = set()
    for w in (focus_words or []):
        focus_ids.update(pipe.tokenizer(w, add_special_tokens=False).input_ids)
    p = resolve_wildcards(full_prompt, rng).replace("\n", " BREAK ")
    n = resolve_wildcards(neg or "", rng).replace("\n", " BREAK ")
    pid, pw, bos, eos = _tokens_and_weights(parse_prompt_attention(p), pipe.tokenizer,
                                            focus_ids, focus_w, context_w)
    nid, nw, _, _ = _tokens_and_weights(parse_prompt_attention(n), pipe.tokenizer)  # neg normal
    empty = [bos] + [eos] * 76
    while len(pid) < len(nid):
        pid.append(list(empty)); pw.append([1.0] * 77)
    while len(nid) < len(pid):
        nid.append(list(empty)); nw.append([1.0] * 77)
    with torch.no_grad():
        pe, pp = _encode_chunks_sdxl(pipe, pid, pw, device)
        # O NEGATIVO NAO DEPENDE DO CENTRO. Esta funcao e' chamada uma vez por
        # centro e so' o POSITIVO muda (foco/contexto); o `nid/nw` sai identico
        # em todas. Medido: 15 centros gastaram 211s no encode, e metade disso
        # era recalcular o mesmo negativo 15 vezes na CPU.
        #
        # A chave inclui o formato do positivo porque o negativo e' PADDED ate
        # o comprimento dele (o while acima) — mesmo texto com contagem de
        # blocos diferente da tensor diferente.
        chave = (id(pipe), device_key(device), tuple(map(tuple, nid)), len(pid))
        cache = _CACHE_NEG.get(chave)
        if cache is None:
            cache = _encode_chunks_sdxl(pipe, nid, nw, device)
            if len(_CACHE_NEG) > 8:      # so' o prompt da vez importa
                _CACHE_NEG.clear()
            _CACHE_NEG[chave] = cache
        npe, npp = cache
    return pe, npe, pp, npp


# ------------------------------------------------------------------ 5. generico (T5)
def weight_token_embeddings(embeds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Aplica pesos por token a QUALQUER embedding [B,L,D] (ex.: T5 antes do pooling),
    preservando a media (mesmo metodo dos CLIP). weights: [B,L]."""
    orig = embeds.mean(dim=(1, 2), keepdim=True)
    out = embeds * weights.unsqueeze(-1).to(embeds.dtype)
    new = out.mean(dim=(1, 2), keepdim=True)
    return out * (orig / (new + 1e-6))
