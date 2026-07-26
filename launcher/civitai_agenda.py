"""Agenda de posts do CivitAI: leitura em calendario + agendamento.

PRINCIPIO: O AGENDAMENTO NAO MORA AQUI.
    Quem guarda a data de publicacao e' o CivitAI (campo `publishedAt` do post).
    Esta agenda so' LE e ESCREVE esse campo no servidor deles. Consequencia
    direta, que e' o requisito: o PC pode estar desligado na hora marcada — o
    post sai do mesmo jeito, porque quem publica e' o servidor. Nao existe
    agendador local, nem tarefa do Windows, nem nada que dependa da maquina
    estar de pe. Se existisse, seria um relogio paralelo que sairia de sincronia
    com o do site.

DUAS FONTES, DE CONFIABILIDADE DIFERENTE
    fonte "oficial"  -> API v1 publica (GET /images?username=...).
        Suportada e estavel, mas so' enxerga o que JA FOI PUBLICADO: a
        referencia da API nao tem rota de posts, e /images nao devolve
        `publishedAt` nem marca de rascunho. Serve para o historico do
        calendario, nao para o futuro.

    fonte "sessao"   -> tRPC interno (post.update / post.get), o mesmo que o
        proprio site chama ao clicar em "Schedule Publish". E' a UNICA via que
        enxerga post agendado e que consegue marcar data. Nao e' documentada e
        pode mudar sem aviso. Roda DENTRO de uma janela pywebview logada em
        civitai.com, em JS same-origin — o Python nunca toca no cookie de
        sessao, mesmo modelo de confianca de uma aba de navegador.

    A UI degrada sozinha: sem a fonte "sessao" o calendario ainda abre, so'
    mostra o passado e avisa que o futuro esta indisponivel.
"""
from __future__ import annotations

import calendar
import datetime as dt
from collections import defaultdict

import civitai_auth

TRPC = "https://civitai.com/api/trpc"


def _dia(iso: str) -> str:
    """ISO -> 'YYYY-MM-DD' local, tolerante a formato."""
    if not iso:
        return ""
    try:
        s = iso.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s).astimezone().strftime("%Y-%m-%d")
    except Exception:
        return str(iso)[:10]


def malha(ano: int, mes: int) -> dict:
    """Esqueleto do mes p/ o front desenhar sem recalcular calendario em JS."""
    cal = calendar.Calendar(firstweekday=6)          # domingo primeiro
    semanas = [[(d.isoformat() if d.month == mes else None) for d in semana]
               for semana in cal.monthdatescalendar(ano, mes)]
    return {"ano": ano, "mes": mes, "semanas": semanas,
            "hoje": dt.date.today().isoformat()}


def publicados(username: str, limite: int = 200) -> dict:
    """Historico via API oficial. Agrupa imagens por post."""
    if not username:
        eu = civitai_auth.quem_sou()
        if not eu.get("ok"):
            return {"ok": False, "erro": eu.get("erro", "sem usuario")}
        username = eu["usuario"]

    r = civitai_auth.api("images", {"username": username, "limit": min(limite, 200),
                                    "sort": "Newest"})
    if not r.get("ok"):
        return r

    por_post = defaultdict(lambda: {"imagens": 0, "capa": None, "quando": ""})
    for img in (r["dados"].get("items") or []):
        pid = img.get("postId")
        if pid is None:
            continue
        p = por_post[pid]
        p["imagens"] += 1
        p["capa"] = p["capa"] or img.get("url")
        p["quando"] = p["quando"] or img.get("createdAt", "")

    itens = [{"post_id": pid, "dia": _dia(v["quando"]), "quando": v["quando"],
              "imagens": v["imagens"], "capa": v["capa"],
              "estado": "publicado", "fonte": "oficial"}
             for pid, v in por_post.items()]
    itens.sort(key=lambda x: x["quando"], reverse=True)
    return {"ok": True, "username": username, "itens": itens}


# ---------------------------------------------------------------------------
# Agendados e agendamento PELA CREDENCIAL OFICIAL (OAuth no navegador padrao
# ou API key). Nada de janela embutida: pedir a senha do usuario dentro de uma
# janela do proprio app e' errado mesmo quando funciona, e o gerenciador de
# senhas dele vive no navegador.
#
# Situacao medida: os escopos do CivitAI descrevem MediaRead como "View images,
# videos & posts" e MediaWrite como "Upload media & create posts", mas a
# referencia da API v1 nao lista NENHUMA rota de posts. Um dos dois esta
# desatualizado. Estas funcoes tentam as rotas plausiveis e, se nenhuma
# responder, dizem exatamente isso — em vez de deixar a agenda parecendo
# completa com o futuro faltando.
# ---------------------------------------------------------------------------
ROTAS_POSTS = ["posts", "user/posts", "me/posts"]


def agendados_oficial() -> dict:
    tok, _origem = civitai_auth.token_ativo()
    if not tok:
        return {"ok": False, "erro": "sem credencial"}
    tentadas = []
    for rota in ROTAS_POSTS:
        r = civitai_auth.api(rota, {"limit": 100})
        if r.get("ok"):
            dados = r["dados"]
            itens = dados.get("items") if isinstance(dados, dict) else None
            if itens is None:
                tentadas.append(f"{rota}: 200 mas sem 'items'")
                continue
            saida = []
            for p in itens:
                pub = p.get("publishedAt")
                saida.append({"post_id": p.get("id"), "quando": pub,
                              "titulo": p.get("title") or "",
                              "imagens": len(p.get("images") or []),
                              "capa": None})
            return {"ok": True, "itens": saida, "rota": rota}
        tentadas.append(f"{rota}: {r.get('erro')}")
    return {"ok": False, "itens": [],
            "erro": "a API oficial do CivitAI não expõe rota de posts — os "
                    "AGENDADOS não são legíveis por ela (só o já publicado). "
                    "Tentei: " + "; ".join(tentadas)}


def agendar_oficial(post_id: int, quando_iso: str) -> dict:
    """Valida entrada antes de qualquer chamada; hoje nao ha rota oficial de
    escrita de post, entao devolve o motivo em vez de falhar obscuro."""
    pid = int(post_id)
    quando = dt.datetime.fromisoformat(quando_iso.replace("Z", "+00:00"))
    tok, _ = civitai_auth.token_ativo()
    if not tok:
        return {"ok": False, "erro": "sem credencial — entre pelo navegador"}
    return {"ok": False, "post_id": pid, "quando": quando.isoformat(),
            "erro": "a API oficial não tem rota de escrita de post, então o "
                    "GSMDE não consegue gravar a data. Agende pelo site; "
                    "quando o CivitAI publicar a rota (o escopo MediaWrite já "
                    "prevê 'create posts'), isto passa a funcionar sem mudar "
                    "mais nada."}


# ---------------------------------------------------------------------------
# JS do tRPC — MANTIDO SO' COMO REFERENCIA, nao e' chamado por nada.
# Era executado numa janela embutida logada; a janela saiu (ver
# app.py::civitai_sessao_abrir). Fica aqui documentando o contrato interno
# caso um dia haja um caminho legitimo de sessao.
# ---------------------------------------------------------------------------
JS_LISTAR_AGENDADOS = r"""
(async () => {
  // post.getMyDraftsAndScheduled nao e' publico; usamos o mesmo caminho que a
  // pagina /user/<eu>/posts?section=scheduled usa. Se o contrato mudar, isto
  // devolve {erro:...} e a agenda cai para so'-publicados em vez de quebrar.
  try {
    const url = '%TRPC%/post.getInfinite?input=' + encodeURIComponent(JSON.stringify({
      json: { period:'AllTime', sort:'Newest', browsingLevel:31,
              draftOnly:false, pending:true, limit:100 }
    }));
    const r = await fetch(url, {credentials:'include', headers:{'content-type':'application/json'}});
    if (!r.ok) return {erro: 'HTTP ' + r.status};
    const j = await r.json();
    const itens = (((j.result||{}).data||{}).json||{}).items || [];
    return {itens: itens.map(p => ({
      post_id: p.id,
      quando: p.publishedAt || null,
      titulo: p.title || '',
      imagens: (p.images||[]).length,
      capa: (p.images||[])[0] ? (p.images[0].url||null) : null
    }))};
  } catch (e) { return {erro: String(e)}; }
})()
"""

JS_AGENDAR = r"""
(async () => {
  try {
    const r = await fetch('%TRPC%/post.update', {
      method:'POST', credentials:'include',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({json:{id: %POST_ID%, publishedAt: '%QUANDO%'}})
    });
    const txt = await r.text();
    return r.ok ? {ok:true, resposta: txt.slice(0,300)}
                : {ok:false, erro:'HTTP ' + r.status, resposta: txt.slice(0,300)};
  } catch (e) { return {ok:false, erro:String(e)}; }
})()
"""


def js_listar_agendados() -> str:
    return JS_LISTAR_AGENDADOS.replace("%TRPC%", TRPC)


def js_agendar(post_id: int, quando_iso: str) -> str:
    # o post_id entra como numero e a data e' validada antes de virar string:
    # nada de interpolar texto livre do usuario dentro do JS.
    pid = int(post_id)
    quando = dt.datetime.fromisoformat(quando_iso.replace("Z", "+00:00"))
    return (JS_AGENDAR.replace("%TRPC%", TRPC)
                      .replace("%POST_ID%", str(pid))
                      .replace("%QUANDO%", quando.isoformat()))


def juntar(pubs: list, agendados: list) -> dict:
    """Une as duas fontes num mapa dia -> itens, que e' o que o calendario desenha."""
    por_dia = defaultdict(list)
    for it in pubs:
        if it.get("dia"):
            por_dia[it["dia"]].append(it)
    for it in agendados or []:
        if not it.get("quando"):
            continue
        d = _dia(it["quando"])
        por_dia[d].append({**it, "dia": d, "estado": "agendado", "fonte": "sessao"})
    for d in por_dia:
        por_dia[d].sort(key=lambda x: x.get("quando") or "")
    return dict(por_dia)
