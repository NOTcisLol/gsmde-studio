"""Login opcional no Hugging Face: OAuth 2.0 + PKCE no navegador padrao, ou
token pessoal (hf_...).

Mesmo desenho do civitai_auth.py — inclusive o cofre DPAPI, que e' importado de
la' em vez de duplicado. Diferencas reais em relacao ao CivitAI:

  - A HF aceita "public app" (sem client secret) explicitamente, feito p/ app
    nativo/CLI. Registro em https://huggingface.co/settings/applications/new
  - Escopos sao outros. Para BAIXAR modelo basta `read-repos`; `gated-repos`
    cobre os repositorios com termo de uso aceito (varios SDXL sao assim).
  - Existe tambem o device code flow (/oauth/device), util quando nao ha
    navegador na maquina. Aqui usamos loopback, que e' o que o usuario pediu.
  - O token pessoal da HF e' o mesmo que o huggingface_hub usa, entao quem ja
    tem `huggingface-cli login` feito pode so' apontar p/ ele.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import HTTPServer

# reusa cofre (DPAPI), servidor de callback e helpers HTTP
from civitai_auth import (_Retorno, _cofre_gravar, _cofre_ler, _get_json,
                          _pkce, _post_form)

OAUTH = "https://huggingface.co/oauth"
API = "https://huggingface.co/api"
ESCOPOS_PADRAO = ["openid", "profile", "read-repos", "gated-repos"]

# Portas proprias, p/ nao colidir com o loopback do CivitAI se os dois logins
# forem feitos ao mesmo tempo. Mesmo motivo la': porta fixa e' registravel,
# porta aleatoria nao.
PORTAS = [7794, 7795, 7796]


def redirect_uris() -> list:
    return [f"http://127.0.0.1:{p}/callback" for p in PORTAS]
FOLGA = 120
_CHAVE = "hf"          # sub-arvore propria dentro do mesmo cofre


def _hf() -> dict:
    return _cofre_ler().get(_CHAVE) or {}


def _salva(d: dict) -> None:
    cofre = _cofre_ler()
    cofre[_CHAVE] = d
    _cofre_gravar(cofre)


def login(client_id: str, escopos=None, timeout=180) -> dict:
    if not client_id:
        return {"ok": False, "erro": "sem client_id — crie um app em "
                                     "huggingface.co/settings/applications/new "
                                     "(pode ser 'public app', sem secret)"}
    escopos = escopos or ESCOPOS_PADRAO
    verifier, challenge = _pkce()
    estado = secrets.token_urlsafe(24)

    _Retorno.resultado = {}
    _Retorno.estado_esperado = estado
    srv = porta = None
    for p in PORTAS:
        try:
            srv, porta = HTTPServer(("127.0.0.1", p), _Retorno), p
            break
        except OSError:
            continue
    if srv is None:
        return {"ok": False, "erro": f"portas {PORTAS} ocupadas"}
    redirect_uri = f"http://127.0.0.1:{porta}/callback"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"{OAUTH}/authorize?" + urllib.parse.urlencode({
            "response_type": "code", "client_id": client_id,
            "redirect_uri": redirect_uri, "scope": " ".join(escopos),
            "state": estado, "code_challenge": challenge,
            "code_challenge_method": "S256",
        })
        if not webbrowser.open(url):
            return {"ok": False, "erro": "nao consegui abrir o navegador", "url": url}
        limite = time.time() + timeout
        while not _Retorno.resultado and time.time() < limite:
            time.sleep(0.25)
        res = _Retorno.resultado
        if not res:
            return {"ok": False, "erro": f"timeout de {timeout}s"}
        if res.get("error") or not res.get("code"):
            return {"ok": False, "erro": res.get("error") or "sem code"}
        tok = _post_form(f"{OAUTH}/token", {
            "grant_type": "authorization_code", "code": res["code"],
            "redirect_uri": redirect_uri, "client_id": client_id,
            "code_verifier": verifier,
        })
    except urllib.error.HTTPError as e:
        return {"ok": False, "erro": f"HTTP {e.code}: "
                                    f"{e.read().decode('utf-8', 'replace')[:400]}"}
    except Exception as e:
        return {"ok": False, "erro": str(e)}
    finally:
        srv.shutdown()
        srv.server_close()

    if not tok.get("access_token"):
        return {"ok": False, "erro": f"resposta sem access_token: {tok}"}
    _salva({"access_token": tok["access_token"],
            "refresh_token": tok.get("refresh_token"),
            "expira_em": time.time() + int(tok.get("expires_in") or 28800),
            "escopos": tok.get("scope", " ".join(escopos)),
            "client_id": client_id})
    return {"ok": True, **quem_sou()}


def _renova(d: dict) -> bool:
    if not d.get("refresh_token"):
        return False
    try:
        tok = _post_form(f"{OAUTH}/token", {
            "grant_type": "refresh_token", "refresh_token": d["refresh_token"],
            "client_id": d.get("client_id", "")})
    except Exception:
        return False
    if not tok.get("access_token"):
        return False
    d.update({"access_token": tok["access_token"],
              "refresh_token": tok.get("refresh_token", d["refresh_token"]),
              "expira_em": time.time() + int(tok.get("expires_in") or 28800)})
    _salva(d)
    return True


def token_ativo() -> tuple[str, str]:
    d = _hf()
    if d.get("access_token"):
        if d.get("expira_em", 0) - FOLGA <= time.time() and not _renova(d):
            d = {}
        if d.get("access_token"):
            return d["access_token"], "oauth"
    if d.get("token_pessoal"):
        return d["token_pessoal"], "token"
    return "", ""


def set_token(tok: str) -> dict:
    tok = (tok or "").strip()
    d = _hf()
    if tok:
        d["token_pessoal"] = tok
    else:
        d.pop("token_pessoal", None)
    _salva(d)
    return quem_sou()


def logout() -> dict:
    cofre = _cofre_ler()
    cofre.pop(_CHAVE, None)
    _cofre_gravar(cofre)
    return {"ok": True}


def status() -> dict:
    d = _hf()
    return {"oauth": bool(d.get("access_token")), "token": bool(d.get("token_pessoal")),
            "logado": bool(d.get("access_token") or d.get("token_pessoal")),
            "escopos": d.get("escopos", ""), "expira_em": d.get("expira_em"),
            "usuario": d.get("usuario"), "client_id": d.get("client_id", "")}


def quem_sou() -> dict:
    tok, origem = token_ativo()
    if not tok:
        return {"ok": False, "erro": "sem credencial"}
    try:
        me = _get_json(f"{API}/whoami-v2", tok)
    except urllib.error.HTTPError as e:
        return {"ok": False, "erro": f"HTTP {e.code} — credencial rejeitada"}
    except Exception as e:
        return {"ok": False, "erro": str(e)}
    nome = me.get("name") or me.get("fullname")
    if nome:
        d = _hf()
        d["usuario"] = nome
        _salva(d)
    return {"ok": True, "origem": origem, "usuario": nome, **status()}
