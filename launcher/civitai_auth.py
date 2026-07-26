"""Login opcional no CivitAI: OAuth 2.0 Authorization Code + PKCE (RFC 7636),
no NAVEGADOR PADRAO DO SISTEMA, com redirect de loopback (RFC 8252).

É o mesmo desenho que o Claude Code usa: o app abre a página de autorização no
navegador de verdade, o usuário loga lá (inclusive com o botão do Google, que
funciona porque é um navegador legítimo), e o navegador redireciona de volta
para http://127.0.0.1:<porta>/callback, onde este módulo captura o `code` e o
troca por tokens.

POR QUE NAO DENTRO DO PYWEBVIEW
    O Google bloqueia OAuth em webview embutido desde 2023 (`disallowed_useragent`)
    justamente porque o app hospedeiro consegue ler as teclas e os cookies da
    sessão. pywebview é exatamente esse caso. Fora o bloqueio, o desenho certo é
    este: a senha nunca passa por código nosso, e o token é escopado e revogável
    pelo usuário em civitai.com a qualquer momento.

DUAS CREDENCIAIS, PROPOSITOS DIFERENTES
    - API key pessoal: você gera nas configurações da conta, cobre todos os
      endpoints autenticados. A doc deles recomenda "apenas para uso pessoal".
      Funciona sem registrar app nenhum.
    - OAuth: para quando outra pessoa usar o GSMDE. Exige um client_id
      registrado por você no portal do CivitAI.
    As duas chegam na mesma API e no mesmo header (`Authorization: Bearer ...`),
    então o resto do app não precisa saber qual está em uso.

ARMAZENAMENTO
    Token e API key ficam cifrados com DPAPI do Windows (CryptProtectData),
    atrelados à conta de usuário: outro usuário da máquina não consegue decifrar,
    e nada fica em texto puro no disco. Sem dependência externa.
"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

AUTH_BASE = "https://auth.civitai.com/api/auth/oauth"
API_BASE = "https://civitai.com/api/v1"
LAUNCHER = Path(__file__).resolve().parent
COFRE = LAUNCHER / "civitai_auth.bin"

# MediaRead/MediaWrite sao os que interessam: a doc de escopos descreve
# MediaRead como "View images, videos & posts" e MediaWrite como
# "Upload media & create posts". UserRead vem sempre.
# ESCOPOS: o CivitAI NAO aceita nomes no parametro `scope`.
#
# Mandar "UserRead MediaRead ..." devolve {"error":"invalid_scope"}. O que ele
# espera e' um BITMASK em decimal — o discovery
# (auth.civitai.com/.well-known/openid-configuration) lista `scopes_supported`
# como potencias de 2 (1, 2, 4, ... 67108864), nao como strings. Os nomes da
# tabela da documentacao sao rotulo de exibicao, nao valor de protocolo.
#
# A ordem abaixo (posicao = numero do bit) foi VERIFICADA contra o exemplo da
# propria doc: scope=114689 = UserRead|AIServicesRead|AIServicesWrite|BuzzRead,
# e 114689 acende exatamente os bits 0, 14, 15 e 16 — que sao as posicoes
# desses quatro nesta lista. Se a ordem estivesse errada em qualquer ponto
# antes do 14, esses tres nao cairiam onde caem.
BITS = ["UserRead", "UserWrite",
        "ModelsRead", "ModelsWrite", "ModelsDelete",
        "MediaRead", "MediaWrite", "MediaDelete",
        "ArticlesRead", "ArticlesWrite", "ArticlesDelete",
        "BountiesRead", "BountiesWrite", "BountiesDelete",
        "AIServicesRead", "AIServicesWrite",
        "BuzzRead",
        "CollectionsRead", "CollectionsWrite",
        "SocialWrite",
        "NotificationsRead", "NotificationsWrite",
        "VaultRead", "VaultWrite"]

# UserRead(1) + ModelsRead(4) + MediaRead(32) + MediaWrite(64) = 101
ESCOPOS_PADRAO = ["UserRead", "ModelsRead", "MediaRead", "MediaWrite"]


def mascara(escopos) -> int:
    """Nomes -> inteiro. Escopo desconhecido explode aqui, na chamada, em vez
    de virar um `invalid_scope` opaco depois de abrir o navegador."""
    total = 0
    for e in escopos:
        if e not in BITS:
            raise ValueError(f"escopo desconhecido: {e} (validos: {BITS})")
        total |= 1 << BITS.index(e)
    return total

# PORTAS FIXAS para o redirect de loopback.
#
# Antes isto era porta 0 (o SO escolhia uma livre). Elegante, e impossivel de
# registrar: o CivitAI exige que os Redirect URIs sejam declarados na criacao
# do app, e uma porta que muda a cada login nunca casaria. A RFC 8252 pede que
# o servidor aceite qualquer porta em loopback, mas nao da' p/ contar com isso.
#
# Sao tres p/ o login nao morrer se uma estiver ocupada — todas registradas no
# app, e usamos a primeira que subir.
PORTAS = [7791, 7792, 7793]


def redirect_uris() -> list:
    """As URIs que precisam estar cadastradas no app do CivitAI."""
    return [f"http://127.0.0.1:{p}/callback" for p in PORTAS]

# Margem antes do vencimento p/ renovar (o access token dura 1h).
FOLGA_RENOVACAO = 120


# ---------------------------------------------------------------------------
# Cofre (DPAPI)
# ---------------------------------------------------------------------------
class _BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(dados: bytes) -> _BLOB:
    buf = ctypes.create_string_buffer(dados, len(dados))
    return _BLOB(len(dados), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _dpapi(dados: bytes, proteger: bool) -> bytes:
    entrada, saida = _blob(dados), _BLOB()
    fn = (ctypes.windll.crypt32.CryptProtectData if proteger
          else ctypes.windll.crypt32.CryptUnprotectData)
    ok = fn(ctypes.byref(entrada), None, None, None, None, 0, ctypes.byref(saida))
    if not ok:
        raise OSError("DPAPI falhou (erro %d)" % ctypes.GetLastError())
    try:
        return ctypes.string_at(saida.pbData, saida.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(saida.pbData)


def _cofre_ler() -> dict:
    if not COFRE.exists():
        return {}
    try:
        return json.loads(_dpapi(COFRE.read_bytes(), proteger=False).decode("utf-8"))
    except Exception:
        # cofre de outro usuario/maquina ou corrompido: trata como deslogado em
        # vez de explodir — o pior resultado aqui e' pedir login de novo.
        return {}


def _cofre_gravar(dados: dict) -> None:
    bruto = json.dumps(dados, ensure_ascii=False).encode("utf-8")
    tmp = COFRE.with_suffix(".tmp")
    tmp.write_bytes(_dpapi(bruto, proteger=True))
    tmp.replace(COFRE)


def logout() -> dict:
    """Esquece as credenciais locais. NAO revoga no servidor — para revogar de
    verdade o usuario usa civitai.com (ou chamamos /revoke, ver revogar())."""
    tok = _cofre_ler().get("oauth", {}).get("refresh_token")
    if COFRE.exists():
        COFRE.unlink()
    if tok:
        try:
            _post_form(f"{AUTH_BASE}/revoke", {"token": tok})
        except Exception:
            pass
    return {"ok": True}


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib — sem dependencia nova no venv)
# ---------------------------------------------------------------------------
def _post_form(url: str, campos: dict) -> dict:
    corpo = urllib.parse.urlencode(campos).encode()
    req = urllib.request.Request(url, data=corpo, method="POST", headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": "GSMDE-Studio/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def _get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "GSMDE-Studio/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------
def _pkce() -> tuple[str, str]:
    """(verifier, challenge S256). O CivitAI exige S256 p/ todo cliente."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).decode().rstrip("=")


# ---------------------------------------------------------------------------
# Servidor de loopback que recebe o redirect
# ---------------------------------------------------------------------------
_PAGINA = """<!doctype html><meta charset="utf-8">
<title>GSMDE - login CivitAI</title>
<style>body{font:16px system-ui;background:#12141a;color:#e6e6e6;
display:grid;place-items:center;height:100vh;margin:0;text-align:center}
.c{max-width:420px}h1{font-size:20px;margin:0 0 8px}p{opacity:.75;line-height:1.5}
.ok{color:#7ee787}.err{color:#ff7b72}</style>
<div class="c"><h1 class="%s">%s</h1><p>%s</p></div>"""


class _Retorno(BaseHTTPRequestHandler):
    resultado: dict = {}
    estado_esperado = ""

    def log_message(self, *a):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        q = urllib.parse.parse_qs(parsed.query)
        erro = (q.get("error") or [""])[0]
        code = (q.get("code") or [""])[0]
        estado = (q.get("state") or [""])[0]

        # o `state` amarra a resposta ao pedido: sem conferir, qualquer pagina
        # aberta no navegador poderia bater neste callback e injetar um code.
        if estado != _Retorno.estado_esperado:
            erro, code = "state_invalido", ""

        if erro or not code:
            titulo, classe = "Login nao concluido", "err"
            texto = f"Motivo: {erro or 'sem codigo de autorizacao'}. Pode fechar esta aba."
        else:
            titulo, classe = "Pronto!", "ok"
            texto = "Login concluido. Pode fechar esta aba e voltar ao GSMDE."

        pagina = (_PAGINA % (classe, titulo, texto)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(pagina)))
        self.end_headers()
        self.wfile.write(pagina)
        _Retorno.resultado = {"code": code, "error": erro}


def login(client_id: str, escopos=None, timeout=180) -> dict:
    """Fluxo completo. BLOQUEIA ate o usuario concluir no navegador ou dar timeout.

    Chamar de uma thread (a Api do pywebview ja roda fora da thread de UI)."""
    if not client_id:
        return {"ok": False, "erro": "sem client_id — registre um app em civitai.com "
                                     "e informe o client_id nas configuracoes"}
    escopos = escopos or ESCOPOS_PADRAO
    verifier, challenge = _pkce()
    estado = secrets.token_urlsafe(24)

    # Loopback fixo em 127.0.0.1 (nao 'localhost'): resolucao de nome pode ir
    # para ::1 e o redirect nao casar com o que foi registrado.
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
        return {"ok": False, "erro": f"portas {PORTAS} todas ocupadas — feche o "
                                     f"que estiver usando ou registre outra"}
    redirect_uri = f"http://127.0.0.1:{porta}/callback"

    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"{AUTH_BASE}/authorize?" + urllib.parse.urlencode({
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": str(mascara(escopos)),      # bitmask decimal, nao nomes
            "state": estado,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })
        if not webbrowser.open(url):
            return {"ok": False, "erro": "nao consegui abrir o navegador padrao",
                    "url": url}

        limite = time.time() + timeout
        while not _Retorno.resultado and time.time() < limite:
            time.sleep(0.25)
        res = _Retorno.resultado
        if not res:
            return {"ok": False, "erro": f"timeout de {timeout}s aguardando o navegador"}
        if res.get("error") or not res.get("code"):
            return {"ok": False, "erro": res.get("error") or "sem code"}

        tok = _post_form(f"{AUTH_BASE}/token", {
            "grant_type": "authorization_code",
            "code": res["code"],
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        })
    except urllib.error.HTTPError as e:
        detalhe = e.read().decode("utf-8", "replace")[:400]
        return {"ok": False, "erro": f"HTTP {e.code} na troca do code: {detalhe}"}
    except Exception as e:
        return {"ok": False, "erro": str(e)}
    finally:
        srv.shutdown()
        srv.server_close()

    if not tok.get("access_token"):
        return {"ok": False, "erro": f"resposta sem access_token: {tok}"}

    cofre = _cofre_ler()
    cofre["oauth"] = {
        "access_token": tok["access_token"],
        "refresh_token": tok.get("refresh_token"),
        "expira_em": time.time() + int(tok.get("expires_in") or 3600),
        "escopos": tok.get("scope", " ".join(escopos)),
        "client_id": client_id,
    }
    _cofre_gravar(cofre)
    return {"ok": True, **status()}


def _renova(cofre: dict) -> bool:
    o = cofre.get("oauth") or {}
    if not o.get("refresh_token"):
        return False
    try:
        tok = _post_form(f"{AUTH_BASE}/token", {
            "grant_type": "refresh_token",
            "refresh_token": o["refresh_token"],
            "client_id": o.get("client_id", ""),
        })
    except Exception:
        return False
    if not tok.get("access_token"):
        return False
    o["access_token"] = tok["access_token"]
    o["refresh_token"] = tok.get("refresh_token", o["refresh_token"])
    o["expira_em"] = time.time() + int(tok.get("expires_in") or 3600)
    cofre["oauth"] = o
    _cofre_gravar(cofre)
    return True


def token_ativo() -> tuple[str, str]:
    """(token, origem) — renova o OAuth se estiver perto de vencer.
    Prefere OAuth; cai para a API key se nao houver sessao."""
    cofre = _cofre_ler()
    o = cofre.get("oauth") or {}
    if o.get("access_token"):
        if o.get("expira_em", 0) - FOLGA_RENOVACAO <= time.time():
            if not _renova(cofre):
                o = {}
        if o.get("access_token"):
            return o["access_token"], "oauth"
    if cofre.get("api_key"):
        return cofre["api_key"], "api_key"
    return "", ""


def set_api_key(chave: str) -> dict:
    chave = (chave or "").strip()
    cofre = _cofre_ler()
    if chave:
        cofre["api_key"] = chave
    else:
        cofre.pop("api_key", None)
    _cofre_gravar(cofre)
    return status()


def status() -> dict:
    """Estado do login, SEM devolver o token ao front (o JS nao precisa dele —
    quem fala com o CivitAI e' o Python)."""
    cofre = _cofre_ler()
    o = cofre.get("oauth") or {}
    tem_key = bool(cofre.get("api_key"))
    st = {
        "oauth": bool(o.get("access_token")),
        "api_key": tem_key,
        "escopos": o.get("escopos", ""),
        "expira_em": o.get("expira_em"),
        "logado": bool(o.get("access_token") or tem_key),
        "usuario": cofre.get("usuario"),
    }
    return st


def quem_sou() -> dict:
    """Confirma a credencial contra o servidor e memoriza o nome de usuario."""
    token, origem = token_ativo()
    if not token:
        return {"ok": False, "erro": "sem credencial (nem OAuth nem API key)"}
    try:
        if origem == "oauth":
            me = _get_json(f"{AUTH_BASE}/userinfo", token)
        else:
            me = _get_json(f"{API_BASE}/me", token)
    except urllib.error.HTTPError as e:
        return {"ok": False, "erro": f"HTTP {e.code} — credencial rejeitada", "origem": origem}
    except Exception as e:
        return {"ok": False, "erro": str(e), "origem": origem}
    nome = me.get("username") or me.get("name") or me.get("preferred_username")
    if nome:
        cofre = _cofre_ler()
        cofre["usuario"] = nome
        _cofre_gravar(cofre)
    return {"ok": True, "origem": origem, "usuario": nome, "bruto": me}


def api(caminho: str, params: dict | None = None) -> dict:
    """GET autenticado na API oficial. `caminho` relativo a /api/v1."""
    token, origem = token_ativo()
    if not token:
        return {"ok": False, "erro": "nao logado"}
    url = f"{API_BASE}/{caminho.lstrip('/')}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        return {"ok": True, "origem": origem, "dados": _get_json(url, token)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "erro": f"HTTP {e.code}",
                "detalhe": e.read().decode("utf-8", "replace")[:400]}
    except Exception as e:
        return {"ok": False, "erro": str(e)}
