"""Sensor de VRAM que enxerga o SPILL — o que o torch nao consegue ver.

O PROBLEMA QUE ISTO RESOLVE
    `torch.cuda.mem_get_info()` pergunta ao runtime quanto de VRAM ha livre. No
    Windows/WDDM isso e' insuficiente: quando a placa enche, o driver NAO devolve
    OOM — ele migra alocacoes para a memoria COMPARTILHADA (RAM via PCIe) e segue
    como se nada fosse. O torch continua reportando que esta tudo bem enquanto,
    na pratica, cada acesso aquela memoria atravessa o barramento.

    E o que explica o sintoma classico: `Compute` em 97% no Gerenciador de
    Tarefas com a geracao 8x mais lenta. A GPU nao esta calculando — esta parada
    esperando pagina. Medido nesta maquina: 7,6/8,0 GB dedicados + 5,2 GB
    derramados = 87 s/it, contra ~11 s/it sem derramar.

    O proprio motor ja sabia disso (o comentario em `_politica_vram` fala do
    "livre=0.0 com a placa longe de cheia") e contornava limpando tudo antes de
    medir. Mas com um sensor cego so da para medir UMA VEZ, antes do laco; se as
    ativacoes crescerem no meio (resolucao maior), ninguem percebe.

A FONTE DE VERDADE
    Os contadores de performance do Windows, que leem o estado do WDDM:

        \\GPU Adapter Memory(<luid>)\\Dedicated Usage   VRAM fisica em uso (todos)
        \\GPU Adapter Memory(<luid>)\\Shared Usage      compartilhada em uso (todos)
        \\GPU Process Memory(pid_N_<luid>)\\Local Usage      VRAM fisica NOSSA
        \\GPU Process Memory(pid_N_<luid>)\\Non Local Usage  SPILL NOSSO

    "Local" = memoria fisica da placa. "Non Local" = memoria de sistema exposta a
    GPU, isto e', o derramamento. Essa e' a metrica que faltava.

POR QUE ctypes E NAO Get-Counter
    Um `Get-Counter` custa 200-500 ms (sobe o PowerShell inteiro) — inviavel
    dentro do laco de denoise. A PDH chamada direto custa microssegundos depois
    da abertura da query.

POR QUE PdhAddEnglishCounterW
    Esta maquina roda Windows em portugues. `PdhAddCounterW` espera o nome do
    contador NO IDIOMA DO SISTEMA — "\\GPU Adapter Memory\\Dedicated Usage"
    falharia numa instalacao localizada. A variante `English` aceita sempre o
    nome canonico, entao o codigo funciona em qualquer locale.
"""
from __future__ import annotations

import ctypes
import os
import re
import time
from ctypes import wintypes

# --- PDH ---------------------------------------------------------------------
PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2
ERROR_SUCCESS = 0


def _rc(v: int) -> int:
    """Normaliza o retorno da PDH para SEM sinal.

    O ctypes assume `int` (32 bits COM sinal) no retorno de funcao de DLL, entao
    PDH_MORE_DATA (0x800007D2) chega como -2147481134 e nunca casa com a
    constante. Sem isto o sensor le zero em tudo, silenciosamente."""
    return v & 0xFFFFFFFF


class _CounterValue(ctypes.Structure):
    # DWORD + padding + union de 8 bytes. Em x64 o campo grande fica alinhado
    # em 8, entao a struct tem 16 bytes — deixar o ctypes calcular o padding.
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _CounterItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _CounterValue)]


_ADAPTER = r"\GPU Adapter Memory(*)\{}"
_PROCESSO = r"\GPU Process Memory(*)\{}"

# instancia por processo: "pid_10956_luid_0x00000000_0x00012d76_phys_0"
_RE_PID = re.compile(r"^pid_(\d+)_(luid_.+)$")


class _Query:
    """Uma query PDH aberta, com os contadores que nos interessam."""

    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh.dll")
        self.h = wintypes.HANDLE()
        if _rc(self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.h))) != ERROR_SUCCESS:
            raise OSError("PdhOpenQueryW falhou")
        self.cont = {}
        for chave, caminho in (
            ("ad_ded", _ADAPTER.format("Dedicated Usage")),
            ("ad_shr", _ADAPTER.format("Shared Usage")),
            ("pr_loc", _PROCESSO.format("Local Usage")),
            ("pr_nloc", _PROCESSO.format("Non Local Usage")),
        ):
            c = wintypes.HANDLE()
            rc = _rc(self.pdh.PdhAddEnglishCounterW(self.h, caminho, 0, ctypes.byref(c)))
            if rc == ERROR_SUCCESS:
                self.cont[chave] = c
        if not self.cont:
            raise OSError("nenhum contador de GPU disponivel")
        self.pdh.PdhCollectQueryData(self.h)

    def coleta(self) -> None:
        self.pdh.PdhCollectQueryData(self.h)

    def valores(self, chave) -> dict:
        """instancia -> bytes. Vazio se o contador nao existir."""
        c = self.cont.get(chave)
        if c is None:
            return {}
        tam = wintypes.DWORD(0)
        n = wintypes.DWORD(0)
        rc = _rc(self.pdh.PdhGetFormattedCounterArrayW(
            c, PDH_FMT_LARGE, ctypes.byref(tam), ctypes.byref(n), None))
        if rc != PDH_MORE_DATA or tam.value == 0:
            return {}
        buf = ctypes.create_string_buffer(tam.value)
        rc = _rc(self.pdh.PdhGetFormattedCounterArrayW(
            c, PDH_FMT_LARGE, ctypes.byref(tam), ctypes.byref(n), buf))
        if rc != ERROR_SUCCESS:
            return {}
        itens = ctypes.cast(buf, ctypes.POINTER(_CounterItem))
        out = {}
        for i in range(n.value):
            it = itens[i]
            if it.szName:
                out[it.szName] = int(it.FmtValue.largeValue)
        return out

    def fechar(self):
        try:
            self.pdh.PdhCloseQuery(self.h)
        except Exception:
            pass


def _vram_total_bytes() -> int:
    """Capacidade fisica da placa. O torch e' a fonte mais confiavel; o registro
    e' o plano B (Win32_VideoController.AdapterRAM e' int32 e mente acima de 4GB)."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    try:
        import winreg
        chave = (r"SYSTEM\CurrentControlSet\Control\Class"
                 r"\{4d36e968-e325-11ce-bfc1-08002be10318}\0000")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, chave) as k:
            return int(winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0])
    except Exception:
        return 0


GB = float(1 << 30)


class VramSensor:
    """Leitura do estado REAL da placa, separando o que e' nosso do que e' do resto.

    pids: quais processos contam como "nossos". Por padrao o processo atual — no
    worker do GSMDE e' exatamente quem aloca. Aceita lista para o caso de o motor
    rodar em subprocesso separado da UI.

    intervalo: idade maxima da amostra em cache. O laco de denoise pode chamar
    `ler()` a cada passo sem custo — so re-consulta quando a amostra envelhece.
    """

    def __init__(self, pids=None, intervalo=1.0):
        self.pids = set(pids or [os.getpid()])
        self.intervalo = float(intervalo)
        self.total_gb = _vram_total_bytes() / GB
        self._q = None
        self._cache = None
        self._t = 0.0
        self._luid = None
        try:
            self._q = _Query()
        except Exception as e:                    # sem sensor o motor deve seguir
            self._erro = str(e)

    @property
    def ok(self) -> bool:
        return self._q is not None

    def _adaptador(self, por_pid: dict, ad: dict) -> str:
        """Qual placa e' 'a' placa. Deduz pelo LUID do NOSSO processo: se estamos
        alocando nela, e' nela que a geracao roda. So cai para 'a de maior uso'
        quando ainda nao alocamos nada."""
        if self._luid:
            return self._luid
        for inst in por_pid:
            m = _RE_PID.match(inst)
            if m and int(m.group(1)) in self.pids:
                self._luid = m.group(2)
                return self._luid
        return max(ad, key=ad.get) if ad else ""

    def ler(self, forcar=False) -> dict:
        """Estado atual, em GB. Todos os campos existem mesmo sem sensor (zeros)."""
        agora = time.time()
        if not forcar and self._cache and (agora - self._t) < self.intervalo:
            return self._cache
        base = {"ok": False, "total": self.total_gb, "dedicada_total": 0.0,
                "dedicada_nossa": 0.0, "compartilhada_total": 0.0,
                "nonlocal_nosso": 0.0, "spill_real": 0.0, "reservado_torch": 0.0,
                "outros": 0.0, "outros_shr": 0.0,
                "livre": self.total_gb, "livre_pct": 100.0}
        if not self._q:
            self._cache, self._t = base, agora
            return base

        self._q.coleta()
        ad_ded = self._q.valores("ad_ded")
        ad_shr = self._q.valores("ad_shr")
        pr_loc = self._q.valores("pr_loc")
        pr_nloc = self._q.valores("pr_nloc")

        luid = self._adaptador(pr_loc, ad_ded)

        def nosso(d):
            t = 0
            for inst, v in d.items():
                m = _RE_PID.match(inst)
                if m and int(m.group(1)) in self.pids and (not luid or m.group(2) == luid):
                    t += v
            return t

        ded_total = ad_ded.get(luid, sum(ad_ded.values()) if ad_ded else 0) / GB
        shr_total = ad_shr.get(luid, sum(ad_shr.values()) if ad_shr else 0) / GB
        ded_nossa = nosso(pr_loc) / GB
        nonlocal_nosso = nosso(pr_nloc) / GB

        # SPILL DE VERDADE vs. MEMORIA PINADA
        #   'Non Local Usage' NAO e' so derramamento: RAM pinada pelo host tambem
        #   entra. Medido: 1,5GB de pin_memory fez o contador saltar 0,00 -> 2,01GB
        #   sem nenhuma eviccao. Como a paginacao por bloco pina TODOS os centros
        #   (6 x 0,56GB = ~3,4GB), usar esse numero como alarme daria falso
        #   positivo justamente no caminho que queremos usar.
        #
        #   O sinal limpo e' outro: o que o torch acha que reservou NA PLACA menos
        #   o que a placa diz ter fisicamente nosso. Se o torch reservou 1,50GB e o
        #   hardware so tem 1,16GB nosso, os 0,34GB que faltam foram despejados.
        #   RESSALVA: `memory_reserved()` fala do PROCESSO ATUAL. Num monitor
        #   externo (outro processo) esse numero e' zero e o despejo ficaria
        #   invisivel — pior, sairia como "0,00GB, tudo bem". Entao so calcula
        #   quando estamos medindo a nos mesmos; fora disso devolve None, que o
        #   relatorio imprime como "n/d" em vez de fingir um zero.
        reservado = 0.0
        proprio = self.pids == {os.getpid()}
        if proprio:
            try:
                import torch
                if torch.cuda.is_available():
                    reservado = torch.cuda.memory_reserved() / GB
            except Exception:
                pass
        spill_real = max(0.0, reservado - ded_nossa) if proprio else None

        total = self.total_gb or ded_total
        livre = max(0.0, total - ded_total)
        out = {
            "ok": True,
            "total": total,                       # capacidade fisica da placa
            "dedicada_total": ded_total,          # fisica em uso (nos + Windows)
            "dedicada_nossa": ded_nossa,          # fisica em uso so nossa
            "outros": max(0.0, ded_total - ded_nossa),        # SO: parte fisica
            "outros_shr": max(0.0, shr_total - nonlocal_nosso),  # SO: compartilhada
            "compartilhada_total": shr_total,     # compartilhada em uso (todos)
            "nonlocal_nosso": nonlocal_nosso,     # nosso Non Local = spill + pinada
            "spill_real": spill_real,             # DESPEJO de verdade  <-- o alvo
            "reservado_torch": reservado,         # o que o torch acha que tem
            "livre": livre,                       # fisica livre de verdade
            "livre_pct": (livre / total * 100.0) if total else 0.0,
        }
        self._cache, self._t = out, agora
        return out

    def resumo(self) -> str:
        d = self.ler()
        if not d["ok"]:
            return "[vram] sensor indisponivel"
        # mostra FISICA e COMPARTILHADA lado a lado: olhar so a fisica esconde
        # metade do consumo (o desktop usa ~1,7 fisica + ~0,8 compartilhada).
        s = (f"[vram] fisica {d['dedicada_total']:.2f}/{d['total']:.1f}GB "
             f"(nossa {d['dedicada_nossa']:.2f} + SO {d['outros']:.2f}) | "
             f"compart. {d['compartilhada_total']:.2f}GB "
             f"(nossa {d['nonlocal_nosso']:.2f} + SO {d['outros_shr']:.2f}) | "
             f"livre {d['livre']:.2f}GB ({d['livre_pct']:.0f}%)")
        if d["spill_real"] is None:
            s += " | despejo n/d (medindo outro processo)"
        elif d["spill_real"] > 0.05:
            s += f" | DESPEJO {d['spill_real']:.2f}GB"
        return s

    def __del__(self):
        if self._q:
            self._q.fechar()


if __name__ == "__main__":
    import sys
    alvo = [int(a) for a in sys.argv[1:] if a.isdigit()] or None
    s = VramSensor(pids=alvo)
    print(f"sensor: {'ok' if s.ok else 'INDISPONIVEL'} | placa {s.total_gb:.2f}GB "
          f"| pids {sorted(s.pids)}\n")
    for _ in range(3):
        d = s.ler(forcar=True)
        print(s.resumo())
        time.sleep(1)
    print()
    for k, v in s.ler(forcar=True).items():
        print(f"  {k:16} {v}")
