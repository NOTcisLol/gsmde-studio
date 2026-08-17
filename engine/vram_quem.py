"""Quem esta ocupando a VRAM fisica, por processo, com nome.

Antes de tentar "expulsar o SO da placa" e' preciso saber QUEM esta la. Numa
maquina de trabalho a resposta costuma nao ser "o Windows": e' o navegador, a
propria UI (WebView2), o player de video. Isso muda o remedio — fechar uma aba
resolve mais que mexer no registro.

Tambem lista as placas disponiveis: havendo uma iGPU, mandar o desktop para ela
libera a dedicada INTEIRA, que e' a solucao limpa.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vram_sensor import _Query, _RE_PID, GB          # noqa: E402


def nomes(pids) -> dict:
    """pid -> nome do executavel (uma chamada so p/ todos)."""
    if not pids:
        return {}
    lista = ",".join(str(p) for p in pids)
    ps = (f"Get-Process -Id {lista} -EA SilentlyContinue | "
          f"ForEach-Object {{ \"$($_.Id)|$($_.ProcessName)\" }}")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=25).stdout
    except Exception:
        return {}
    d = {}
    for l in out.splitlines():
        if "|" in l:
            a, b = l.split("|", 1)
            if a.strip().isdigit():
                d[int(a)] = b.strip()
    return d


def placas() -> list:
    ps = ("Get-CimInstance Win32_VideoController | "
          "ForEach-Object { \"$($_.Name)|$($_.AdapterCompatibility)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=25).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def main():
    print("=== placas de video nesta maquina ===")
    ps = placas()
    for p in ps:
        n, f = (p.split("|", 1) + [""])[:2]
        print(f"  {n}   [{f}]")
    if len(ps) > 1:
        print("  >> ha mais de uma placa: mandar o desktop p/ a secundaria")
        print("     libera a dedicada INTEIRA (Configuracoes > Tela > Graficos).")
    else:
        print("  >> placa unica: nao da p/ mover o desktop para outra GPU.")

    q = _Query()
    q.coleta()
    loc = q.valores("pr_loc")
    nloc = q.valores("pr_nloc")

    por_pid = {}
    for inst, v in loc.items():
        m = _RE_PID.match(inst)
        if m:
            por_pid.setdefault(int(m.group(1)), [0, 0])[0] += v
    for inst, v in nloc.items():
        m = _RE_PID.match(inst)
        if m:
            por_pid.setdefault(int(m.group(1)), [0, 0])[1] += v

    nm = nomes(list(por_pid))
    linhas = sorted(por_pid.items(), key=lambda kv: -kv[1][0])
    print("\n=== VRAM FISICA por processo (o que disputa com o GSMDE) ===")
    print(f"  {'pid':>7}  {'processo':<24} {'fisica':>9} {'compart.':>10}")
    tot_f = tot_c = 0.0
    for pid, (f, c) in linhas:
        fg, cg = f / GB, c / GB
        tot_f += fg
        tot_c += cg
        if fg < 0.01 and cg < 0.01:
            continue
        print(f"  {pid:>7}  {nm.get(pid, '?'):<24} {fg:8.2f}G {cg:9.2f}G")
    print(f"  {'':>7}  {'TOTAL':<24} {tot_f:8.2f}G {tot_c:9.2f}G")
    print("\n  Fechar/minimizar os maiores ANTES de gerar devolve essa fisica.")
    print("  O spill DELES nao doi: navegador e desktop nao sao sensiveis a banda.")


if __name__ == "__main__":
    main()
