"""Acompanha a VRAM de um processo por FORA, sem tocar no que esta sendo medido.

POR QUE POR FORA
    Instrumentar o motor mudaria o que se quer observar: cada leitura dentro do
    laco compete pelo mesmo processo e mascara o efeito. Aqui um processo separado
    amostra os contadores do Windows e grava a linha do tempo; o motor nao sabe
    que esta sendo medido.

O QUE SAI
    Um CSV por amostra e um resumo no fim com os PICOS — que e' o que decide se
    a geracao coube ou derramou. Media nao serve: o estouro e' um instante.

USO
    python vram_monitor.py                # acha o worker do GSMDE sozinho
    python vram_monitor.py --pid 1234     # processo explicito
    python vram_monitor.py --seg 0.5      # intervalo de amostragem
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vram_sensor import VramSensor                              # noqa: E402


def acha_worker() -> list:
    """PIDs de python rodando o worker/motor do GSMDE."""
    try:
        saida = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'gsmde_worker|refine_gsmde' } | "
             "ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=25).stdout
        return [int(l) for l in saida.split() if l.strip().isdigit()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, action="append", default=None)
    ap.add_argument("--seg", type=float, default=1.0)
    ap.add_argument("--saida", default=r"D:\GSMDE\outputs\treino\vram_timeline.csv")
    ap.add_argument("--espera", type=float, default=180.0,
                    help="segundos aguardando o worker aparecer")
    args = ap.parse_args()

    pids = args.pid
    if not pids:
        print("procurando o worker do GSMDE...", flush=True)
        t0 = time.time()
        while time.time() - t0 < args.espera:
            pids = acha_worker()
            if pids:
                break
            time.sleep(2)
        if not pids:
            print("worker nao apareceu; monitorando so o total da placa", flush=True)
            pids = [os.getpid()]
    print(f"monitorando pids {pids}", flush=True)

    s = VramSensor(pids=pids, intervalo=0.0)      # sem cache: quem espaca e' o laco
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    campos = ["t", "dedicada_nossa", "dedicada_total", "outros", "compartilhada_total",
              "nonlocal_nosso", "spill_real", "reservado_torch", "livre", "livre_pct"]
    picos = {c: 0.0 for c in campos[1:]}
    minimo_livre = 1e9
    t0 = time.time()
    n = 0
    try:
        with open(args.saida, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            while True:
                d = s.ler(forcar=True)
                if not d["ok"]:
                    print("sensor indisponivel", flush=True)
                    return
                linha = {"t": round(time.time() - t0, 1)}
                # spill_real e' None quando medimos OUTRO processo (o torch do
                # monitor nao sabe o que o worker reservou). Fica em branco no CSV
                # em vez de virar um zero que passaria por "sem despejo".
                linha.update({c: ("" if d[c] is None else round(d[c], 3))
                              for c in campos[1:]})
                w.writerow(linha)
                fh.flush()
                for c in campos[1:]:
                    if d[c] is not None:
                        picos[c] = max(picos[c], d[c])
                minimo_livre = min(minimo_livre, d["livre"])
                n += 1
                if n % 10 == 0:
                    print(f"  t={linha['t']:6.0f}s  {s.resumo()}", flush=True)
                time.sleep(args.seg)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n=== PICOS em {n} amostras ({time.time()-t0:.0f}s) ===")
        print(f"  VRAM fisica NOSSA .......... {picos['dedicada_nossa']:6.2f} GB")
        print(f"  VRAM fisica TOTAL .......... {picos['dedicada_total']:6.2f} GB")
        print(f"  do SO (outros processos) ... {picos['outros']:6.2f} GB")
        print(f"  compartilhada total ........ {picos['compartilhada_total']:6.2f} GB")
        print(f"  nossa Non Local (pin+spill)  {picos['nonlocal_nosso']:6.2f} GB")
        print(f"  DESPEJO (reservado-fisico) . {picos['spill_real']:6.2f} GB")
        print(f"  minimo de fisica livre ..... {minimo_livre:6.2f} GB")
        print(f"\n  csv: {args.saida}")
        if picos["spill_real"] > 0.2:
            print("\n  >> DERRAMOU. A placa nao coube o que o torch achou que cabia.")
        else:
            print("\n  >> sem despejo relevante.")


if __name__ == "__main__":
    main()
