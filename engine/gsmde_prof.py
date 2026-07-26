"""
Instrumentacao do GSMDE: quem gasta o que, em tempo, CPU, RAM e VRAM.

A metrica que decide otimizacao nao e' "utilizacao da GPU %" — e' a razao
GPU/PAREDE por fase:

    gpu ~= parede  -> a placa esta calculando: so acelera com menos trabalho
    gpu << parede  -> a placa esta OCIOSA esperando: overhead, transferencia,
                      CPU, disco. E' aqui que offload/unload/cache pagam.

Eventos CUDA medem tempo de kernel de verdade (nao amostragem), entao a conta e'
exata. VRAM vem de mem_get_info (placa inteira, inclui outros processos), nao so
do alocador do torch — foi o que escondeu o derrame na primeira medicao.

Liga com GSMDE_PROF=1. Sem isso, custo zero.
"""
from __future__ import annotations
import os
import time
from collections import defaultdict
from contextlib import contextmanager

ON = os.environ.get("GSMDE_PROF", "") not in ("", "0")

_t0 = time.time()
_acc = defaultdict(lambda: {"n": 0, "parede": 0.0, "gpu": 0.0})
_psutil = None
_proc = None
if ON:
    try:
        import psutil
        _psutil = psutil
        _proc = psutil.Process()
        _proc.cpu_percent(None)          # 1a chamada arma o contador
    except Exception:
        pass


def _torch():
    import torch
    return torch


def hw():
    """(cpu%, ram_proc_GB, ram_sist_GB/total, vram_usada_GB/total, vram_torch_GB)"""
    d = {}
    if _psutil and _proc:
        d["cpu"] = _proc.cpu_percent(None)
        d["cpu_sist"] = _psutil.cpu_percent(None)
        d["ram"] = _proc.memory_info().rss / 2**30
        vm = _psutil.virtual_memory()
        d["ram_sist"] = (vm.total - vm.available) / 2**30
        d["ram_tot"] = vm.total / 2**30
    try:
        t = _torch()
        if t.cuda.is_available():
            livre, total = t.cuda.mem_get_info()
            d["vram"] = (total - livre) / 2**30      # a PLACA inteira, nao so o torch
            d["vram_tot"] = total / 2**30
            d["vram_torch"] = t.cuda.memory_allocated() / 2**30
            d["vram_resv"] = t.cuda.memory_reserved() / 2**30
    except Exception:
        pass
    return d


def linha(tag, parede=None, gpu=None):
    if not ON:
        return
    d = hw()
    # relogio de parede: sem isto nao da p/ cruzar a fase com o amostrador de filas
    # da GPU (que roda noutro processo) e descobrir QUEM usa a fila 3D.
    p = [f"[prof {time.strftime('%H:%M:%S')} {time.time()-_t0:6.1f}s] {tag:34s}"]
    if parede is not None:
        ocio = (1 - gpu / parede) * 100 if (gpu is not None and parede > 0.001) else None
        p.append(f"parede {parede*1000:7.1f}ms")
        if gpu is not None:
            p.append(f"gpu {gpu*1000:7.1f}ms")
            p.append(f"ocioso {ocio:5.1f}%" if ocio is not None else "")
    if "cpu" in d:
        p.append(f"cpu {d['cpu']:5.1f}%/{d['cpu_sist']:4.1f}%")
        p.append(f"ram {d['ram']:5.2f}G (sist {d['ram_sist']:5.2f}/{d['ram_tot']:.0f}G)")
    if "vram" in d:
        p.append(f"vram {d['vram']:5.2f}/{d['vram_tot']:.1f}G "
                 f"(torch {d['vram_torch']:5.2f}G resv {d['vram_resv']:5.2f}G)")
    print(" | ".join(x for x in p if x), flush=True)


@contextmanager
def fase(tag, acumula=True):
    """Mede parede + tempo REAL de kernel na GPU (eventos CUDA)."""
    if not ON:
        yield
        return
    t = _torch()
    usa_gpu = t.cuda.is_available()
    if usa_gpu:
        # Sincroniza ANTES: sem isso o evento pega kernels ja enfileirados de fases
        # anteriores e o "gpu" estoura o "parede" (ocioso negativo). Custa um pouco
        # de overhead, mas e' o que torna a atribuicao por fase honesta.
        t.cuda.synchronize()
        e0, e1 = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        e0.record()
    w0 = time.time()
    try:
        yield
    finally:
        parede = time.time() - w0
        gpu = None
        if usa_gpu:
            e1.record()
            t.cuda.synchronize()
            gpu = e0.elapsed_time(e1) / 1000.0
        if acumula:
            a = _acc[tag]
            a["n"] += 1
            a["parede"] += parede
            a["gpu"] += gpu or 0.0
        linha(tag, parede, gpu)


def report():
    if not ON or not _acc:
        return
    tot = sum(a["parede"] for a in _acc.values())
    print("\n" + "=" * 104, flush=True)
    print(f"{'fase':32s} {'n':>4} {'parede':>9} {'gpu':>9} {'ocioso':>7} {'%tempo':>7}  onde ataca",
          flush=True)
    print("-" * 104, flush=True)
    for tag, a in sorted(_acc.items(), key=lambda x: -x[1]["parede"]):
        oc = (1 - a["gpu"] / a["parede"]) * 100 if a["parede"] > 0.001 else 0
        pct = a["parede"] / tot * 100 if tot else 0
        dica = ("GPU calculando — so cai com menos trabalho" if oc < 25 else
                "MISTO — parte overhead" if oc < 60 else
                "GPU OCIOSA — overhead/transferencia/CPU: e' aqui que offload paga")
        print(f"{tag:32s} {a['n']:>4} {a['parede']:>8.2f}s {a['gpu']:>8.2f}s "
              f"{oc:>6.1f}% {pct:>6.1f}%  {dica}", flush=True)
    print("-" * 104, flush=True)
    gtot = sum(a["gpu"] for a in _acc.values())
    print(f"{'TOTAL':32s} {'':>4} {tot:>8.2f}s {gtot:>8.2f}s "
          f"{(1-gtot/tot)*100 if tot else 0:>6.1f}% {100.0:>6.1f}%", flush=True)
    print("=" * 104, flush=True)
