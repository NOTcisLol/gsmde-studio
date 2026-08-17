"""
Motor GSMDE reutilizavel (Secao 7.1 + 7.2 do doc).

Encapsula o que foi validado na geracao multi-centro:
  - especialistas dedicados carregados como adapters peft nomeados;
  - CFG DENTRO do proprio modelo de cada centro (uncond+cond do MESMO adapter) —
    misturar uncond do base com cond do especialista amplifica o deslocamento
    base->especialista em cfg x e estoura a imagem;
  - mascaras espaciais por cross-attention (attn2) -> cada centro pinta so o seu
    territorio; centros globais (style/camera) entram em blend leve;
  - txt2img e img2img (denoise parcial) para as camadas de refino;
  - TE 100% CPU, VAE tiled, yields anti-TDR.
"""
from __future__ import annotations
import contextlib
import json
import os
import re
import sys

# ANTES do torch: sem AOTriton o scaled_dot_product_attention do ROCm cai no
# backend "math", que materializa a matriz de atencao inteira. A 1152px isso
# custava 4.9GB de ativacoes -> pico 10.5GB numa placa de 8GB -> o driver
# derramava ~4.7GB na memoria compartilhada e cada forward passava a arrastar
# peso pelo PCIe: 78s/passo. Com AOTriton: pico 7.5GB e 6.6s/passo (12x).
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")

import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

from gsmde_prof import fase, linha, report   # GSMDE_PROF=1 liga; sem isso, custo zero

# Pesos: prefere o NVMe Gen4 (G:), cai no SATA (D:) se nao existir la. So a carga
# FRIA (apos reiniciar o Windows) sente a diferenca — depois o arquivo vive no cache
# de RAM do sistema e a leitura sai a ~7GB/s de qualquer disco. Medido: 20GB copiados
# em 110s; a carga fria da base cai de ~12s (SATA) p/ ~1s (Gen4).
# GSMDE_BASE / GSMDE_SPEC no ambiente sobrescrevem, p/ testar em outra maquina.
def _pesos(env, *candidatos):
    do_env = os.environ.get(env, "")
    if do_env and Path(do_env).exists():
        return do_env
    for c in candidatos:
        if Path(c).exists():
            return c
    return candidatos[-1]


BASE = _pesos("GSMDE_BASE",
              r"G:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors",
              r"D:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors")
# A biblioteca em D:\Models e' a canonica (2026-07-26). G:\Trainer_v13 ficou
# como copia PARADA e vinha primeiro nesta lista — com ela existindo, o motor
# carregava 29 centros velhos e ignorava os 7 mais novos, sem erro nenhum, so'
# imagem pior. Ordem invertida de proposito: o novo caminho ganha, e o G: fica
# apenas como reserva se alguem apagar a biblioteca.
SPEC = Path(_pesos("GSMDE_SPEC",
                   r"D:\Models\gsmde\specialists",
                   r"G:\Trainer_v13\models\specialists"))
# ---------------------------------------------------------------------------
# Orcamento de RAM para o buffer de centros
# ---------------------------------------------------------------------------
# Os centros moram na RAM e sobem p/ a VRAM um por vez. Quem limita quantos
# cabem, portanto, e' a RAM — nao a VRAM. Este teto e' EXPLICITO e do usuario:
# sem ele o processo cresce ate o Windows comecar a paginar para o disco, e ai
# tudo (inclusive o que nao e' nosso) fica lento sem nenhum aviso.
#
# DOIS NUMEROS SEPARADOS, e a distincao importa:
#
#   1. O BUFFER e' escolha do usuario. Ele pede 25%, 50%, o que quiser — nao ha
#      validacao contra a RAM livre no momento de configurar. Faria pouco
#      sentido: livre e' um valor instantaneo, e configurar com o navegador
#      aberto daria um teto diferente de configurar com ele fechado.
#
#   2. O TETO e' sobre o USO TOTAL DO SISTEMA, medido em tempo de execucao, e
#      independe da configuracao. E' o que cobre o caso real: o usuario escolheu
#      50%, comecou a gerar, abriu outro programa, e agora a maquina inteira
#      esta perto de saturar. Quem manda parar e' o estado da maquina, nao o que
#      foi pedido antes.
#
# Acima de 80% avisa; acima de 83% e' erro critico e a geracao para. A folga
# entre os dois existe para dar chance de fechar algo antes do corte — e o corte
# vem ANTES do swap, nao depois: uma vez em swap, cada paginacao de centro vira
# leitura de disco e um passo que levava segundos passa a levar minutos.
USO_ALERTA = 0.80
USO_CRITICO = 0.83
RAM_BUFFER_PADRAO = 0.25          # do total, quando o usuario nao define


class RamInsuficiente(RuntimeError):
    """Erro CRITICO: o sistema saturou a RAM. Bloqueia/aborta a geracao."""


def _ram():
    import psutil
    m = psutil.virtual_memory()
    return m.total / 2**30, m.available / 2**30, m.percent / 100.0


def orcamento_ram(pedido_gb=None):
    """Buffer pedido pelo usuario, SEM veto. (orcamento, total, livre, uso)."""
    total, livre, uso = _ram()
    if pedido_gb in (None, 0, ""):
        pedido_gb = float(os.environ.get("GSMDE_RAM_BUFFER_GB", 0)) or \
                    total * RAM_BUFFER_PADRAO
    return float(pedido_gb), total, livre, uso


# Faixas de estouro do buffer. NENHUMA delas interrompe: sao ruido deliberado
# p/ o usuario arrumar a configuracao. O unico que para a geracao e' o teto de
# 83% do sistema, porque ali o risco e' a maquina inteira entrar em swap.
OVERLOAD_N1 = 0.00   # passou do buffer
OVERLOAD_N2 = 0.20   # passou 20% alem do buffer


def nivel_overload(preciso_gb, orcamento_gb):
    """0 = cabe | 1 = estourou | 2 = estourou com folga (>20%)."""
    if orcamento_gb <= 0:
        return {"nivel": 0, "excesso": 0.0, "recado": ""}
    excesso = (preciso_gb / orcamento_gb) - 1.0
    if excesso > OVERLOAD_N2:
        return {"nivel": 2, "excesso": excesso,
                "recado": "Reduza 'Max. centros' ou aumente o buffer de RAM. "
                          "A geracao SEGUE, mas com o sistema perto do limite "
                          "ela fica lenta e sujeita a parada em 83%."}
    if excesso > OVERLOAD_N1:
        return {"nivel": 1, "excesso": excesso,
                "recado": "Cabe apertado. Vale aumentar o buffer ou tirar um centro."}
    return {"nivel": 0, "excesso": excesso, "recado": ""}


def confere_ram(etapa="", extra_gb=0.0):
    """Teto ABSOLUTO sobre o uso do sistema. Chamada na carga e entre passos.

    `extra_gb` projeta o que ainda vai ser alocado (ex.: os centros que faltam
    ler), p/ barrar ANTES de estourar em vez de depois."""
    total, livre, uso = _ram()
    projetado = uso + (extra_gb / total if total else 0)
    onde = f" [{etapa}]" if etapa else ""
    if projetado >= USO_CRITICO:
        raise RamInsuficiente(
            f"RAM do sistema em {projetado:.0%}{onde} (teto critico "
            f"{USO_CRITICO:.0%}). Livre: {livre:.1f} de {total:.1f} GB. "
            f"Feche algum programa — parando aqui para a maquina nao entrar em "
            f"swap, onde cada passo levaria minutos.")
    if projetado >= USO_ALERTA:
        print(f"[gsmde] AVISO: RAM do sistema em {projetado:.0%}{onde} "
              f"(critico em {USO_CRITICO:.0%}, livre {livre:.1f} GB)", flush=True)
    return {"uso": projetado, "livre": livre, "total": total,
            "alerta": projetado >= USO_ALERTA}


# Pastas de centros BAIXADOS (HF/CivitAI/biblioteca local). Cada ficha em
# specialists_<origem>/<nome>.json aponta o .safetensors com "arquivo" —
# diferente dos treinados aqui, que seguem o padrao <SPEC>/<nome>/specialist.
FICHAS_EXTERNAS = [Path(r"D:\GSMDE\trainer") / f"specialists_{o}"
                   for o in ("local", "hf", "civitai")]
_CACHE_EXTERNO: dict = {}


def _indice_externo() -> dict:
    """nome do centro -> caminho do peso, lido das fichas geradas pelo catalogo."""
    if _CACHE_EXTERNO:
        return _CACHE_EXTERNO
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
                _CACHE_EXTERNO[f.stem] = Path(arq)
    return _CACHE_EXTERNO


def _formato_lora(st: dict) -> str:
    """'peft' (treinado aqui) ou 'kohya' (baixado). O carregamento difere: o
    diffusers converte kohya sozinho, mas so' a partir do dict cru."""
    for k in st:
        if "lora_down" in k or "lora_up" in k or k.startswith("lora_unet"):
            return "kohya"
        if "lora_A" in k or "lora_B" in k:
            return "peft"
    return "peft"


def te_modo():
    """Como o TEXT ENCODER usa a placa. Independente do offload do UNet.

      cpu         (padrao) nunca toca a GPU
      residente   sobe no init e FICA — soma ~1,4GB de CLIP+CLIP-G ao UNet
      sequencial  sobe SO' para encodar e desce antes do denoise

    O sequencial existe porque as duas fases nao competem no tempo: o prompt e'
    encodado UMA vez, antes do primeiro passo, e depois o TE nao faz mais nada.
    Deixa-lo residente e' pagar VRAM o denoise inteiro por um trabalho que durou
    segundos. Numa placa de 8GB com o UNet em ~5,2GB, residente derrama; o
    sequencial pode caber porque o UNet ainda nem esta no lugar quando o TE sobe.
    """
    return os.environ.get("GSMDE_TE_OFFLOAD", "cpu").strip().lower()


@contextlib.contextmanager
def te_na_placa(pipe, dev, dtype=torch.float16):
    """Sobe os dois text encoders, entrega, e devolve p/ a CPU liberando a VRAM."""
    modo = te_modo()
    if modo != "sequencial" or dev is None or str(dev) == "cpu":
        yield None
        return
    t0 = time.time()
    for te in (pipe.text_encoder, pipe.text_encoder_2):
        if te is not None:
            te.to(dev, dtype)
    subiu = time.time() - t0
    try:
        yield dev
    finally:
        for te in (pipe.text_encoder, pipe.text_encoder_2):
            if te is not None:
                te.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[gsmde] TE sequencial: subiu em {subiu:.1f}s, "
              f"desceu e liberou a VRAM", flush=True)


def _carrega_centro(pipe, nome, st, formato):
    """Anexa um centro ao UNet. Dois formatos, dois caminhos.

    peft  (treinado aqui): o prefixo 'unet.' faz o diffusers rotear certo.
    kohya (baixado): NAO passa por load_lora_weights. Medido — ele chama o
      carregador do TEXT ENCODER mesmo quando o LoRA nao tem chave nenhuma de
      TE (o IDK_depth-of-field tem 0 de 1680), e la' o rank_dict sai vazio e
      estoura `IndexError: list index out of range` em get_peft_kwargs. Como
      toda chave convertida do kohya sai com prefixo 'unet.', o dict de TE e'
      SEMPRE vazio: o erro era garantido para qualquer LoRA baixado.
      load_lora_adapter e' o carregador do proprio UNet — nao sabe o que e'
      text encoder, entao nao ha caminho para quebrar. E casa com o desenho:
      centro do GSMDE atua no UNet; quem manda no texto e' o roteador.
    """
    if formato == "peft":
        pipe.load_lora_weights({f"unet.{k}": v for k, v in st.items()},
                               adapter_name=nome)
    else:
        conv, alphas = type(pipe).lora_state_dict(st)
        pipe.unet.load_lora_adapter(conv, prefix="unet", network_alphas=alphas,
                                    adapter_name=nome)


def _localiza_centro(nome, spec_root):
    """Treinado aqui tem prioridade; externo entra pelo indice das fichas.
    Sem isto o motor so' sabia montar <spec_root>/<nome>/specialist.safetensors,
    entao qualquer LoRA baixado era invisivel — o roteador ate escolhia o
    centro, e o load estourava."""
    p = Path(spec_root) / nome / "specialist.safetensors"
    if p.exists():
        return p, "peft"
    ext = _indice_externo().get(nome)
    if ext is None:
        return None, ""
    try:
        from safetensors import safe_open
        with safe_open(str(ext), framework="pt") as f:
            fmt = _formato_lora({k: None for k in f.keys()})
    except Exception:
        fmt = "kohya"
    return ext, fmt


NEG = ("worst quality, low quality, bad anatomy, bad hands, extra digits, jpeg artifacts, "
       "signature, watermark, blurry, lowres, deformed")


def yield_gpu(ms=40):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if ms > 0:
        time.sleep(ms / 1000.0)


def parse_centers(spec: str):
    """'person:girl,woman;scenary:park,trees' -> [('person',['girl','woman']),...]

    INSTANCIAS: o mesmo especialista pode virar VARIOS centros, cada um com seu
    territorio — 'person#1:vampire,fangs;person#2:werewolf,fur'. E' a resposta do
    GSMDE ao attribute bleeding: no monolitico, 'um vampiro e um lobisomem' sai
    hibrido porque um so campo de atencao serve os dois. Aqui sao duas mascaras
    disjuntas; o lobisomem nao alcanca o pixel do vampiro. Nao e' restringir um
    modelo — e' rodar o mesmo especialista duas vezes, em dois lugares.
    """
    out = []
    for part in (spec or "").split(";"):
        if not part.strip():
            continue
        nome, words = part.split(":", 1)
        out.append((nome.strip(), [w.strip() for w in words.split(",") if w.strip()]))
    return out


def base_do_centro(cid: str) -> str:
    """'person#2' -> 'person' (o adapter). O '#n' so distingue a instancia."""
    return cid.split("#", 1)[0].split("@", 1)[0]


def parse_regiao(cid: str):
    """'person#1@0,0,0.5,1' -> ('person#1', (0,0,0.5,1)). Sem @ -> regiao None."""
    if "@" in cid:
        nome, box = cid.split("@", 1)
        try:
            x0, y0, x1, y1 = (float(v) for v in box.split(","))
            return nome, (x0, y0, x1, y1)
        except ValueError:
            pass
    return cid, None


def auto_regioes(ids):
    """Instancias do MESMO modelo (person#1, person#2) se dividem a tela sozinhas:
    2 -> esquerda/direita, 3 -> tercos, 4 -> quadrantes. Uma unica instancia de um
    modelo (scenary) NAO ganha regiao — pinta a tela toda. E' o que resolve o
    attribute bleeding: cada personagem tem territorio disjunto POR CONSTRUCAO, nao
    por softmax de uma atencao ja contaminada.

    Regiao explicita (@box) sempre vence a automatica.
    """
    from collections import defaultdict
    grupos = defaultdict(list)
    reg = {}
    for cid in ids:
        nome, box = parse_regiao(cid)
        if box is not None:
            reg[nome] = box
        else:
            grupos[base_do_centro(nome)].append(nome)
    for base, insts in grupos.items():
        if len(insts) < 2:
            continue                       # 1 instancia = tela toda, sem regiao
        k = len(insts)
        cols = 2 if k <= 4 else 3
        rows = (k + cols - 1) // cols
        for i, cid in enumerate(insts):
            if cid in reg:
                continue
            c, r = i % cols, i // cols
            # margem: os personagens nao tomam a tela inteira, senao o fundo (centro
            # sem caixa) fica sem lugar. Uma faixa vertical de 0.08-0.98 deixa ceu em
            # cima e chao embaixo p/ o scenary. Largura tambem com folga p/ costurar.
            # faixa morta entre personagens (0.04 de gap) + margem p/ o fundo
            reg[cid] = (c / cols + 0.04, r / rows + 0.08,
                        (c + 1) / cols - 0.04, (r + 1) / rows - 0.02)
    return reg


class RegionalSelfAttn:
    """Processor de SELF-attention (attn1) com mascara de regiao.

    A PECA QUE FALTAVA. A cross-attention ja isola O QUE cada centro pinta (a fatia).
    Mas a self-attention ve o latente INTEIRO: o vampiro-esquerda enxerga o
    lobisomem-direita e, com o vies de 'um personagem' que o SDXL aprendeu, costura
    os dois num corpo so. A mascara de saida recorta a metade certa, mas ela ja foi
    composta como parte de UMA criatura.

    Aqui, durante o forward de um centro com regiao R, os pixels so atendem a pixels
    DENTRO de R. O centro compoe entao um sujeito inteiro e auto-contido, sem ver (e
    sem se fundir com) o que os vizinhos pintam. E' isto que diz ao contexto global
    que sao personagens SEPARADOS. Centro sem regiao (scenary) -> self-attn normal,
    que e' a cola global do fundo.

    Delega tudo ao processor original e so injeta o bias — assim nao reimplemento a
    logica do attn (residual, norm, LoRA nos to_q/k/v continuam valendo).
    """
    def __init__(self, inner):
        self.inner = inner
        self.region = None        # [H,W] em {0..1} na resolucao do latente, ou None

    def _hw(self, n):
        """(h,w) do bloco com n posicoes, descendo a piramide do latente (ceil a cada
        downsample, como a conv stride 2 da UNet). O criterio antigo era sqrt(n) e
        exigia bloco QUADRADO: em latente quadrado isso casa em todo nivel, mas em
        imagem 16:9 (ou qualquer proporcao livre) nao casa em NENHUM — e a isolacao
        da self-attn se desligava inteira, sem aviso."""
        if self.region is None:
            return None
        h, w = self.region.shape
        for _ in range(5):
            if h * w == n:
                return (h, w)
            h, w = -(-h // 2), -(-w // 2)
        return None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kw):
        # sem regiao, ou cross-attn, ou ja tem mask -> processor original intacto.
        cross = encoder_hidden_states is not None
        n = hidden_states.shape[1]
        hw = None if cross else self._hw(n)
        if cross or self.region is None or attention_mask is not None or hw is None:
            return self.inner(attn, hidden_states, encoder_hidden_states,
                              attention_mask, **kw)
        # SELF-attn com regiao: computa direto (delegar quebra o reshape do
        # prepare_attention_mask do diffusers). Replica o AttnProcessor2_0 padrao,
        # com um bias que suprime as KEYS fora da regiao — assim o centro compoe um
        # sujeito auto-contido, sem enxergar os vizinhos. LoRA nos to_q/k/v vale
        # porque chamamos attn.to_q(...) (os Linear ja embrulhados por peft).
        b, _, _ = hidden_states.shape
        res = hidden_states
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        h = attn.heads
        d = q.shape[-1] // h
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)
        rm = F.interpolate(self.region.view(1, 1, *self.region.shape).float(),
                           size=hw, mode="bilinear",
                           align_corners=False).reshape(n)
        bias = rm.clamp(1e-3).log().to(q.dtype).view(1, 1, 1, n)   # -inf-ish fora
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(b, n, h * d).to(res.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if getattr(attn, "residual_connection", False):
            out = out + res
        return out / getattr(attn, "rescale_output_factor", 1.0)


class XAttnProbe:
    """Captura os mapas token->pixel dos blocos attn2 do UNet."""
    def __init__(self, unet):
        self.maps, self.hooks, self.enabled = [], [], False
        self.target_npix = None      # compat: casar UMA resolucao exata
        self.dims = {}               # n_pix -> (h,w) das resolucoes aceitas
        self.avisou = False
        for name, mod in unet.named_modules():
            if name.endswith("attn2"):
                # with_kwargs=True e' ESSENCIAL: o BasicTransformerBlock do diffusers
                # chama attn2(hs, encoder_hidden_states=..., ...) por KEYWORD, e o
                # hook classico so recebe posicionais. Sem isto 'inp' tinha so o
                # hidden_states, o codigo caia em ehs=hs e o to_k(1280) recebia onde
                # espera 2048 -> excecao de shape engolida pelo except -> ZERO mapas.
                try:
                    self.hooks.append(
                        mod.register_forward_hook(self._mk(True), with_kwargs=True))
                except TypeError:            # torch antigo: sem with_kwargs
                    self.hooks.append(mod.register_forward_hook(self._mk(False)))

    def _mk(self, com_kwargs=False):
        def hook(module, inp, *resto):
            if not self.enabled:
                return
            kwargs = resto[0] if com_kwargs and len(resto) > 1 else {}
            hs = inp[0]
            # Descartar cedo: recomputar QxK em bloco que nao entra na mascara so
            # queima VRAM. 'dims' lista as resolucoes ACEITAS (a piramide do
            # latente); target_npix e' o modo antigo, de uma resolucao so.
            if self.dims:
                if hs.shape[1] not in self.dims:
                    return
            elif self.target_npix and hs.shape[1] != self.target_npix:
                return
            ehs = kwargs.get("encoder_hidden_states")
            if ehs is None and len(inp) > 1:
                ehs = inp[1]
            if ehs is None:
                return          # sem contexto de texto nao ha mapa token->pixel
            # com IP-Adapter o diffusers empacota (texto, imagem) numa TUPLA; a
            # mascara vem das ancoras de TEXTO, entao pega o primeiro. Sem isto o
            # to_k recebia a tupla e a sonda morria — mascaras caiam no uniforme
            # justamente quando o contexto global era ligado.
            if isinstance(ehs, (tuple, list)):
                ehs = ehs[0]
            try:
                q, k = module.to_q(hs), module.to_k(ehs)
                h = module.heads
                b, n, d = q.shape
                q = q.view(b, n, h, d // h).transpose(1, 2)
                k = k.view(b, k.shape[1], h, d // h).transpose(1, 2)
                attn = ((q @ k.transpose(-1, -2)) * (1.0 / (d // h) ** 0.5)).softmax(-1)
                self.maps.append(attn.mean(1).detach())      # [b,n_pix,n_tok]
                del attn, q, k
            except Exception as e:
                # NAO engole em silencio: era o except mudo que escondia a sonda
                # quebrada — o pipeline seguia "funcionando" com mascara uniforme
                # 1/N e ninguem ficava sabendo. Avisa UMA vez por sessao.
                if not self.avisou:
                    self.avisou = True
                    print(f"[gsmde] sonda attn2 falhou ({type(e).__name__}: {e}) — "
                          f"sem mascaras de atencao nesta sessao", flush=True)
        return hook

    def close(self):
        for h in self.hooks:
            h.remove()


MANIFESTO = Path(r"D:\Models\gsmde\backbones.json")


def backbones_disponiveis():
    """Lista do manifesto, filtrada pelo que existe MESMO no disco.

    O manifesto e' a fonte unica para a UI e para o motor. Se a UI listasse por conta
    propria e o motor resolvesse por conta propria, os dois divergiriam em silencio no
    dia em que um checkpoint fosse apagado — e o sintoma seria 'escolhi d3u10 e saiu
    outra coisa', que ninguem liga a um select desatualizado.
    """
    import json
    try:
        d = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    except Exception:
        return [{"id": "original", "rotulo": "Original", "recomendada": True}]
    fora = [b for b in d.get("backbones", [])
            if b["id"] != "original" and not Path(b["arquivo"] or "").exists()]
    for b in fora:
        print(f"[gsmde] backbone '{b['id']}' esta no manifesto mas nao no disco",
              flush=True)
    return [b for b in d.get("backbones", []) if b not in fora]


class GSMDE:
    def _troca_unet(self, pipe, variante):
        """Substitui a UNet da pipeline pela do checkpoint da variante."""
        from diffusers import UNet2DConditionModel
        from gera_student import _reajusta_por_forma
        alvo = next((b for b in backbones_disponiveis() if b["id"] == variante), None)
        if not alvo or not alvo.get("arquivo"):
            print(f"[gsmde] variante '{variante}' indisponivel; seguindo na original",
                  flush=True)
            self.variante = "original"
            return
        ck = torch.load(alvo["arquivo"], map_location="cpu", weights_only=False)
        u = UNet2DConditionModel.from_config(ck.get("cfg") or ck.get("config"))
        # A config do diffusers nao representa toda cirurgia: o corte de FFN encolhe
        # matrizes que a config declara em tamanho original. Redimensionar pela forma
        # do proprio state_dict cobre isso e qualquer cirurgia futura de largura.
        _reajusta_por_forma(u, ck["state_dict"])
        u.load_state_dict(ck["state_dict"], strict=True)
        # a troca acontece com TUDO na CPU; quem sobe para a placa e' o fluxo normal
        pipe.unet = u.to(torch.float16)
        self.variante_info = {k: alvo.get(k) for k in
                              ("id", "gb_fp16", "params_B", "lora_pct")}
        print(f"[gsmde] backbone {variante}: {alvo.get('gb_fp16')} GB, "
              f"{alvo.get('lora_pct')}% dos modulos de LoRA encaixam", flush=True)
        del ck

    def __init__(self, centers, globals_=(), adapter_scale=0.8, cfg=5.5,
                 yield_ms=40, base=BASE, spec_root=SPEC, device=None,
                 page_adapters=True, vae_offload=True, vae_tile=512,
                 vram_reserva=0.0, variante="original"):
        from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
        from safetensors.torch import load_file
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.regional = centers                     # [(nome,[palavras])]
        self.globs = list(globals_)
        # adaptadores que viajam DENTRO da passada de outro; ver guided()
        self.caronas = []
        self.adapter_scale, self.cfg, self.yield_ms = adapter_scale, cfg, yield_ms
        self.vae_offload = vae_offload
        self.fatias = {}
        self.regioes = {}
        # prompt weighting / contexto+foco por centro
        self.weighted_prompt = True       # enfase/wildcard/BREAK nos CLIP
        self._pw_seed = 0                  # semente dos wildcards {A|B}
        self.auto_center_focus = True      # cada centro: foco erguido, global abaixado
        self.center_focus_weight = 1.15    # peso do FOCO do centro (>1)
        self.center_context_weight = 0.90  # peso do resto do prompt no centro (<1; 1.0=off)
        # backbone dominante: SEM isto, o foco por centro INUNDA (o centro substitui a
        # base na sua regiao). Com isto, backbone=1.0 e cada centro so da nudge escalado
        # pela cobertura. Default de instancia; denoise(backbone_assert=) sobrepoe.
        self.backbone_assert = 0.6

        linha("INICIO")
        with fase("load: base do disco"):
            pipe = StableDiffusionXLPipeline.from_single_file(base, torch_dtype=torch.float16)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        # ------------------------------------------------- backbone reduzido
        #
        # Troca a UNet por uma variante de corte assimetrico. Medido em 12/08 numa
        # bancada de 54 geracoes a 1024 (ver D:/GSMDE/docs/relatorio.md):
        #
        #   a ORIGINAL e' a unica que transborda para a RAM (0,30 GB contra 0,08 de
        #   todas as reduzidas). Com 4,78 GB de UNet mais ativacao a 1024, a placa de
        #   8 GB nao fecha a conta. Encolher nao e' so' economia — e' o que tira a
        #   geracao da zona de despejo.
        #
        # O CUSTO e' que nem todo modulo de LoRA encontra alvo: os que somem estao em
        # down_blocks.2, e o efeito aparece em detalhe de material e objeto pequeno,
        # nao em pessoa. Por isso a escolha fica com o usuario, com o numero a vista.
        self.variante = variante or "original"
        self.variante_info = {}
        if self.variante != "original":
            with fase(f"load: backbone {self.variante}"):
                self._troca_unet(pipe, self.variante)
        # O UNet SO' SOBE DEPOIS DOS CENTROS.
        #
        # Antes ele ia para a placa aqui, e o load_lora_weights la' embaixo
        # anexava cada adapter a um UNet que JA estava na GPU — ou seja, todo
        # centro nascia na VRAM. Medido: 5,14 GB de UNet + 6 centros de ~0,5 GB
        # = pico de 7,73 de 8,00 GB, com derrame de 0,2 -> 9,2 GB na
        # compartilhada, ANTES de a geracao comecar.
        #
        # A politica de paginacao ate' corrigia depois (devolvia tudo p/ a RAM,
        # media, trazia so' o que cabia), mas tarde: o derrame ja tinha
        # acontecido. E o sintoma final era `unspecified launch failure`, que
        # neste build do ROCm e' como a falta de VRAM se manifesta — parecia bug
        # de kernel e mandou a investigacao para o AOTRITON e para a paginacao.
        #
        # Carregando com o UNet na CPU, os adapters nascem na RAM (que e' onde
        # a paginacao quer que eles fiquem) e a placa nunca ve os dois juntos.
        _subir_unet_depois = True
        with fase("load: vae + te -> cpu"):
            # enable_tiling() sozinho e' DECORATIVO: o limiar padrao e'
            # tile_sample_min_size=1024, entao a 768 o VAE nao tila nada e faz o
            # decode inteiro de uma vez -> 6.69GB na placa com a UNet residente.
            # A partir dai o dwm nao consegue VRAM, a fila 3D vai a 100% e o
            # Windows trava (o compute continua baixo — nao e' o modelo, e' o
            # compositor faminto). Baixar o limiar e' o que faz o tiled VAE existir.
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            pipe.vae.tile_sample_min_size = vae_tile
            pipe.vae.tile_latent_min_size = max(32, vae_tile // 8)
            pipe.vae.tile_overlap_factor = 0.25
            pipe.vae.to("cpu" if vae_offload else self.dev)
            # bf16, NAO fp32. O build ROCm do torch vem sem MKL, entao o GEMM fp32
            # na CPU cai num caminho ingenuo: 75-90s para encodar um prompt (91% do
            # tempo de um job!). O oneDNN tem caminho nativo bf16 -> 1.5s, 61x mais
            # rapido, sem tocar na VRAM. E bf16 tem o expoente do fp32 e a mantissa
            # do fp16 — que e' o dtype em que o SDXL padrao roda os TEs de qualquer jeito.
            torch.set_num_threads(os.cpu_count() or 8)
            self.te_device, te_dtype = self._resolve_te()
            # Em 'sequencial' o TE fica na CPU aqui e so' sobe na hora de
            # encodar — subir agora seria ocupar a placa durante todo o denoise
            # por um trabalho que ja terminou.
            if te_modo() == "sequencial" and torch.cuda.is_available():
                self.te_device_calc = torch.device("cuda")
                self.te_device = torch.device("cpu")
                te_dtype = torch.float32 if te_dtype == torch.bfloat16 else te_dtype
                print("[gsmde] TE em modo SEQUENCIAL: sobe so' para o encode",
                      flush=True)
            else:
                self.te_device_calc = self.te_device
            pipe.text_encoder.to(self.te_device, te_dtype)
            pipe.text_encoder_2.to(self.te_device, te_dtype)
        self.pipe = pipe

        # instancias compartilham o adapter: person#1 e person#2 carregam 'person'
        # uma vez so. Duas mascaras, um peso — o custo de VRAM nao dobra.
        unicos = []
        for c in [base_do_centro(n) for n, _ in self.regional] + self.globs:
            if c not in unicos:
                unicos.append(c)
        # --- orcamento de RAM, ANTES de ler qualquer peso ---
        # Somar o tamanho em disco custa um stat por arquivo; carregar e depois
        # descobrir que nao cabe custa minutos e pode levar a maquina ao swap.
        orc, ram_total, ram_livre, ram_uso = orcamento_ram()
        pesos = []
        for c in unicos:
            cam, fmt = _localiza_centro(c, spec_root)
            pesos.append((c, cam, fmt, (cam.stat().st_size / 2**30) if cam else 0.0))
        preciso = sum(g for _, _, _, g in pesos)
        print(f"[gsmde] buffer de centros: {preciso:.2f} GB para {len(pesos)} centro(s) | "
              f"orcamento {orc:.1f} GB | RAM {ram_uso:.0%} usada, "
              f"{ram_livre:.1f}/{ram_total:.1f} GB livres", flush=True)

        # 1) teto ABSOLUTO do sistema, projetando o que ainda sera lido.
        #    Independe do buffer configurado — e' o estado da maquina que manda.
        confere_ram("carga dos centros", extra_gb=preciso)

        # 2) buffer do usuario: SO' ALERTA, nunca corta nem bloqueia.
        #    Ele escolheu o numero; se escolheu apertado demais, quem tem de
        #    corrigir e' ele. Cortar centros pelas costas daria uma imagem pior
        #    sem explicacao — o alerta incomoda, mas e' honesto.
        ov = nivel_overload(preciso, orc)
        if ov["nivel"]:
            print(f"[gsmde] {'!' * ov['nivel']} OVERLOAD nivel {ov['nivel']}: "
                  f"{len(pesos)} centros pedem {preciso:.2f} GB, "
                  f"{ov['excesso']:.0%} acima do buffer de {orc:.1f} GB. "
                  f"{ov['recado']}", flush=True)

        for c, caminho, formato, _gb in pesos:
            with fase("load: especialista (709MB)"):
                if caminho is None:
                    print(f"[gsmde] centro '{c}' nao encontrado — pulando", flush=True)
                    continue
                # Um LoRA baixado que nao carrega NAO PODE derrubar a rodada.
                # Sao pesos de terceiros, com receitas variadas; alguns nao
                # encaixam nesta arquitetura. Medido: 748cmSDXL toca camadas
                # fora da atencao (emb_layers do time embedding) e o conversor
                # do diffusers nao as mapeia. Pular esse e seguir com os outros
                # e' melhor que perder a geracao inteira por causa de um.
                try:
                    _carrega_centro(pipe, c, load_file(str(caminho)), formato)
                except Exception as e:
                    print(f"[gsmde] centro '{c}' NAO carregou "
                          f"({type(e).__name__}: {str(e)[:110]}) — seguindo sem ele",
                          flush=True)
                    continue
                print(f"[gsmde] centro '{c}' <- {formato} {caminho.name}", flush=True)

        self.probe = XAttnProbe(pipe.unet)
        # self-attn regional: envolve o processor de cada attn1 (nao attn2). Um so
        # processor compartilhado; a regiao ativa e' trocada por centro no denoise.
        self._embrulha_selfattn()

        # --- Paginacao explicita dos nichos (Secao 6) ---
        # So UM adapter esta ativo por forward, mas todos ficavam residentes:
        # UNet 5GB + N x 709MB + ativacoes estourava os 8GB e o driver passava a
        # paginar sozinho, ao acaso, pelo PCIe. Aqui os nichos moram na RAM (o
        # sistema tem 40GB) e so o centro ativo sobe — deterministico, uma vez por
        # centro por passo, em vez de thrash do driver.
        self.paged = {}
        for mod in pipe.unet.modules():
            for attr in ("lora_A", "lora_B"):
                d = getattr(mod, attr, None)
                if d is None:
                    continue
                for nome in list(d.keys()):
                    self.paged.setdefault(nome, []).append(d[nome])
        # Mascaras de territorio por cross-attention. Declarado aqui (e nao so lido
        # por getattr) porque o worker so repassa opcao que o motor ANUNCIA ter.
        # False = todo centro pinta a tela toda com peso 1/N — que era, sem querer,
        # o comportamento efetivo enquanto a sonda estava quebrada.
        self.usar_mascaras = True
        self.residentes = set()
        # A politica esta no caminho critico de TODO carregamento: se ela falhar, o
        # motor inteiro nao sobe. Otimizacao nao pode derrubar o basico — qualquer
        # erro aqui volta ao comportamento historico (paginar tudo).
        if _subir_unet_depois:
            with fase("load: unet -> gpu (apos os centros)"):
                # `.to()` percorre a arvore de submodulos, e os nichos JA ESTAO
                # pendurados na UNet — subiriam junto com ela. Era o furo na
                # estrategia descrita nas linhas ~611-622: carregar com a UNet na
                # CPU faz os adapters nascerem na RAM, mas este `.to()` os levava
                # de volta. Medido em 16/08, 1024 com 6 centros e 2 caronas: pico
                # de 8,59 GB na carga (4,78 da base + 3,81 dos nichos) numa placa
                # de 8,00 -> 2,98 GB de despejo ANTES do primeiro passo. O laco em
                # si estava limpo (0,10 GB), e paginar os nichos nao adiantava:
                # a politica so' roda depois, tarde demais.
                #
                # Desanexa os nichos, sobe so' a base, reanexa. Estado final e' o
                # mesmo de antes (nichos na RAM, a politica decide quem sobe) —
                # muda apenas que a placa nunca ve os dois somados.
                guardados = []
                for mod in pipe.unet.modules():
                    for attr in ("lora_A", "lora_B"):
                        sub = mod._modules.get(attr)
                        if isinstance(sub, torch.nn.Module):
                            guardados.append((mod, attr, sub))
                            mod._modules[attr] = torch.nn.ModuleDict()
                try:
                    pipe.unet.to(self.dev)
                finally:
                    # `finally`: se o .to() falhar no meio, a UNet nao pode ficar
                    # sem os adapters — seria um motor mudo, sem erro visivel.
                    for mod, attr, sub in guardados:
                        mod._modules[attr] = sub
                print(f"[gsmde] base -> placa sem os {len(guardados)} modulos de "
                      f"nicho (eles ficam na RAM ate' a politica decidir)", flush=True)
        # ANTES da politica: ela move nichos, e precisa mover blocos, nao modulos.
        try:
            self._monta_blocos()
        except Exception as e:
            print(f"[gsmde] blocos contiguos indisponiveis ({e}); "
                  f"paginando modulo a modulo", flush=True)
            self.blocos = {}
        try:
            self._vram_reserva = vram_reserva
            self.page_adapters = self._politica_vram(page_adapters, vram_reserva)
        except Exception as e:
            print(f"[gsmde] politica de paginacao falhou ({e}); paginando tudo",
                  flush=True)
            self.residentes = set()
            self.page_adapters = True
        # (o posicionamento dos nichos ja foi feito por _politica_vram: ela pagina
        # tudo, mede a folga REAL e traz de volta so o que cabe)
        n_mods = sum(len(v) for v in self.paged.values())
        print(f"[gsmde] regionais={[n for n,_ in self.regional]} globais={self.globs} "
              f"scale={adapter_scale} | paginacao={'RAM' if page_adapters else 'VRAM'} "
              f"({len(self.paged)} nichos, {n_mods} modulos)", flush=True)
        self.vram("apos carregar")

    def unet_offload(self, fora=True):
        """Tira/devolve a UNet (5,6GB) da placa. VRAM SEQUENCIAL: um modelo de cada
        vez. Deixar a UNet residente enquanto o ESRGAN trabalha estoura os 8GB, o
        driver derrama na compartilhada e — como o desktop divide a mesma GPU — o
        Windows inteiro trava. E' o que o SD.Next faz e por isso ele nao trava."""
        with fase(f"unet -> {'RAM' if fora else 'VRAM'}"):
            self.pipe.unet.to("cpu" if fora else self.dev)
            if fora:
                torch.cuda.empty_cache()

    def _set_regiao_selfattn(self, mask):
        """Ativa a mascara de regiao na self-attn de todos os attn1 (mask=None limpa)."""
        for w in self.selfattn:
            w.region = mask

    def globais_como_carona(self, peso=None):
        """Move os globais de passada propria para carona, e devolve quantos moveu.

        POR QUE ISTO E' CORRETO, E NAO UM ATALHO
            Um global e' definido por NAO TER TERRITORIO: ele e' misturado na tela
            inteira, sem mascara. A passada propria existe para permitir mascarar a
            predicao por regiao — beneficio que o global, por definicao, nao usa.
            Ele pagava o custo de um centro sem consumir o que esse custo compra.

        O QUE MUDA NA MATEMATICA
            Antes:  comb = comb*(1-w) + w * eps_global      (mistura de PREDICOES)
            Depois: a correcao de baixo posto do global entra nos PESOS junto com a
            do centro, e a predicao ja' sai influenciada pelos dois.

            Nao e' a mesma operacao. Misturar predicoes e' linear na saida; compor
            adaptadores e' linear nos pesos e nao-linear na saida. O resultado visual
            muda, e por isso isto e' OPCAO e nao troca silenciosa — o peso pede
            calibracao propria.
        """
        if not self.globs:
            return 0
        w = self.adapter_scale if peso is None else peso
        movidos = [(n, w) for n in self.globs]
        self.caronas = (self.caronas or []) + movidos
        self.globs = []
        print(f"[gsmde] {len(movidos)} global(is) viraram carona a peso {w}: "
              f"{[n for n, _ in movidos]} — {len(movidos)} passada(s) por passo a menos",
              flush=True)
        return len(movidos)

    def _perf_add(self, chave, dt):
        """Acumulador de tempo por componente. time.time() puro, SEM synchronize:
        sincronizar p/ medir mudaria o que se mede (o proprio yield_gpu ja sincroniza
        e e' um dos suspeitos). Mede o tempo em que a CPU fica presa em cada parte."""
        d = getattr(self, "_perf", None)
        if d is None:
            d = self._perf = {}
        d[chave] = d.get(chave, 0.0) + dt

    def _pred_centro(self, cid, li, t, add_time, regiao_sa=None):
        """Predicao de ruido JA orientada (CFG interna) de UM centro sobre o latente
        li. regiao_sa: mascara de self-attn (None = self-attn global). Extraido do
        guided() do denoise para a colagem poder chamar com latentes independentes."""
        nome = base_do_centro(cid)
        cidl = parse_regiao(cid)[0]
        pe, npe, pp, npp = self.fatias.get(cidl, self.fatias.get(
            cid, (self.pe, self.npe, self.pp, self.npp)))
        _t = time.time()
        self._set_regiao_selfattn(regiao_sa)
        self._perf_add("regiao selfattn", time.time() - _t)
        if self.page_adapters:
            _t = time.time()
            self._page(nome, self.dev)
            self._perf_add("paginar->VRAM", time.time() - _t)
        _t = time.time()
        self.pipe.unet.set_adapters([nome], [self.adapter_scale])
        self._perf_add("set_adapters", time.time() - _t)
        _t = time.time()
        with torch.no_grad():
            o = self.pipe.unet(torch.cat([li, li]), t,
                               encoder_hidden_states=torch.cat([npe, pe]),
                               added_cond_kwargs=self._add_cond(npp, pp, add_time)).sample
        self._perf_add("unet centro (enfileira)", time.time() - _t)
        _t = time.time()
        yield_gpu(self.yield_ms)
        self._perf_add("yield_gpu (sync+sleep)", time.time() - _t)
        u, c = o.chunk(2)
        if self.page_adapters:
            _t = time.time()
            self._despeja_por_pressao(nome)
            self._perf_add("despejo", time.time() - _t)
        _t = time.time()
        self._set_regiao_selfattn(None)
        self._perf_add("regiao selfattn", time.time() - _t)
        return u + self.cfg * (c - u)

    @staticmethod
    def _resolve_te():
        """(device, dtype) do text encoder. Ele NAO roda no dispositivo de computacao
        por projeto — o UNet fica com a placa inteira. Onde ele roda e' escolha:

          GSMDE_TE_DEVICE = cpu (padrao) | dml | cuda
          GSMDE_TE_DTYPE  = bf16 (padrao) | fp16 | fp32

        PADRAO fp16, POR MEDICAO (bench_te.py, 2026-07-26, 6 centros):

            cuda bf16 sequencial   0.92s      cpu fp16 residente    5.33s
            cuda fp16 sequencial   0.93s      cpu bf16 residente    8.10s
            cuda fp32 sequencial   1.96s      cpu fp32 residente  212.19s

        Tres coisas que so' a medicao mostrou:

        1. fp32 na CPU e' catastrofico (212s) — o build ROCm vem sem MKL e cai
           num caminho ingenuo. Isso ja se sabia.
        2. Mas o padrao era bf16, e fp16 e' 1,5x MAIS RAPIDO na CPU (5.33 vs
           8.10). O comentario antigo afirmava que o oneDNN favorecia bf16;
           nao e' o que acontece aqui.
        3. E fp16 ainda e' MAIS PRECISO: erro relativo 4e-03 contra 4e-02 do
           bf16 (o bf16 tem 7 bits de mantissa, o fp16 tem 10). Domina nos dois
           eixos, entao virou o padrao.

        Na GPU as tres precisoes custam ~1s: ali NAO ha motivo p/ baixar
        precisao, o ganho e' de decimos de segundo numa fase que roda uma vez.

        Em maquina com iGPU (notebook), 'dml'+fp16 usa a placa integrada — que
        estaria ociosa — e libera a CPU, que la tem so 4 nucleos.
        """
        alvo = os.environ.get("GSMDE_TE_DEVICE", "cpu").strip().lower()
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}.get(
                  os.environ.get("GSMDE_TE_DTYPE", "").strip().lower())
        if alvo == "dml":
            try:
                import torch_directml
                d = torch_directml.device()
                print(f"[gsmde] TE em DirectML ({torch_directml.device_name(0)})", flush=True)
                return d, dt or torch.float16      # DirectML nao tem bf16
            except Exception as e:
                print(f"[gsmde] DirectML indisponivel ({e}); TE na CPU", flush=True)
        elif alvo == "cuda":
            try:
                if torch.cuda.is_available():
                    return torch.device("cuda"), dt or torch.float16
            except Exception:
                pass
        return torch.device("cpu"), dt or torch.float16

    def _revisa_residencia(self, w, h):
        """Reavalia quem fica na placa AGORA, com a resolucao real em maos.

        A politica original rodava so' no __init__, quando o tamanho da imagem
        ainda nao existe — e reservava um valor fixo. So' que as ativacoes
        crescem com a AREA, e o hires muda a area no meio do caminho: uma
        residencia aprovada a 1280x720 pode nao valer a 1920x1080.

        Aqui a area vira parte da decisao, e a revisao acontece a cada mudanca
        de tamanho (inclusive entre as camadas do hires).
        """
        area = int(w) * int(h)
        if getattr(self, "_area_alvo", None) == area:
            return                                  # nada mudou
        self._area_alvo = area
        if not getattr(self, "paged", None):
            return
        try:
            total = torch.cuda.mem_get_info()[1] / 2**30
        except Exception:
            return

        # PREVISAO: cabe? A referencia sai do medido nesta maquina — a 768x768
        # o pico ficou perto de 5,5 GB com o backbone residente; escalando pela
        # area da uma estimativa grosseira mas util, porque o erro que importa
        # (passar de 8 GB) e' grande, nao sutil.
        # A referencia foi medida com o BACKBONE RESIDENTE. Com o accelerate
        # transmitindo o UNet por blocos ele deixa de ocupar os ~4,8 GB e a previsao
        # antiga vira alarme falso — 1024 com offload sequential foi medido rodando
        # a 25,1 s/passo, sem despejo. Desconta o backbone quando ele nao mora la'.
        PICO_REF_GB, AREA_REF, BACKBONE_GB = 5.5, 768 * 768, 4.8
        com_offload = getattr(self, "offload_base", "none") not in (None, "", "none")
        base_ref = PICO_REF_GB - (BACKBONE_GB if com_offload else 0.0)
        pico = base_ref * max(1.0, area / AREA_REF)
        if pico > total * 0.95:
            print(f"[gsmde] AVISO: {w}x{h} deve pedir ~{pico:.1f} GB de pico e a "
                  f"placa tem {total:.1f} GB. Nesta build do ROCm faltar VRAM nao "
                  f"da OOM limpo — da 'unspecified launch failure' e trava a placa. "
                  f"Considere gerar menor e subir no hires.", flush=True)

        # Com o accelerate no comando, a residencia dos nichos ja' foi decidida por
        # `aplica_offload_base`, que desligou a paginacao DE PROPOSITO (ver o
        # comentario dos "dois gerentes" logo acima dele). Re-rodar a politica aqui
        # religa o segundo gerente toda vez que a area muda — inclusive entre as
        # camadas do hires. A area nova ja' ficou registrada; e' so' nao repaginar.
        if com_offload:
            return
        try:
            self.page_adapters = self._politica_vram(
                "auto", getattr(self, "_vram_reserva", 0.0))
        except Exception as e:
            print(f"[gsmde] revisao de residencia falhou ({e}); mantendo a anterior",
                  flush=True)

    def _politica_vram(self, modo=True, reserva_gb=0.0):
        """Quais nichos ficam RESIDENTES na placa. Devolve se ainda ha paginacao.

        Paginar e' um seguro contra placa pequena, nao uma virtude: cada nicho sobe e
        desce o PCIe UMA VEZ POR PASSO. Em 26 passos com 5 centros sao 260 idas e
        voltas — numa placa com folga isso e' tempo de compute jogado fora esperando
        barramento. Entao a decisao vem do sensor, nao de constante:

            modo="ram"   -> tudo paginado (comportamento historico, placa apertada)
            modo="vram"  -> nada paginado (placa grande; assume que cabe)
            modo="auto"  -> mede a folga e enche ate o orcamento

        A reserva protege as ATIVACOES, que crescem com a resolucao e sao o que
        realmente estoura em 8GB. Sem reserva, encher a VRAM de nicho faria o hires
        morrer — trocaria um gargalo por um crash.
        """
        self.residentes = set()
        self.ordem_uso = []                       # LRU: o mais antigo sai primeiro
        if modo in (False, "vram", "residente"):
            self.residentes = set(self.paged)
            for nome in self.paged:               # garante que estao MESMO na placa
                if not self._bloco_para(nome, self.dev):
                    for m in self.paged[nome]:
                        m.to(self.dev, non_blocking=True)
            self.teto_alocado = float("inf")      # nunca despeja
            print("[gsmde] paginacao OFF: todos os nichos residentes (pedido)", flush=True)
            return False
        if modo in (True, "ram", "paginado"):
            for nome in self.paged:
                if not self._bloco_para(nome, "cpu"):
                    for m in self.paged[nome]:
                        m.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()
            self.teto_alocado = 0.0               # sempre despeja
            print("[gsmde] paginacao RAM: todos os nichos paginados (pedido)", flush=True)
            return True
        # --- auto ---
        try:
            livre, total = torch.cuda.mem_get_info()
            livre, total = livre / 2**30, total / 2**30
        except Exception:
            return True                       # sem sensor, o seguro e' paginar
        # RESERVA PROPORCIONAL A AREA.
        #
        # Era constante (2,5 GB), calibrada a 768x768. As ativacoes crescem com a
        # AREA da imagem, entao a 1280x720 (1,56x a area) a premissa ficava curta
        # e a politica aprovava uma configuracao que nao cabia. Medido: a
        # dedicada batia 7,73 de 8,00 GB, o WDDM comecava a empurrar para a
        # compartilhada (0,2 -> 9,3 GB em 24s) e o driver despejava em massa.
        #
        # E o modo de falha nao ajudava a diagnosticar: neste build do ROCm
        # esgotar a VRAM NAO da `hipErrorOutOfMemory`, da `unspecified launch
        # failure` — o mesmo sintoma de um kernel quebrado. Isso mandou a
        # investigacao para o AOTRITON e para a paginacao antes de chegar aqui.
        # A BASE ERA UM CHUTE DE PIOR CASO, E COMIA O ORCAMENTO INTEIRO.
        #
        # `max(2.5, 0.30*total)` da 2,5 GB numa placa de 8. So que as ativacoes
        # foram MEDIDAS (2026-07-28): ~0,49 GB a 589.824 px, com o pico real da
        # geracao subindo pouco alem disso. Reservar 2,5 para gastar 0,5 jogava
        # fora 2 GB de folga.
        #
        # O efeito era invisivel ate o backbone encolher: com o teacher nao sobrava
        # nada mesmo, entao a constante nunca era o gargalo. Com o backbone podado
        # havia 2,51 GB livres e a politica ainda decidiu "0 de 10 nichos
        # residentes" — porque 2,51 - 2,50 = 0,01. Nao faltava memoria; faltava a
        # politica enxergar a memoria que existia.
        #
        # Agora a base sai da MEDIDA com margem de seguranca, nao do medo. O piso
        # existe para o caso de a medicao nao valer (build diferente, resolucao
        # muito pequena): abaixo dele nao se economiza nada util mesmo.
        AREA_REF = 768 * 768
        ATIV_REF_GB = 0.49                    # medido nesta maquina, a AREA_REF
        MARGEM = 1.6                          # 60% sobre o medido
        PISO_GB = 0.8
        area = getattr(self, "_area_alvo", AREA_REF) or AREA_REF
        escala = max(1.0, area / AREA_REF)
        if reserva_gb:                        # pedido explicito ganha da medida
            reserva = float(reserva_gb) * escala
        else:
            reserva = max(PISO_GB, ATIV_REF_GB * MARGEM * escala)
        # TETO DE OCUPACAO (escolha do usuario: 95%). No Windows o WDDM derrama em
        # vez de travar, entao a folga pode ser menor que num Linux — mas alguma
        # folga tem de sobrar, senao cada pouso de centro despeja algo.
        teto = 0.95 * total
        reserva = max(reserva, total - teto)

        def _bytes(o):
            # self.paged guarda os MODULOS lora_A/lora_B (nn.Linear), nao tensores:
            # quem tem numel() e' o parametro dentro deles.
            if hasattr(o, "parameters"):
                return sum(p.numel() * p.element_size() for p in o.parameters())
            return o.numel() * o.element_size()

        # nicho menor primeiro: com 89MB a 709MB, cabem mais nichos por GB gasto
        tam = {n: sum(_bytes(t) for t in ts) / 2**30 for n, ts in self.paged.items()}
        # MEDIR NO ESTADO LIMPO. Com os nichos ainda na placa o 'livre' do driver e'
        # fantasia: em 8GB a carga ja transbordou p/ a memoria COMPARTILHADA (medido:
        # 5,8GB de spill no worker) e mem_get_info reporta livre=0.0 com a placa longe
        # de cheia. Pior: a placa nao e' so nossa — dwm.exe e o WebView2 tambem moram
        # nela. Entao primeiro devolve TUDO p/ a RAM, limpa o cache do alocador, e so
        # ai pergunta quanto sobra de verdade. Ai o numero ja desconta os outros
        # processos e nao esta contaminado por spill nosso.
        for nome in self.paged:
            if not self._bloco_para(nome, "cpu"):
                for m in self.paged[nome]:
                    m.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        livre = torch.cuda.mem_get_info()[0] / 2**30
        orcamento = livre - reserva
        for nome in sorted(tam, key=tam.get):
            if tam[nome] <= orcamento:
                self.residentes.add(nome)
                orcamento -= tam[nome]
        for nome in self.residentes:              # traz de volta so quem coube
            if not self._bloco_para(nome, self.dev):
                for m in self.paged[nome]:
                    m.to(self.dev, non_blocking=True)
        # TETO da NOSSA alocacao, p/ o despejo por pressao decidir durante o laco.
        # Medido agora, uma vez: 'livre' aqui ja desconta os outros processos (o
        # desktop custa ~1,56GB nesta maquina). Dentro do laco nao da p/ reconsultar
        # mem_get_info: quando um nicho volta p/ a RAM a memoria vai p/ o pool do
        # torch, nao p/ o driver — o 'livre' nao subiria e despejariamos tudo.
        self.teto_alocado = (torch.cuda.memory_allocated() / 2**30) + livre - reserva
        fica = sum(tam[n] for n in self.residentes)
        print(f"[gsmde] paginacao AUTO: placa {total:.1f}GB, livre (apos limpar) "
              f"{livre:.2f}GB, reserva {reserva:.1f}GB p/ ativacoes "
              f"(area {int(getattr(self, '_area_alvo', 768*768))}px) -> "
              f"{len(self.residentes)}/{len(self.paged)} nichos residentes "
              f"({fica:.2f}GB), {len(self.paged) - len(self.residentes)} paginados",
              flush=True)
        return len(self.residentes) < len(self.paged)

    # unidade indivisivel do device_map (mesma lista do SD.Next): partir abaixo de
    # um Linear/Conv nao faz sentido — o hook move o modulo inteiro ou nada.
    NO_SPLIT = ["Linear", "Conv1d", "Conv2d", "Conv3d",
                "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]

    def aplica_offload_base(self, modo="none", teto_gb=0.0):
        """Offload do UNet BASE (~4,8GB) — o piso de VRAM do GSMDE.

        Ate aqui o motor so paginava os NICHOS (2,8GB); o backbone nascia residente e
        nunca saia da placa. Numa placa de 8GB isso fixa um piso que faz o resto
        estourar: medido 7,6GB com 4 centros residentes -> spill -> 53,9s/passo,
        contra 15,0s/passo com os nichos paginados. O SD.Next nao tem esse piso
        porque parte o proprio UNet (marca d'agua 0,6 = 4,8GB de 8) e mantem o resto
        na RAM (ele usa 14-23GB de RAM contra 7,7GB nossos, com 40GB disponiveis).

        Modos (os mesmos nomes do SD.Next):
          none        UNet inteiro residente (comportamento historico)
          sequential  cada SUBMODULO sobe na hora do forward e desce depois. Esvazia
                      mais a VRAM e e' o que mais depende do PCIe. No SD.Next
                      costuma quebrar o text encoder (1 device por vez); aqui nao
                      ha esse risco: o TE do GSMDE ja roda na CPU por projeto.
          model       o UNet inteiro sobe/desce por chamada (granularidade grossa)
          group       blocos de N submodulos, com stream de prefetch se disponivel
          balanced    device_map com teto: parte o UNet entre GPU e CPU
        """
        import accelerate
        unet = self.pipe.unet
        try:                                   # estado limpo antes de trocar de modo
            accelerate.hooks.remove_hook_from_module(unet, recurse=True)
        except Exception:
            pass
        if modo in (None, "", "none"):
            unet.to(self.dev)
            self.offload_base = "none"
            return "none"
        try:
            if modo == "sequential":
                unet.to("cpu")
                accelerate.cpu_offload(unet, execution_device=self.dev)
            elif modo == "model":
                unet.to("cpu")
                accelerate.cpu_offload_with_hook(unet, execution_device=self.dev)
            elif modo == "group":
                if not hasattr(unet, "enable_group_offload"):
                    raise RuntimeError("diffusers sem enable_group_offload")
                unet.to("cpu")
                unet.enable_group_offload(
                    onload_device=self.dev, offload_device=torch.device("cpu"),
                    offload_type="block_level", num_blocks_per_group=1,
                    non_blocking=True, use_stream=True)
            elif modo == "balanced":
                total = torch.cuda.mem_get_info()[1] / 2**30
                teto = float(teto_gb) or (0.6 * total)     # 0.6 = default do SD.Next
                unet.to("cpu")
                mapa = accelerate.infer_auto_device_map(
                    unet, max_memory={0: int(teto * 2**30), "cpu": int(24 * 2**30)},
                    no_split_module_classes=self.NO_SPLIT)
                accelerate.dispatch_model(unet, device_map=mapa,
                                          main_device=self.dev, force_hooks=True)
                na_gpu = sum(1 for v in mapa.values() if v not in ("cpu", "disk"))
                print(f"[gsmde] offload balanced: teto {teto:.2f}GB | "
                      f"{na_gpu}/{len(mapa)} submodulos na GPU", flush=True)
            else:
                raise RuntimeError(f"modo desconhecido: {modo}")
        except Exception as e:
            print(f"[gsmde] offload '{modo}' falhou ({type(e).__name__}: {e}); "
                  f"voltando p/ UNet residente", flush=True)
            try:
                accelerate.hooks.remove_hook_from_module(unet, recurse=True)
            except Exception:
                pass
            unet.to(self.dev)
            self.offload_base = "none"
            return "none"
        # com o accelerate gerindo o UNet, a paginacao de nicho seria um SEGUNDO
        # gerente mexendo nos mesmos tensores — desliga p/ nao brigarem.
        self.page_adapters = False
        self.residentes = set(self.paged)
        self.offload_base = modo
        livre = torch.cuda.mem_get_info()[0] / 2**30
        print(f"[gsmde] offload base = {modo} | livre agora {livre:.2f}GB "
              f"(paginacao de nicho desligada: quem manda e' o accelerate)", flush=True)
        return modo

    def reaplica_politica(self, modo=True, reserva_gb=0.0):
        """Redecide a residencia SEM recarregar o motor.

        Trocar 'paginacao' na UI custava um reload de ~7GB (1,5min) porque a opcao
        entrava na chave do cache do worker. Mas mudar de politica e' so MOVER
        tensores entre RAM e VRAM — o modelo e' o mesmo. Tambem e' o gancho p/
        re-decidir a cada rodada depois que o fim do job devolveu os nichos p/ a RAM.
        """
        try:
            self.page_adapters = self._politica_vram(modo, reserva_gb)
        except Exception as e:
            print(f"[gsmde] politica de paginacao falhou ({e}); paginando tudo", flush=True)
            self.residentes = set()
            self.page_adapters = True
        return self.page_adapters

    def _monta_blocos(self):
        """Um BUFFER CONTIGUO por centro, em vez de um .to() por modulo.

        O PORQUE, medido: com 7 centros o motor reportava "7840 modulos", e a
        paginacao fazia um .to() em CADA um, a cada passo — ~196 mil
        transferencias minusculas numa geracao de 25 passos. Nao e' so' lento:
        essa enxurrada de alocacoes pequenas fragmenta o heap do driver, e sob
        WDDM o commit escorre para a memoria COMPARTILHADA (medido: 9,5 GB de
        shared com apenas 4,87 GB alocados pelo torch). Dai vinham os tres
        sintomas juntos — 86% do tempo preso em yield_gpu esperando a fila, o
        spill, e por fim `hipErrorLaunchFailure` travando a placa.

        Aqui os pesos de um centro viram UM tensor plano. Paginar passa a ser
        uma copia so' por centro (7 em vez de 7840), e os parametros viram
        VIEWS dentro do bloco — o forward continua lendo os mesmos tensores,
        sem copia extra.
        """
        self.blocos = {}
        for nome, mods in self.paged.items():
            params = [q for m in mods for q in m.parameters(recurse=False)]
            if not params:
                continue
            dts = {q.dtype for q in params}
            if len(dts) != 1:
                # dtypes misturados nao cabem num bloco unico; esse centro fica
                # no caminho antigo em vez de quebrar.
                print(f"[gsmde] centro '{nome}': dtypes {dts} — sem bloco contiguo",
                      flush=True)
                continue
            total = sum(q.numel() for q in params)
            # SEM pin_memory. Medido nesta maquina: alocar 4,24 GB pinados
            # levou a memoria COMPARTILHADA da GPU de 0,82 para 8,83 GB — o
            # Windows conta memoria page-locked como shared GPU memory, e nao
            # devolve nem depois do free. Era a causa do spill saltar de ~2 GB
            # para ~11 GB depois que os blocos contiguos entraram.
            #
            # Pinar so' serve p/ o non_blocking=True ser assincrono de verdade.
            # Aqui a copia e' UMA por centro por passo (nao 1120), entao o ganho
            # da assincronia e' pequeno e nao paga 8 GB de shared.
            flat = torch.empty(total, dtype=params[0].dtype, device="cpu")
            meta, off = [], 0
            for q in params:
                n = q.numel()
                flat[off:off + n].copy_(q.data.detach().reshape(-1))
                meta.append((q, off, n, tuple(q.shape)))
                off += n
            for q, o, n, shp in meta:       # aponta p/ o bloco na RAM
                q.data = flat[o:o + n].view(shp)
            self.blocos[nome] = {"cpu": flat, "meta": meta, "gpu": None}
        if self.blocos:
            n_mod = sum(len(v["meta"]) for v in self.blocos.values())
            gb = sum(v["cpu"].numel() * v["cpu"].element_size()
                     for v in self.blocos.values()) / 2**30
            print(f"[gsmde] blocos contiguos: {len(self.blocos)} centros, "
                  f"{gb:.2f} GB | 1 transferencia por centro em vez de {n_mod}",
                  flush=True)

    def _bloco_para(self, nome, dev):
        """Move o bloco inteiro e reaponta as views. True se tratou aqui."""
        b = getattr(self, "blocos", {}).get(nome)
        if b is None:
            return False
        if dev == "cpu":
            if b["gpu"] is not None:
                for q, o, n, shp in b["meta"]:
                    q.data = b["cpu"][o:o + n].view(shp)
                b["gpu"] = None               # libera a copia da placa
        else:
            if b["gpu"] is None:
                # Aloca a cada subida DE PROPOSITO: guardar o buffer da placa
                # manteria os 7 centros residentes (~3,5 GB) e anularia a
                # paginacao. O alocador do torch ja recicla blocos do mesmo
                # tamanho, entao nao ha churn real no driver.
                g = b["cpu"].to(dev, non_blocking=True)
                b["gpu"] = g
                for q, o, n, shp in b["meta"]:
                    q.data = g[o:o + n].view(shp)
        return True

    def _page(self, nome, dev, forcar=False):
        if not forcar and nome in getattr(self, "residentes", ()):
            return                            # mora na placa: nao paga PCIe
        with fase(f"paginar nicho -> {'VRAM' if dev != 'cpu' else 'RAM'}"):
            if not self._bloco_para(nome, dev):     # bloco contiguo quando ha
                for m in self.paged.get(nome, []):  # senao, caminho antigo
                    m.to(dev, non_blocking=True)
        if dev != "cpu":                      # LRU: usado agora = o mais recente
            ordem = getattr(self, "ordem_uso", None)
            if ordem is not None:
                if nome in ordem:
                    ordem.remove(nome)
                ordem.append(nome)

    def _despeja_por_pressao(self, nome):
        """Estrategia do SD.Next (marca d'agua baixa): o nicho SO volta p/ a RAM se
        a memoria apertou. Antes descia sempre, e em placa com folga isso era
        0,99s/passo de PCIe pago a toa (medido: 1024px, 1 centro, 12 passos —
        49,20s paginando contra 37,33s residente).

        Vantagem sobre decidir na carga: aqui a residencia se ajusta sozinha ao que
        a rodada REALMENTE gasta. Se o hires apertar, os nichos saem; se sobrar
        espaco, eles ficam — sem precisar acertar uma reserva no chute.
        """
        teto = getattr(self, "teto_alocado", 0.0)
        if teto == float("inf"):
            return                                       # modo 'manter na placa'
        alocado = torch.cuda.memory_allocated() / 2**30
        if alocado <= teto:
            if nome not in self.residentes:              # cabe: promove a residente
                self.residentes.add(nome)
                if getattr(self, "_diag_promo", True):
                    print(f"[gsmde] nicho '{nome}' fica na placa "
                          f"(alocado {alocado:.2f}G <= teto {teto:.2f}G)", flush=True)
            return
        # apertou: devolve este e, se ainda apertado, os residentes mais ANTIGOS
        self.residentes.discard(nome)
        self._page(nome, "cpu", forcar=True)
        for outro in list(getattr(self, "ordem_uso", [])):
            if torch.cuda.memory_allocated() / 2**30 <= teto:
                break
            if outro in self.residentes:
                self.residentes.discard(outro)
                self._page(outro, "cpu", forcar=True)
                print(f"[gsmde] pressao: devolvendo '{outro}' p/ a RAM", flush=True)

    def vram(self, tag=""):
        if not torch.cuda.is_available():
            return 0.0
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        print(f"[gsmde] VRAM {tag}: alocada {a:.2f} GB | reservada {r:.2f} GB", flush=True)
        return a

    # ---- texto ----
    def _center_context_prompt(self, full, words, weight=None):
        """Prompt por-centro CONTEXTO+FOCO (ideia do usuario, versao BRANDA): o prompt
        INTEIRO preserva o contexto/cena; as palavras-foco do centro entram no MESMO
        chunk com um peso leve (center_focus_weight), reforcando o foco daquele centro
        SEM dobrar. A duplicata em chunk proprio (BREAK) sobre-pesa e frita — ainda mais
        combinada com o adapter especializado — entao usamos enfase in-context.
        weight=1.0 (ou sem words) -> so contexto (== ler o prompt inteiro)."""
        weight = self.center_focus_weight if weight is None else weight
        if not words or abs(weight - 1.0) < 1e-3:
            return full
        foco = ", ".join(words)
        return f"({foco}:{weight:.2f}), " + full

    def set_controlnet(self, cond_pil, path=r"D:\Models\controlnet\xl", scale=1.0):
        """Liga o OpenPose ControlNet SDXL: cond_pil = imagem do esqueleto (formato
        OpenPose). O forward do ControlNet e' injetado no loop base-puro do denoise
        (residuais down/mid somados ao UNet) -> a personagem e' RE-DESENHADA na pose
        alvo (cria informacao nova). cond_pil=None desliga."""
        import numpy as np
        if cond_pil is None:
            self.cn_cond = None
            return
        if getattr(self, "controlnet", None) is None:
            from diffusers import ControlNetModel
            with fase("load: controlnet openpose"):
                self.controlnet = ControlNetModel.from_pretrained(
                    path, torch_dtype=torch.float16).to(self.dev)
        a = torch.from_numpy(np.asarray(cond_pil.convert("RGB"), np.float32) / 255)
        self.cn_cond = a.permute(2, 0, 1).unsqueeze(0).to(self.dev, torch.float16)
        self.cn_scale = float(scale)

    def _embrulha_selfattn(self):
        """(Re)instala o RegionalSelfAttn nos attn1. Precisa ser chamavel de novo
        porque load_ip_adapter faz set_attn_processor(AttnProcessor2_0()) e apaga
        TODOS os processadores — levando junto o isolamento de self-attn por
        territorio. Como o IP-Adapter instala os processadores dele nos attn2
        (cross-attn, onde o image embed entra), os attn1 ficam livres p/ voltarem.
        Sem isto, ligar contexto global desligaria os territorios em silencio."""
        self.selfattn = []
        for name, mod in self.pipe.unet.named_modules():
            if name.endswith("attn1") and hasattr(mod, "processor"):
                if isinstance(mod.processor, RegionalSelfAttn):
                    self.selfattn.append(mod.processor)
                    continue
                wrap = RegionalSelfAttn(mod.processor)
                mod.processor = wrap
                self.selfattn.append(wrap)
        return len(self.selfattn)

    def set_ip_adapter(self, ref_pil, scale=0.6, path=r"D:\Models\ipadapter",
                       subfolder="sdxl_models", weight="ip-adapter_sdxl.safetensors"):
        """Trava IDENTIDADE/aparencia a partir de uma imagem de referencia (o frame 0
        do video). Injeta os image_embeds na cross-attn (attn2) de TODO frame -> cada
        frame gera FRESCO da pose + identidade fixa, sem generation loss do i2i encadeado.
        ref_pil=None desliga."""
        if ref_pil is None:
            self.ip_embeds = None
            if getattr(self, "_ip_loaded", False):
                try:
                    self.pipe.set_ip_adapter_scale(0.0)
                except Exception:
                    pass
            return
        if not getattr(self, "_ip_loaded", False):
            with fase("load: ip-adapter"):
                # o RegionalSelfAttn do engine quebra a conversao do load_ip_adapter;
                # reseta pro processor padrao (a self-attn regional nao e' usada no
                # video base-puro) antes de carregar o IP-Adapter.
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                self.pipe.load_ip_adapter(path, subfolder=subfolder, weight_name=weight)
                # os pesos novos (IP attn + image encoder) carregam na CPU -> move p/ GPU fp16
                self.pipe.unet.to(self.dev, torch.float16)
                if getattr(self.pipe, "image_encoder", None) is not None:
                    self.pipe.image_encoder.to(self.dev, torch.float16)
                # devolve o isolamento por territorio que o set_attn_processor apagou
                n = self._embrulha_selfattn()
                print(f"[gsmde] ip-adapter carregado | self-attn regional "
                      f"reinstalada em {n} blocos", flush=True)
            self._ip_loaded = True
        self.pipe.set_ip_adapter_scale(float(scale))
        with torch.no_grad():
            emb = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=ref_pil, ip_adapter_image_embeds=None,
                device=self.dev, num_images_per_prompt=1, do_classifier_free_guidance=True)
        self.ip_embeds = [e.to(self.dev, torch.float16) for e in emb]

    def _cond(self, text_embeds, time_ids):
        """added_cond_kwargs com image_embeds NO TAMANHO DE LOTE CERTO.

        Com o IP-Adapter carregado o UNet passa a ter encoder_hid_dim_type=
        'ip_image_proj' e EXIGE image_embeds em TODO forward — inclusive na sonda
        de mascaras, que roda com lote 1 enquanto os embeds foram preparados com
        CFG (lote 2). Montar o dicionario a mao em 8 lugares diferentes garantia
        que algum deles esqueceria; agora todos passam por aqui.
        """
        d = {"text_embeds": text_embeds, "time_ids": time_ids}
        ip = getattr(self, "ip_embeds", None)
        if ip is not None:
            b = text_embeds.shape[0]
            ajust = []
            for e in ip:
                if e.shape[0] == b:
                    ajust.append(e)
                elif e.shape[0] > b:
                    ajust.append(e[-b:])                  # metade condicional
                else:
                    reps = [b // max(1, e.shape[0])] + [1] * (e.dim() - 1)
                    ajust.append(e.repeat(*reps))
            d["image_embeds"] = ajust
        return d

    def _add_cond(self, npp, pp, add_time):
        """Atalho p/ os forwards com CFG em lote 2 (uncond+cond)."""
        return self._cond(torch.cat([npp, pp]), torch.cat([add_time, add_time]))

    def _encode(self, prompt, neg):
        # enfase/wildcard/BREAK (A1111/SD.Next) via prompt_weighting; fallback ao
        # encode_prompt padrao se o modulo faltar ou o prompt nao tiver sintaxe.
        tem_sintaxe = re.search(r"[()\[\]{}]|\bBREAK\b|\n",
                                (prompt or "") + (neg or "")) is not None
        pesos = getattr(self, "weighted_prompt", True) and tem_sintaxe
        # CHUNKING: prompt acima de 75 tokens TAMBEM tem que passar por aqui, com ou
        # sem sintaxe de peso. O encode_prompt do diffusers trunca em 77 em silencio
        # — e como os centros ja chunkavam (get_center_weighted_embeddings), o
        # BACKBONE ficava lendo menos prompt que os especialistas.
        longo = False
        try:
            import prompt_weighting as _W
            longo = (_W.conta_tokens(self.pipe.tokenizer, prompt) > 75
                     or _W.conta_tokens(self.pipe.tokenizer, neg) > 75)
        except Exception:
            pass
        use_w = pesos or longo
        # sem pesos ligados, os parenteses valem como TEXTO (nao vira enfase surpresa)
        self._literal = not pesos
        if use_w:
            try:
                import random as _rnd
                import prompt_weighting as _W
                rng = _rnd.Random(getattr(self, "_pw_seed", 0))
                pe, npe, pp, npp = _W.get_weighted_sdxl_embeddings(
                    self.pipe, prompt, neg, rng, device=self.te_device,
                    literal=self._literal)
                if longo:
                    print(f"[gsmde] prompt longo: {_W.conta_tokens(self.pipe.tokenizer, prompt)}"
                          f" tokens -> {pe.shape[1] // 77} blocos de 75 (chunking, nada truncado)",
                          flush=True)
            except Exception as e:
                print(f"[gsmde] prompt_weighting falhou ({e}); usando encode padrao", flush=True)
                use_w = False
        if not use_w:
            with torch.no_grad():
                pe, npe, pp, npp = self.pipe.encode_prompt(
                    prompt=prompt, prompt_2=prompt, negative_prompt=neg, negative_prompt_2=neg,
                    device=self.te_device, num_images_per_prompt=1,
                    do_classifier_free_guidance=True)
        d = self.dev
        return (pe.to(d, torch.float16), npe.to(d, torch.float16),
                pp.to(d, torch.float16), npp.to(d, torch.float16))

    def set_prompt(self, prompt, neg=NEG, fatias=None):
        """fatias: {id_do_centro: prompt proprio}. Sem isso, todos leem o prompt inteiro.

        POR QUE FATIAR
        1) Ataca o attribute bleeding na ORIGEM. Com prompt unico, o centro do
           vampiro le a palavra 'werewolf' e a atencao dele ja mistura os dois — a
           mascara so limpa o estrago no output. Fatiado, o vampiro nunca ve o
           lobisomem: nao ha o que vazar.
        2) O teto de 77 tokens do CLIP deixa de ser GLOBAL e vira POR CENTRO. Seis
           centros = ~462 tokens uteis, sem chunking e sem truncar, cada um gastando
           o seu so no que e' dele.

        A mascara continua vindo do prompt COMPLETO (sonda no base): e' ele que diz
        ONDE cada coisa esta na tela. A fatia diz O QUE cada centro pinta. Papeis
        diferentes — misturar os dois quebraria a localizacao.
        """
        # resolve wildcards {A|B} UMA vez (semeado) -> encode e tokenizacao dos centros
        # veem o mesmo texto; guarda a versao limpa (sem pesos) p/ mapear center_tokens.
        try:
            import random as _rnd
            import prompt_weighting as _W
            _rng = _rnd.Random(getattr(self, "_pw_seed", 0))
            prompt = _W.resolve_wildcards(prompt, _rng)
            neg = _W.resolve_wildcards(neg, _rng)
            self._clean_prompt = _W.clean_text(prompt)
        except Exception:
            self._clean_prompt = prompt
        self.prompt = prompt
        # regioes disjuntas por construcao — a peca que faltava. As instancias do
        # mesmo modelo se dividem a tela; assim o territorio nao depende da atencao
        # (que e' onde o bleeding vive).
        self.regioes = auto_regioes([cid for cid, _ in self.regional])
        if self.regioes:
            print(f"[gsmde] regioes: { {k: tuple(round(x,2) for x in v) for k,v in self.regioes.items()} }",
                  flush=True)
        with fase("prompt: encode (TE na CPU)"):
            self.pe, self.npe, self.pp, self.npp = self._encode(prompt, neg)
            # guarda o modo do encode do BASE: as fatias dos centros chamam _encode
            # depois e sobrescreveriam self._literal — e o mapa de tokens abaixo tem
            # que descrever a sequencia do base, nao a da ultima fatia.
            self._literal_base = getattr(self, "_literal", False)
            self.fatias = {}
            for cid, txt in (fatias or {}).items():
                if txt and txt.strip():
                    self.fatias[cid] = self._encode(txt.strip(), neg)
            # AUTO contexto+foco: sem fatias explicitas, cada centro le o prompt inteiro
            # com o GLOBAL abaixado (context_w) e o FOCO erguido (focus_w). Backbone=1.0.
            if not self.fatias and self.regional and getattr(self, "auto_center_focus", True):
                import random as _rnd
                import prompt_weighting as _W
                d = self.dev
                # UMA subida do TE para TODOS os centros: em sequencial nao
                # faz sentido subir e descer 15 vezes — o custo do transito
                # ficaria maior que o do encode.
                alvo = getattr(self, "te_device_calc", self.te_device)
                with te_na_placa(self.pipe, alvo) as dev_seq:
                    onde = dev_seq or self.te_device
                    for cid, words in self.regional:
                        if not words:
                            continue
                        rng = _rnd.Random(getattr(self, "_pw_seed", 0))
                        e = _W.get_center_weighted_embeddings(
                            self.pipe, prompt, neg, words,
                            focus_w=self.center_focus_weight,
                            context_w=self.center_context_weight,
                            rng=rng, device=onde)
                        self.fatias[cid] = tuple(x.to(d, torch.float16) for x in e)
                if self.fatias:
                    print(f"[gsmde] contexto+foco por centro: {len(self.fatias)} centros "
                          f"(foco x{self.center_focus_weight} / global x{self.center_context_weight}, "
                          f"backbone 1.0)", flush=True)
        if self.fatias:
            print(f"[gsmde] prompt fatiado: "
                  + " | ".join(f"{k}={len(v.split())}w" for k, v in (fatias or {}).items() if v)
                  + f"  (teto de tokens virou {len(self.fatias)}x77)", flush=True)
        tok = self.pipe.tokenizer_2
        # o indice tem que ser o da SEQUENCIA QUE O ENCODER VIU. Com chunking, cada
        # bloco gasta 2 posicoes (bos/eos) e o pad vai ate 77 — a tokenizacao plana
        # truncada em 77 apontaria as ancoras das mascaras p/ palavras erradas
        # (e perderia tudo depois do token 77).
        lit = getattr(self, "_literal_base", False)
        try:
            import prompt_weighting as _W
            ids = _W.ids_em_blocos(tok, prompt if lit else self._clean_prompt, literal=lit)
        except Exception:
            ids = tok(self._clean_prompt, truncation=True, max_length=77).input_ids
        toks = tok.convert_ids_to_tokens(ids)
        self.center_tokens = {}      # chave = id da instancia (person#1), nao o adapter
        for n, words in self.regional:
            want = {p for w in words for p in tok.tokenize(w)}
            self.center_tokens[n] = [i for i, t in enumerate(toks) if t in want]
        print(f"[gsmde] tokens/centro: { {n: len(v) for n, v in self.center_tokens.items()} }",
              flush=True)

    # ---- VAE ----
    # O VAE so trabalha no inicio e no fim, mas ocupava VRAM o denoise inteiro.
    def _vae(self, on):
        if self.vae_offload:
            with fase(f"vae -> {'VRAM' if on else 'RAM'}"):
                self.pipe.vae.to(self.dev if on else "cpu")
                if not on:
                    torch.cuda.empty_cache()

    def encode_image(self, img: Image.Image):
        import numpy as np
        self._vae(True)
        x = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)
        x = (x.float() / 127.5 - 1.0).unsqueeze(0).to(self.dev, torch.float16)
        with fase("vae: encode"), torch.no_grad():
            lat = self.pipe.vae.encode(x).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        self._vae(False)
        return lat.to(torch.float16)

    def decode(self, lat) -> Image.Image:
        import numpy as np
        self._vae(True)
        with fase("vae: decode"), torch.no_grad():
            img = self.pipe.vae.decode(lat / self.pipe.vae.config.scaling_factor).sample
        a = ((img.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        self._vae(False)
        return Image.fromarray(a)

    # ---- regioes explicitas (caixas disjuntas) ----
    def _box_masks(self, H, W):
        """Caixa (x0,y0,x1,y1) -> mascara suave [H,W], com borda difusa pra costurar
        sem emenda. Normaliza entre os centros COM caixa (particao da unidade);
        centros sem caixa (scenary) ficam de fora e usam a mascara de atencao/tela."""
        if not getattr(self, "regioes", None):
            return None
        feather = max(2, min(H, W) // 12)
        yy = torch.arange(H, device=self.dev).view(H, 1).float()
        xx = torch.arange(W, device=self.dev).view(1, W).float()
        raw = {}
        for cid, (x0, y0, x1, y1) in self.regioes.items():
            l, r = x0 * W, x1 * W
            t_, b = y0 * H, y1 * H
            fx = torch.sigmoid((xx - l) / feather) * torch.sigmoid((r - xx) / feather)
            fy = torch.sigmoid((yy - t_) / feather) * torch.sigmoid((b - yy) / feather)
            raw[cid] = (fx * fy)
        if not raw:
            return None
        # Normalizar por soma crua com clamp minusculo EXPLODE nas bordas difusas de
        # caixas afastadas (diagonal): onde a soma vale ~0.01, m/soma vira valores
        # gigantes -> ruido. So normaliza ONDE ha sobreposicao real (soma > 1); fora
        # disso mantem os valores crus (que estao em [0,1]) e o fundo pega o resto.
        soma = torch.stack(list(raw.values())).sum(0)
        escala = torch.where(soma > 1.0, 1.0 / soma, torch.ones_like(soma))
        return {cid: (m * escala).to(torch.float16) for cid, m in raw.items()}

    @staticmethod
    def _piramide(H, W, niveis=3):
        """n_pix -> (h,w) das resolucoes onde a cross-attn REALMENTE existe.

        Na SDXL o primeiro bloco do UNet e' DownBlock2D, SEM atencao: attn2 so
        aparece a partir de latente/2. Filtrar pela resolucao cheia (H*W) portanto
        nunca casava com bloco nenhum — a sonda ficava muda, 'raw' saia vazio e
        TODO centro caia na mascara uniforme 1/N. Ou seja: os territorios por
        cross-attention, que sao o ponto do projeto, nunca chegaram a existir.

        O downsample da UNet e' conv stride 2 com padding 1 -> a dimensao arredonda
        p/ CIMA (ceil), o que importa em latente de lado impar (resolucao livre).
        """
        d, h, w = {}, int(H), int(W)
        for _ in range(niveis):
            d[h * w] = (h, w)
            h, w = -(-h // 2), -(-w // 2)
        return d

    # ---- mascaras (7.2) ----
    def _masks(self, li, t, H, W, add_time):
        self.probe.maps.clear()
        self.probe.dims = self._piramide(H, W)  # aceita a piramide, nao so H*W
        self.probe.target_npix = None
        self.probe.enabled = True
        self.pipe.unet.disable_adapters()
        with fase("mascara: sonda attn2 (fwd base)"), torch.no_grad():
            self.pipe.unet(li, t, encoder_hidden_states=self.pe,
                           added_cond_kwargs=self._cond(self.pp, add_time))
        self.probe.enabled = False
        if self.probe.maps:
            # quanto a sonda custou de VRAM: ela materializa q@kT por camada attn2 e
            # ACUMULA os mapas. O custo escala com n_tok — com prompt chunkado
            # (4 blocos) e' 4x o de um prompt curto, na resolucao mais alta do
            # pipeline. E' o primeiro lugar a olhar quando o hires estoura.
            mb = sum(x.numel() * x.element_size() for x in self.probe.maps) / 2**20
            porniv = {}
            for x in self.probe.maps:
                d = self.probe.dims.get(x.shape[1])
                porniv[f"{d[0]}x{d[1]}"] = porniv.get(f"{d[0]}x{d[1]}", 0) + 1
            print(f"[gsmde] sonda attn2: {len(self.probe.maps)} mapas "
                  f"({', '.join(f'{k}:{v}' for k, v in porniv.items())}) = {mb:.0f} MB "
                  f"| ctx {self.pe.shape[1]} tokens | latente {H}x{W}", flush=True)
        else:
            print(f"[gsmde] sonda attn2: NENHUM mapa capturado (latente {H}x{W}) — "
                  f"mascaras caem no uniforme 1/N", flush=True)
        raw = {}
        for n, toks in self.center_tokens.items():
            acc, cnt = None, 0
            for m in self.probe.maps:
                hw = self.probe.dims.get(m.shape[1])
                if hw is None or not toks:
                    continue
                valid = [i for i in toks if i < m.shape[2]]
                if not valid:
                    continue
                # cada linha do mapa e' um softmax sobre os tokens, entao a soma
                # sobre as ancoras do centro e' "fracao da atencao daquele pixel
                # que foi p/ este centro" — comparavel entre resolucoes.
                s = m[0, :, valid].sum(-1).float()
                if hw != (H, W):                 # /2 e /4 sobem p/ a grade do latente
                    s = torch.nn.functional.interpolate(
                        s.reshape(1, 1, *hw), size=(H, W),
                        mode="bilinear", align_corners=False).reshape(-1)
                acc = s if acc is None else acc + s
                cnt += 1
            if acc is not None and cnt:
                raw[n] = (acc / cnt).reshape(H, W)
        if not raw:
            self.probe.maps.clear()              # nao segura VRAM ate a proxima sonda
            return None
        self.probe.maps.clear()
        names = list(raw)
        with fase("mascara: montar territorios"):
            st = torch.stack([raw[n] for n in names], 0).float()
            st = (st - st.amin()) / (st.amax() - st.amin() + 1e-6)
            w = torch.softmax(st * 4.0, dim=0)
        return {n: w[i].to(self.dev, torch.float16) for i, n in enumerate(names)}

    # ---- denoise (txt2img e img2img) ----
    def _maybe_preview(self, lat, i, n, every):
        """Preview ao vivo via TAESD. So dispara se o worker setou self._preview_cb e
        every>0. Decodifica o latente atual (barato) e entrega a PIL pequena. Ultimo
        passo sempre entra. Roda no compute-queue + yield -> sem freeze. Cosmetico:
        qualquer erro no decode vira aviso, nunca aborta o denoise."""
        cb = getattr(self, "_preview_cb", None)
        if not cb or not every:
            return
        if i % every != 0 and i != n:
            return
        try:
            if getattr(self, "_tae", None) is None:
                from taesd_preview import TaePreview
                self._tae = TaePreview(self.dev, dtype=torch.float16)
            img = self._tae.decode(lat)
            yield_gpu(self.yield_ms)
            cb(img, i, n)
        except Exception as e:
            print(f"[preview] ignorado: {e}", file=sys.stderr, flush=True)

    def denoise(self, size=None, steps=26, seed=1234, init: Image.Image = None,
                strength=1.0, mask_every=4, global_weight=0.25, size_hw=None,
                batch_cfg=True, progress=True, cb=None, iso=0.4, backbone_assert=None,
                preview_every=0, mask_lat=None, lat_ctx=None):
        """init=None -> txt2img. init=PIL -> img2img com denoise=strength.
        size_hw=(H_px,W_px) permite retangulo; senao usa size quadrado.

        iso: fracao dos passos com ISOLAMENTO da self-attn regional (0..1).
        Dois estagios (ideia do usuario, alinhada a assimetria temporal da §8):
          - passos 0..iso*N: cada centro esboca seu ser/zona CEGO aos vizinhos ->
            identidades distintas nascem na fase de estrutura;
          - passos iso*N..N: self-attn LIBERADA, todos veem a tela toda e harmonizam
            luz/cor/estilo. A estrutura ja esta travada pelo baixo ruido, entao nao
            se re-fundem — so concordam no acabamento. Sem isso, o isolamento total
            separa tambem o FUNDO (metade vermelha, metade branca).
          iso=0 -> nunca isola (comportamento antigo). iso=1 -> isola sempre."""
        pipe, dev = self.pipe, self.dev
        # backbone dominante: usa o param, senao o default de instancia (so com centros)
        if backbone_assert is None and self.regional:
            backbone_assert = getattr(self, "backbone_assert", None)
        if init is not None:
            Wpx, Hpx = init.size
        else:
            Hpx, Wpx = size_hw or (size, size)
        self._revisa_residencia(Wpx, Hpx)
        H, W = Hpx // 8, Wpx // 8
        add_time = torch.tensor([[Hpx, Wpx, 0, 0, Hpx, Wpx]], device=dev, dtype=torch.float16)
        gen = torch.Generator(dev).manual_seed(seed)
        pipe.scheduler.set_timesteps(steps, device=dev)
        ts = pipe.scheduler.timesteps
        if init is None:
            lat = torch.randn((1, 4, H, W), generator=gen, device=dev,
                              dtype=torch.float16) * pipe.scheduler.init_noise_sigma
        else:
            lat0 = self.encode_image(init)
            i0 = max(0, min(len(ts) - 1, int(len(ts) * (1.0 - strength))))
            ts = ts[i0:]
            noise = torch.randn(lat0.shape, generator=gen, device=dev, dtype=torch.float16)
            lat = pipe.scheduler.add_noise(lat0, noise, ts[:1])

        masks = None
        iso_ate = int(iso * len(ts))         # ate que passo a self-attn fica isolada
        t_ini = time.time()
        # INPAINT COM HALO: dentro da mascara o latente evolui a partir de RUIDO PURO
        # (geracao do zero, nao re-noise da imagem); fora dela o contexto original e'
        # reinjetado NO NIVEL DE RUIDO DAQUELE PASSO. E' isso que faz o conteudo novo
        # respeitar luz/perspectiva/paleta do entorno: a atencao sempre enxerga um
        # halo coerente com onde a difusao esta, em vez de um entorno limpo demais.
        ruido_ctx = None
        if mask_lat is not None and lat_ctx is not None:
            ruido_ctx = torch.randn(lat_ctx.shape, generator=gen, device=dev,
                                    dtype=torch.float16)

        for si, t in enumerate(ts):
            self._iso_ativo = si < iso_ate
            if ruido_ctx is not None:
                ctx_t = pipe.scheduler.add_noise(lat_ctx, ruido_ctx, t.reshape(1))
                lat = lat * mask_lat + ctx_t * (1 - mask_lat)
            li = pipe.scheduler.scale_model_input(lat, t)
            if si % mask_every == 0 and self.regional and getattr(self, "usar_mascaras", True):
                _t = time.time()
                masks = self._masks(li, t, H, W, add_time)
                yield_gpu(self.yield_ms)
                self._perf_add("mascaras (sonda+fwd extra)", time.time() - _t)

            # BASE PURO: zero centros = o backbone sozinho. E' o controle dos
            # experimentos — sem ele, uma imagem torta a 2048 nao diz se a culpa e'
            # do multi-centro ou do SDXL fora da resolucao de treino. Sem este ramo,
            # enable_adapters() estoura com "No adapter loaded".
            if not self.regional and not self.globs:
                cn = {}
                if getattr(self, "cn_cond", None) is not None:
                    cond = F.interpolate(self.cn_cond, size=(Hpx, Wpx),
                                         mode="bilinear", align_corners=False)
                    with torch.no_grad():
                        down, mid = self.controlnet(
                            torch.cat([li, li]), t,
                            encoder_hidden_states=torch.cat([self.npe, self.pe]),
                            added_cond_kwargs=self._add_cond(self.npp, self.pp, add_time),
                            controlnet_cond=torch.cat([cond, cond]),
                            conditioning_scale=self.cn_scale, return_dict=False)
                    cn = {"down_block_additional_residuals": down,
                          "mid_block_additional_residual": mid}
                    yield_gpu(self.yield_ms)
                with fase("forward: base puro (lote 2)"), torch.no_grad():
                    o = pipe.unet(torch.cat([li, li]), t,
                                  encoder_hidden_states=torch.cat([self.npe, self.pe]),
                                  added_cond_kwargs=self._add_cond(self.npp, self.pp, add_time),
                                  **cn).sample
                yield_gpu(self.yield_ms)
                u, c = o.chunk(2)
                # generator=gen: mesmo motivo do caminho com centros (ver abaixo) —
                # sem ele o ruido ancestral vem do RNG global e a seed nao fecha.
                lat = pipe.scheduler.step(u + self.cfg * (c - u), t, lat,
                                          generator=gen).prev_sample
                if cb:
                    cb(si + 1, len(ts), Hpx)
                self._maybe_preview(lat, si + 1, len(ts), preview_every)
                continue

            pipe.unet.enable_adapters()
            box_masks = self._box_masks(H, W) if self.regioes else None

            def guided(cid):
                """CFG DENTRO do proprio modelo do centro: uncond e cond do MESMO
                adapter. Misturar uncond do base com cond do especialista soma ao
                diferencial o deslocamento base->especialista e o cfg amplifica-o
                ate saturar o latente (meio-tom). uncond+cond vao no mesmo lote.
                'cid' e' a instancia (person#2); o adapter e' a base dela."""
                nome = base_do_centro(cid)
                cidl = parse_regiao(cid)[0]         # tira o @box; mantem o #instancia
                # a fatia do centro, se houver; senao o prompt inteiro. A chave e' o
                # id LIMPO — a fatia veio como 'person#1', a lista tem 'person#1@...'.
                pe, npe, pp, npp = self.fatias.get(cidl, self.fatias.get(cid,
                                    (self.pe, self.npe, self.pp, self.npp)))
                # self-attn confinada a regiao do centro: e' o que impede a fusao.
                # so isola na fase de estrutura; depois self-attn global harmoniza
                _t = time.time()
                self._set_regiao_selfattn((box_masks or {}).get(cidl)
                                          if getattr(self, "_iso_ativo", True) else None)
                self._perf_add("regiao selfattn", time.time() - _t)
                if self.page_adapters:
                    _t = time.time()
                    self._page(nome, self.dev)      # sobe so o nicho ativo
                    # a carona esta ativa em TODA passada: se for paginada para fora
                    # a cada troca de centro, o custo dela deixa de ser zero
                    for _c, _ in (getattr(self, "caronas", None) or []):
                        self._page(_c, self.dev)
                    self._perf_add("paginar->VRAM", time.time() - _t)
                # CARONAS: adaptadores SEM territorio viajam DENTRO desta passada.
                #
                # Um centro custa uma passada completa do UNet porque a predicao dele
                # precisa ser mascarada por regiao. Quem nao tem regiao — estilo,
                # camera, grade de cor, realce de detalhe — nao usa esse beneficio e
                # nao deveria pagar esse preco: o peft compoe varias correcoes de
                # baixo posto na MESMA multiplicacao, e a carona sai de graca.
                #
                # Medido em 13/08: as passadas sao ~98% do custo. Converter um global
                # de passada propria para carona remove uma passada inteira por passo.
                _t = time.time()
                car = getattr(self, "caronas", None) or []
                if car:
                    pipe.unet.set_adapters(
                        [nome] + [c for c, _ in car],
                        [self.adapter_scale] + [w for _, w in car])
                else:
                    pipe.unet.set_adapters([nome], [self.adapter_scale])
                self._perf_add("set_adapters", time.time() - _t)
                if batch_cfg:
                    _t = time.time()
                    with fase("forward: uncond+cond (lote 2)"), torch.no_grad():
                        o = pipe.unet(torch.cat([li, li]), t,
                                      encoder_hidden_states=torch.cat([npe, pe]),
                                      added_cond_kwargs=self._add_cond(npp, pp, add_time)).sample
                    self._perf_add("unet centro (enfileira)", time.time() - _t)
                    _t = time.time()
                    yield_gpu(self.yield_ms)
                    self._perf_add("yield_gpu (sync+sleep)", time.time() - _t)
                    u, c = o.chunk(2)
                else:
                    with torch.no_grad():
                        u = pipe.unet(li, t, encoder_hidden_states=npe,
                                      added_cond_kwargs=self._cond(npp, add_time)).sample
                    yield_gpu(self.yield_ms)
                    with torch.no_grad():
                        c = pipe.unet(li, t, encoder_hidden_states=pe,
                                      added_cond_kwargs=self._cond(pp, add_time)).sample
                    yield_gpu(self.yield_ms)
                if self.page_adapters:
                    if nome not in [c for c, _ in (getattr(self, "caronas", None) or [])]:
                        self._despeja_por_pressao(nome)  # so devolve se apertou
                self._set_regiao_selfattn(None)     # limpa p/ o proximo
                return u + self.cfg * (c - u)

            # regiao explicita (caixa disjunta) vence a mascara de atencao. As caixas
            # dos personagens repartem a unidade ENTRE SI (particao). Um centro SEM
            # caixa (scenary) e' FUNDO: preenche so o que sobra da cobertura dos que
            # tem caixa (1 - cobertura), atras deles. Sem isso ele somava peso 1/3 por
            # cima do 1,0 dos personagens -> 1,33 de ruido em toda a tela -> satura.
            resto, n_fundo = None, 0
            if box_masks:
                cob = torch.stack(list(box_masks.values())).sum(0).clamp(max=1.0)
                resto = (1.0 - cob)                       # o fundo vai aqui
                # id da lista tem o @box ('person#1@0,0,..'); a caixa e' por id limpo
                # ('person#1'). Sem normalizar, os personagens caiam no ramo do fundo.
                n_fundo = sum(1 for c, _ in self.regional
                              if parse_regiao(c)[0] not in box_masks)
            def _mask_de(n, cidlimpo):
                if box_masks and cidlimpo in box_masks:
                    return box_masks[cidlimpo]            # personagem: sua caixa
                elif box_masks is not None:
                    return resto / max(n_fundo, 1)        # fundo: o que sobra
                elif masks:
                    return masks.get(n)                   # sem regiao: mascara de atencao
                return None

            if backbone_assert is not None:
                # BACKBONE DOMINANTE (ideia do usuario): base-pura em 1.0 + cada centro
                # ADICIONA seu delta so na regiao (mascara), escalado por cobertura.
                # Como m_i integra a cobertura, assert*m_i pesa o centro pela % de
                # latentes que ele ocupa (25% -> ~0.2 com assert 0.8). O centro deixa de
                # SUBSTITUIR o backbone (o que inundava/fritava) e passa a NUDGE-a.
                _t = time.time()
                pipe.unet.disable_adapters(); self._set_regiao_selfattn(None)
                self._perf_add("enable/disable adapters", time.time() - _t)
                cnb = {}
                if getattr(self, "cn_cond", None) is not None:
                    condb = F.interpolate(self.cn_cond, size=(Hpx, Wpx),
                                          mode="bilinear", align_corners=False)
                    with torch.no_grad():
                        dnb, mdb = self.controlnet(
                            torch.cat([li, li]), t,
                            encoder_hidden_states=torch.cat([self.npe, self.pe]),
                            added_cond_kwargs=self._add_cond(self.npp, self.pp, add_time),
                            controlnet_cond=torch.cat([condb, condb]),
                            conditioning_scale=self.cn_scale, return_dict=False)
                    cnb = {"down_block_additional_residuals": dnb,
                           "mid_block_additional_residual": mdb}
                _t = time.time()
                with fase("forward: backbone (lote 2)"), torch.no_grad():
                    o = pipe.unet(torch.cat([li, li]), t,
                                  encoder_hidden_states=torch.cat([self.npe, self.pe]),
                                  added_cond_kwargs=self._add_cond(self.npp, self.pp, add_time),
                                  **cnb).sample
                self._perf_add("unet backbone (enfileira)", time.time() - _t)
                _t = time.time()
                yield_gpu(self.yield_ms)
                self._perf_add("yield_gpu (sync+sleep)", time.time() - _t)
                _t = time.time()
                pipe.unet.enable_adapters()
                self._perf_add("enable/disable adapters", time.time() - _t)
                ub, cb_ = o.chunk(2)
                gb = ub + self.cfg * (cb_ - ub)
                comb = gb
                covs = []
                for n, _ in self.regional:
                    g = guided(n)
                    m = _mask_de(n, parse_regiao(n)[0])
                    if m is None:
                        m = torch.full((H, W), 1.0 / max(len(self.regional), 1),
                                       device=dev, dtype=torch.float16)
                    covs.append((base_do_centro(n), float(m.float().mean())))
                    comb = comb + backbone_assert * m.unsqueeze(0).unsqueeze(0) * (g - gb)
                if si == 0:
                    # sempre, nao so com progress=True: e' este numero que diz se os
                    # territorios sao REAIS ou degenerados. Todos iguais a 1/N = a
                    # sonda nao produziu mascara e cada centro esta pintando a tela
                    # inteira — que foi o estado silencioso do pipeline ate agora.
                    print("[gsmde] cobertura por centro: "
                          + ", ".join(f"{nm}={cv*100:.0f}%(peso {backbone_assert*cv:.2f})"
                                      for nm, cv in covs) + " | backbone 1.0", flush=True)
                for n in self.globs:
                    g = guided(n)
                    comb = comb * (1 - global_weight) + global_weight * g
            else:
                comb = None
                for n, _ in self.regional:
                    g = guided(n)
                    m = _mask_de(n, parse_regiao(n)[0])
                    part = (m.unsqueeze(0).unsqueeze(0) * g) if m is not None \
                        else g / max(len(self.regional), 1)
                    comb = part if comb is None else comb + part
                for n in self.globs:
                    g = guided(n)
                    comb = g if comb is None else comb * (1 - global_weight) + global_weight * g
            with fase("scheduler: step"):
                # generator=gen e' OBRIGATORIO aqui. O EulerAncestral injeta ruido
                # NOVO a cada passo; sem o generator ele puxa do RNG global, e a
                # seed passa a governar so' o latente inicial. Efeito medido em
                # 16/08: duas corridas com a MESMA seed e config deram nitidez
                # 346,2 e 201,5 — a imagem nao era reproduzivel, e comparacao A/B
                # entre condicoes carregava ruido de corrida do tamanho do efeito
                # que se queria medir.
                lat = pipe.scheduler.step(comb, t, lat, generator=gen).prev_sample
            if cb:
                cb(si + 1, len(ts), Hpx)      # p/ a barra de progresso da UI
            self._maybe_preview(lat, si + 1, len(ts), preview_every)
            if progress:
                el = time.time() - t_ini
                eta = el / (si + 1) * (len(ts) - si - 1)
                print(f"\r    [{H*8}px] passo {si+1}/{len(ts)}  {el/(si+1):.1f}s/passo  "
                      f"ETA {eta/60:.1f}min   ", end="", flush=True)
        if progress:
            print(f"\r    [{H*8}px] {len(ts)} passos em {(time.time()-t_ini)/60:.1f}min"
                  f"{' '*20}", flush=True)
        # ONDE O TEMPO FOI. Sem isto so se sabe o total por passo, e a conta do
        # usuario (5 forwards x 2s/it = 10s/passo, medido 45s) fica sem explicacao.
        d = getattr(self, "_perf", None)
        if d:
            tot = time.time() - t_ini
            itens = sorted(d.items(), key=lambda kv: -kv[1])
            resumo = " | ".join(f"{k} {v:.1f}s ({100*v/tot:.0f}%)" for k, v in itens if v > 0.05)
            contado = sum(v for _, v in itens)
            print(f"[perf] {len(ts)} passos em {tot:.1f}s ({tot/len(ts):.2f}s/passo): {resumo}"
                  f" | nao contabilizado {tot - contado:.1f}s", flush=True)
            self._perf = {}
        return self.decode(lat)

    def inpaint_halo(self, imagem, mascara, halo=128, blend=48, steps=30, seed=7,
                     global_weight=0.25, backbone_assert=None, cb=None,
                     preview_every=0, max_px=1_500_000, ctx_global=0.0):
        """Inpaint com halo de contexto — mascara manual, SEM tiling e SEM reescala.

        Tres camadas, de fora p/ dentro (a ordem importa e e' a do ultra_halo):
          1. HALO      anel externo que o modelo VE e nao reescreve. De onde vem
                       luz, perspectiva e paleta p/ o novo conteudo encaixar.
          2. BLENDING  faixa de feather onde o novo se costura ao existente.
          3. NUCLEO    a mascara pintada: difusao a partir de RUIDO PURO, guiada
                       pelo prompt. Geracao do zero, nao re-noise.

        POR QUE SEM TILING E SEM SUPERSAMPLING (decisao do usuario, e ele tem razao):
        no ultra o upscale+underscale existe p/ o modelo desenhar detalhe fino e
        depois adensa-lo. Aqui a regiao quase nunca e' quadrada, e reescalar SO a
        parte mascarada daria estatistica de ruido diferente da vizinhanca intocada
        — a costura apareceria mesmo com feather perfeito ("photoshop mal feito").
        Entao tudo acontece na resolucao nativa da imagem.
        """
        import numpy as np
        from PIL import Image, ImageFilter

        W0, H0 = imagem.size
        m = mascara.convert("L").resize((W0, H0), Image.LANCZOS)
        arr = np.asarray(m, np.float32) / 255.0
        ys, xs = np.where(arr > 0.02)
        if not len(ys):
            raise ValueError("mascara vazia: nada a gerar")

        # recorte = caixa da mascara + halo (contexto) + blend, em multiplo de 8
        folga = int(halo) + int(blend)
        x0 = max(0, int(xs.min()) - folga); y0 = max(0, int(ys.min()) - folga)
        x1 = min(W0, int(xs.max()) + 1 + folga); y1 = min(H0, int(ys.max()) + 1 + folga)
        x0, y0 = x0 & ~7, y0 & ~7
        x1 = min(W0, ((x1 + 7) & ~7)); y1 = min(H0, ((y1 + 7) & ~7))
        cw, ch = (x1 - x0) & ~7, (y1 - y0) & ~7
        if cw < 64 or ch < 64:
            raise ValueError(f"regiao pequena demais ({cw}x{ch})")
        if cw * ch > max_px:
            # sem tiling e sem reescala nao ha como diluir: avisa com o numero em
            # vez de degradar em silencio (degradar traria de volta o mismatch)
            raise ValueError(
                f"regiao de {cw}x{ch} ({cw*ch/1e6:.1f} MP) excede o limite de "
                f"{max_px/1e6:.1f} MP. Pinte uma mascara menor ou reduza o halo.")

        corte = imagem.convert("RGB").crop((x0, y0, x0 + cw, y0 + ch))
        m_corte = m.crop((x0, y0, x0 + cw, y0 + ch))
        # camada 2: o feather que costura. Borrar a mascara cria a rampa de blending.
        m_suave = m_corte.filter(ImageFilter.GaussianBlur(max(1, int(blend) // 2)))

        lat_ctx = self.encode_image(corte)                    # contexto (halo)
        Hl, Wl = lat_ctx.shape[-2], lat_ctx.shape[-1]
        ml = torch.from_numpy(np.asarray(m_suave, np.float32) / 255.0)
        ml = F.interpolate(ml.view(1, 1, ch, cw), size=(Hl, Wl),
                           mode="bilinear", align_corners=False).to(self.dev, torch.float16)

        print(f"[gsmde] inpaint halo: regiao {cw}x{ch} (latente {Wl}x{Hl}) | "
              f"halo {halo}px, blend {blend}px | mascara cobre "
              f"{float(ml.mean())*100:.0f}% do recorte", flush=True)

        # CONTEXTO GLOBAL (2o contexto): a imagem INTEIRA entra como image prompt via
        # IP-Adapter. Sem isto o mundo do modelo acaba no halo — ele redesenha "uma"
        # xicara plausivel em vez DAQUELA xicara, e o que se pede (vapor saindo do cha)
        # sai desconectado da cena. Note que isto NAO e' re-noise: o latente dentro da
        # mascara continua nascendo de ruido puro; a imagem original entra como
        # CONDICIONAMENTO, nao como ponto de partida. Ver e partir de sao coisas
        # diferentes — e' o que permite gerar informacao nova que ainda respeita a cena.
        usou_ip = False
        if ctx_global and ctx_global > 0.01:
            try:
                self.set_ip_adapter(imagem.convert("RGB"), scale=float(ctx_global))
                usou_ip = True
                print(f"[gsmde] contexto global ON (ip-adapter, escala {ctx_global})",
                      flush=True)
            except Exception as e:
                print(f"[gsmde] contexto global falhou ({e}); seguindo so com o halo",
                      flush=True)
        try:
            # strength=1.0 => parte de ruido puro dentro da mascara (geracao do zero)
            novo = self.denoise(size_hw=(ch, cw), steps=steps, seed=seed,
                                strength=1.0, global_weight=global_weight,
                                progress=False, cb=cb, backbone_assert=backbone_assert,
                                preview_every=preview_every,
                                mask_lat=ml, lat_ctx=lat_ctx)
        finally:
            if usou_ip:
                self.set_ip_adapter(None)     # nao vaza p/ a proxima geracao

        # cola de volta com o MESMO feather: fora da mascara o pixel original manda
        saida = imagem.convert("RGB").copy()
        saida.paste(novo, (x0, y0), m_suave)
        return saida

    # ---- z-order (quem fica na frente na colagem) ----
    def _z_order(self):
        """Ordem de profundidade dos centros regionais: fundo primeiro, frente por
        cima. Analise do prompt/nicho, com override explicito '@...@zN'.

        Default: fundo (scenary/biomes/locals) atras; personagens (person/animals)
        na frente. MAS a semantica pode inverter — 'garota de costas numa cadeira'
        quer a cadeira SOBRE a personagem. Palavras de oclusao no prompt do centro
        ('chair','desk','on','holding') sobem o z do movel/objeto. So heuristica; o
        @zN manda quando dado."""
        FUNDO = {"scenary", "biomes", "locals", "background"}
        z = {}
        for cid, palavras in self.regional:
            base = base_do_centro(cid)
            zexp = None
            if "@z" in cid:
                try:
                    zexp = int(cid.split("@z")[1].split("@")[0].split(",")[0])
                except ValueError:
                    zexp = None
            if zexp is not None:
                z[cid] = zexp
            elif base in FUNDO:
                z[cid] = 0
            else:
                # objeto de mobiliario que oclui? sobe meio nivel
                fat = " ".join(palavras).lower()
                oclui = any(w in fat for w in ("chair", "desk", "table", "counter",
                                               "wheel", "steering", "in front"))
                z[cid] = 3 if oclui else 2
        return z

    def collage(self, size=768, seed=1234, bg_steps=8, sketch_steps=15,
                blend_hi_steps=5, blend_lo_steps=10, global_weight=0.25,
                iso=1.0, progress=True, cb=None):
        """Colagem em latente (ideia do usuario, 5 fases). Resolve a oclusao que a
        combinacao convexa por-pixel nao trata: o fundo deixa de ser desenhado sobre
        o personagem, mas PODE oclui-lo quando a semantica pede (cadeira sobre a
        garota de costas), via ordem-z.

          1) BOOTSTRAP  : bg_steps passos so com o FUNDO (peso menor) -> tela+luz global
          2) SKETCH     : cada centro copia o latente e desenha o SEU proprio esboco,
                          isolado (self-attn regional), sketch_steps passos
          3) COLAGEM    : mescla os latentes por caixa E por ORDEM-Z (tras->frente),
                          alpha = mascara suave -> objetos de fundo nao cobrem a frente
          4) BLEND ALTO : blend_hi_steps passos JUNTOS, denoise mais alto, costura as juntas
          5) BLEND BAIXO: blend_lo_steps passos JUNTOS, denoise baixo, acabamento coerente
        """
        pipe, dev = self.pipe, self.dev
        Hpx = Wpx = size
        H, W = Hpx // 8, Wpx // 8
        add_time = torch.tensor([[Hpx, Wpx, 0, 0, Hpx, Wpx]], device=dev, dtype=torch.float16)
        total = bg_steps + sketch_steps + blend_hi_steps + blend_lo_steps
        pipe.scheduler.set_timesteps(total, device=dev)
        ts = pipe.scheduler.timesteps
        # Passo Euler MANUAL, sem estado: o scheduler do diffusers guarda step_index
        # interno que avanca a cada .step(); na fase de sketch eu passo cada centro
        # separado -> o contador estoura. Com sigmas explicitos, cada centro (e cada
        # fase) avanca sozinho sem tocar em estado partilhado.
        sig = pipe.scheduler.sigmas          # [total+1], sig[-1]=0
        def T(i):
            return ts[min(i, len(ts) - 1)]
        def scale(lat, i):                   # scale_model_input do Euler
            return lat / ((sig[i] ** 2 + 1) ** 0.5)
        def passo(lat, i, eps):              # Euler epsilon: lat + eps*(sig_next - sig)
            sn = sig[i + 1] if i + 1 < len(sig) else torch.zeros_like(sig[i])
            return lat + eps * (sn - sig[i])
        gen = torch.Generator(dev).manual_seed(seed)
        lat = torch.randn((1, 4, H, W), generator=gen, device=dev,
                          dtype=torch.float16) * sig[0]
        box = self._box_masks(H, W) or {}
        regionais = [c for c, _ in self.regional]
        fundo_ids = [c for c in regionais if parse_regiao(c)[0] not in box]
        person_ids = [c for c in regionais if parse_regiao(c)[0] in box]
        zc = self._z_order()
        t_ini = time.time()

        def prog(i, tag):
            if cb:
                cb(i, total, Hpx)
            if progress:
                print(f"\r    [colagem {tag}] passo {i}/{total}  "
                      f"{(time.time()-t_ini)/60:.1f}min  ", end="", flush=True)

        i = 0
        # FASE 1 — bootstrap do fundo (global, sem isolar)
        self._set_regiao_selfattn(None)
        for k in range(bg_steps):
            li = scale(lat, i)
            g = None
            for c in (fundo_ids or person_ids):
                pr = self._pred_centro(c, li, T(i), add_time)
                g = pr if g is None else g + pr
            g = g / max(len(fundo_ids or person_ids), 1)
            lat = passo(lat, i, g); i += 1; prog(i, "bg")

        # FASE 2 — cada centro esboca o SEU latente independente, isolado
        lats = {c: lat.clone() for c in regionais}
        for k in range(sketch_steps):
            for c in regionais:
                li = scale(lats[c], i)
                rsa = box.get(parse_regiao(c)[0]) if (iso and c in person_ids) else None
                g = self._pred_centro(c, li, T(i), add_time, regiao_sa=rsa)
                lats[c] = passo(lats[c], i, g)
            i += 1; prog(i, "sketch")

        # FASE 3 — COLAGEM por ordem-z: pinta do fundo p/ a frente, alpha=caixa suave
        lat = lats[fundo_ids[0]] if fundo_ids else next(iter(lats.values()))
        for c in sorted(person_ids, key=lambda x: zc.get(x, 2)):
            a = box.get(parse_regiao(c)[0])
            if a is None:
                continue
            a = a.view(1, 1, H, W)
            lat = lat * (1 - a) + lats[c] * a       # frente cobre o fundo na sua caixa

        # FASES 4+5 — refino conjunto (self-attn global harmoniza), denoise alto->baixo
        self._set_regiao_selfattn(None)
        for k in range(blend_hi_steps + blend_lo_steps):
            li = scale(lat, i)
            comb = None
            cob = torch.stack(list(box.values())).sum(0).clamp(max=1.0) if box else None
            for c in regionais:
                cidl = parse_regiao(c)[0]
                g = self._pred_centro(c, li, T(i), add_time)
                if cidl in box:
                    m = box[cidl]
                elif cob is not None:
                    m = (1 - cob) / max(len(fundo_ids), 1)
                else:
                    m = None
                part = m.view(1, 1, H, W) * g if m is not None else g / len(regionais)
                comb = part if comb is None else comb + part
            for c in self.globs:
                g = self._pred_centro(c, li, T(i), add_time)
                comb = comb * (1 - global_weight) + global_weight * g
            lat = passo(lat, i, comb); i += 1; prog(i, "blend")

        if progress:
            print(f"\r    [colagem] {total} passos em {(time.time()-t_ini)/60:.1f}min{' '*15}",
                  flush=True)
        return self.decode(lat)


# ==========================================================================
# TILED i2i / HIRES EM PEDACOS (fork roxo — DESIGN_tiled_i2i.md)
# MultiDiffusion no NOSSO loop manual (sem pipeline do diffusers = sem freeze):
# upscale global -> denoise por TILES com fusao dos eps no overlap A CADA PASSO
# (o contexto global vive no latente compartilhado; a costura some porque os
# tiles concordam no overlap em todos os passos) -> tiled VAE (ja ativo).
# Sem wd14: cada tile ve o PROMPT GLOBAL; o contexto local vem do latente.
# ==========================================================================

def _tile_pos(L, t, o):
    """Posicoes de janelas de tamanho t com overlap o cobrindo L (ultima alinha no fim)."""
    if L <= t:
        return [0]
    stride = max(1, t - o)
    xs = list(range(0, L - t + 1, stride))
    if xs[-1] != L - t:
        xs.append(L - t)
    return xs


def _feather2d(t, o, dev):
    """Peso do tile: 1 no miolo, rampa linear na zona de overlap (fusao suave)."""
    r = torch.ones(t, device=dev)
    for i in range(min(o, t // 2)):
        w = (i + 1) / (o + 1)
        r[i] = min(r[i].item(), w)
        r[t - 1 - i] = min(r[t - 1 - i].item(), w)
    return (r.view(-1, 1) * r.view(1, -1)).view(1, 1, t, t)


def tiled_i2i(self, image, prompt, neg=None, scale=1.2, tile_px=512, overlap_px=96,
              strength=0.35, steps=20, seed=7, cfg=None, progress=True):
    """i2i de ALTA resolucao em tiles (MultiDiffusion), sem estourar 8GB.

    1) upscale global (Lanczos) por `scale` — contexto inteiro preservado;
    2) encode tiled VAE -> latente GRANDE unico;
    3) a cada passo: eps de cada tile (~512px) e FUSAO ponderada no overlap ->
       um eps global -> um passo de Euler no latente inteiro. Os tiles nunca
       divergem porque concordam no overlap em todos os passos (= sem costura);
    4) decode tiled VAE.
    Base puro (adapters OFF): o refino de detalhe e' do backbone; os centros ja
    fizeram o trabalho semantico na geracao que produziu a imagem de entrada."""
    from PIL import Image as _Img
    pipe, dev = self.pipe, self.dev
    W0, H0 = image.size
    W = max(64, int(W0 * scale)) // 8 * 8
    H = max(64, int(H0 * scale)) // 8 * 8
    img = image.resize((W, H), _Img.LANCZOS)
    self.set_prompt(prompt, neg or NEG)
    pipe.unet.disable_adapters()
    self._set_regiao_selfattn(None)

    lat0 = self.encode_image(img)                     # tiled VAE (ja ativo no motor)
    Hl, Wl = lat0.shape[-2], lat0.shape[-1]
    t = max(32, tile_px // 8)
    o = max(4, overlap_px // 8)
    ys, xs = _tile_pos(Hl, t, o), _tile_pos(Wl, t, o)
    # add_time do TAMANHO DO TILE (o UNet ve um frame de ~512): micro-cond coerente
    # com o que ele realmente processa; original_size = imagem toda (contexto).
    tpx = t * 8
    add_time = torch.tensor([[H, W, 0, 0, tpx, tpx]], device=dev, dtype=torch.float16)

    gen = torch.Generator(dev).manual_seed(seed)
    pipe.scheduler.set_timesteps(steps, device=dev)
    sig = pipe.scheduler.sigmas.to(dev)
    i0 = max(0, min(len(sig) - 2, int((len(sig) - 1) * (1.0 - strength))))
    noise = torch.randn(lat0.shape, generator=gen, device=dev, dtype=torch.float16)
    lat = lat0 + noise * sig[i0]
    cfg = cfg if cfg is not None else self.cfg
    wtile = _feather2d(t, o, dev).to(torch.float16)
    ts_all = pipe.scheduler.timesteps
    t_ini = time.time()
    n_pass = len(sig) - 1 - i0
    for si in range(i0, len(sig) - 1):
        tt = ts_all[si]
        li_full = lat / ((sig[si] ** 2 + 1) ** 0.5)   # scale_model_input (Euler)
        eps_acc = torch.zeros_like(lat)
        w_acc = torch.zeros((1, 1, Hl, Wl), device=dev, dtype=torch.float16)
        for y in ys:
            for x in xs:
                li = li_full[:, :, y:y + t, x:x + t]
                with torch.no_grad():
                    o2 = pipe.unet(torch.cat([li, li]), tt,
                                   encoder_hidden_states=torch.cat([self.npe, self.pe]),
                                   added_cond_kwargs=self._add_cond(self.npp, self.pp, add_time)).sample
                u, c = o2.chunk(2)
                eps = u + cfg * (c - u)
                eps_acc[:, :, y:y + t, x:x + t] += eps * wtile
                w_acc[:, :, y:y + t, x:x + t] += wtile
                yield_gpu(self.yield_ms)
        eps_full = eps_acc / w_acc.clamp(min=1e-4)    # fusao MultiDiffusion
        lat = lat + eps_full * (sig[si + 1] - sig[si])  # Euler stateless
        if progress:
            el = time.time() - t_ini
            done = si - i0 + 1
            print(f"\r    [tiled {W}x{H}] passo {done}/{n_pass} "
                  f"({len(ys)*len(xs)} tiles) {el/done:.1f}s/passo   ", end="", flush=True)
    if progress:
        print("", flush=True)
    return self.decode(lat)


def ultra_tiled(self, image, prompt, neg=None, scale=1.2, seed=7, progress=True):
    """O 'ultra' especifico: passe 1 (detalhe, 0.40) + passe 2 hires leve (0.20).
    Duas passadas do mesmo mecanismo — a 2a amarra qualquer resto de emenda."""
    a = tiled_i2i(self, image, prompt, neg=neg, scale=scale, strength=0.40,
                  steps=20, seed=seed, progress=progress)
    return tiled_i2i(self, a, prompt, neg=neg, scale=1.0, strength=0.20,
                     steps=16, seed=seed + 1, progress=progress)


def ultra_halo(self, image, prompt, neg=None, scale=1.5, core=512, pad=64,
               overlap=96, denoise=0.35, passes=1, steps=18, seed=7, color_str=0.6,
               final_scale=0.9, solid_skip=True, use_centers=False, progress=True):
    """Ultra 'halo v3' NO ENGINE (sem servidor SD.Next, sem freeze).

    Sintese de detalhe por tiles com HALO DE CONTEXTO, validada no headless:
      - cada tile difunde CORE (regiao escrita) + halo de PAD px (contexto, nao
        escrito) -> o modelo ve o entorno e nao alucina na borda;
      - os CORES se sobrepoem (overlap) e colam com feather -> costura escondida;
      - SEM passe final full-res (o que estourava a VRAM em 8GB) -> cada op roda em
        baixa-res, cabe na VRAM, sem spill. i2i denoise baixo (0.35) quase nao
        alucina pois o conteudo principal ja existe.
    img2img de cada tile = self.denoise(init=tile, strength=denoise) -> loop
    hand-rolled com yield_gpu (Compute-queue, nao trava o Windows).
    Base-puro se a instancia nao tem centros; com centros, refina COM os adapters.
    """
    import numpy as np

    def core_positions(size, c, ov):
        if size <= c:
            return [0]
        step = max(1, c - ov)
        pos = list(range(0, size - c, step)) + [size - c]
        return sorted(set(pos))

    def color_match(work, ref, s):
        w = np.asarray(work, np.float32); r = np.asarray(ref, np.float32)
        out = w.copy()
        for ch in range(3):
            wm, ws = w[..., ch].mean(), max(1.0, w[..., ch].std())
            rm, rs = r[..., ch].mean(), max(1.0, r[..., ch].std())
            gain = 1 + (rs / ws - 1) * s
            off = (rm - wm * gain) * s
            out[..., ch] = np.clip(w[..., ch] * gain + off, 0, 255)
        return Image.fromarray(out.astype(np.uint8))

    def feather(w, h, ov, left, top, right, bottom):
        m = np.ones((h, w), np.float32)
        if ov > 0:
            ramp = np.linspace(0, 1, ov, dtype=np.float32)
            if left:   m[:, :ov] *= ramp[None, :]
            if right:  m[:, w-ov:] *= ramp[::-1][None, :]
            if top:    m[:ov, :] *= ramp[:, None]
            if bottom: m[h-ov:, :] *= ramp[::-1][:, None]
        return m

    def is_solid(crop, thresh=7):
        a = np.asarray(crop.resize((32, 32)), np.float32)
        return max(a[..., 0].std(), a[..., 1].std(), a[..., 2].std()) < thresh

    # refino base-puro por padrao: a composicao ja foi colocada na gen; aqui so
    # adiciona detalhe. Com centros, cada tile rodaria N especialistas e as regioes
    # (definidas p/ a imagem inteira) nao mapeiam pros tiles. use_centers=True forca.
    _saved = (self.regional, self.globs, self.fatias)
    if not use_centers:
        self.regional, self.globs, self.fatias = [], [], {}
        # um adapter da gen multi-centro pode ter ficado ATIVO e paginado na CPU;
        # no refino base-puro ele dispararia com peso na CPU -> device mismatch.
        try:
            self.pipe.unet.disable_adapters()
        except Exception:
            pass
    try:
        self.set_prompt(prompt, neg if neg is not None else NEG)
        return _ultra_halo_core(self, image, scale, core, pad, overlap, denoise, passes,
                                steps, seed, color_str, final_scale, solid_skip, progress,
                                core_positions, color_match, feather, is_solid, np)
    finally:
        self.regional, self.globs, self.fatias = _saved
        if not use_centers:
            try:
                self.pipe.unet.enable_adapters()
            except Exception:
                pass


def _ultra_halo_core(self, image, scale, core, pad, overlap, denoise, passes, steps,
                     seed, color_str, final_scale, solid_skip, progress,
                     core_positions, color_match, feather, is_solid, np):
    from PIL import Image
    # 1. upscale global (Lanczos; o detalhe vem do i2i por tile, nao do upscaler)
    W, H = int(image.width * scale) & ~7, int(image.height * scale) & ~7
    big = image.convert("RGB").resize((W, H), Image.LANCZOS)
    work = big.copy(); ref_base = big.copy()
    CORE = min(core, W, H); OV = min(overlap, CORE // 2)
    cxs, cys = core_positions(W, CORE, OV), core_positions(H, CORE, OV)
    cells = [(x, y) for y in cys for x in cxs]
    solid = set()
    if progress:
        print(f"[ultra_halo] {W}x{H} | core={CORE} halo={pad} ov={OV} | "
              f"{len(cxs)}x{len(cys)}={len(cells)} tiles x {passes}p (SEM passe final)", flush=True)
    for p in range(passes):
        # denoise decrescente entre passes: 1o cria detalhe, seguintes refinam sem destruir
        den = denoise if passes == 1 else denoise * (1 - 0.45 * (p / (passes - 1)))
        for ti, (cx, cy) in enumerate(cells):
            cw, ch = min(CORE, W - cx), min(CORE, H - cy)
            if solid_skip and (ti in solid or (p == 0 and is_solid(work.crop((cx, cy, cx + cw, cy + ch))))):
                solid.add(ti); continue
            dx0, dy0 = max(0, cx - pad), max(0, cy - pad)
            dx1, dy1 = min(W, cx + cw + pad), min(H, cy + ch + pad)
            tw, th = (dx1 - dx0) & ~7, (dy1 - dy0) & ~7
            tile = work.crop((dx0, dy0, dx0 + tw, dy0 + th))
            res = self.denoise(init=tile, strength=round(den, 3), steps=steps,
                               seed=seed + ti + p * 1000, progress=False)
            if res.size != (tw, th):
                res = res.resize((tw, th))
            ox, oy = cx - dx0, cy - dy0
            core_res = res.crop((ox, oy, ox + cw, oy + ch))
            core_res = color_match(core_res, ref_base.crop((cx, cy, cx + cw, cy + ch)), color_str)
            mask = feather(cw, ch, OV, cx > 0, cy > 0, cx + cw < W, cy + ch < H)
            cur = np.asarray(work.crop((cx, cy, cx + cw, cy + ch)), np.float32)
            new = np.asarray(core_res, np.float32)
            blend = (new * mask[..., None] + cur * (1 - mask[..., None])).astype(np.uint8)
            work.paste(Image.fromarray(blend), (cx, cy))
            if progress:
                print(f"\r[ultra_halo] passe {p+1}/{passes} tile {ti+1}/{len(cells)} "
                      f"den={den:.3f} dif={tw}x{th}->core{cw}x{ch}    ", end="", flush=True)
    if progress:
        print(flush=True)
    # color-match global + underscaler final (supersampling)
    work = color_match(work, image.resize(work.size), color_str)
    if abs(final_scale - 1) > 1e-3:
        work = work.resize((round(work.width * final_scale),
                            round(work.height * final_scale)), Image.LANCZOS)
    return work


GSMDE.tiled_i2i = tiled_i2i
GSMDE.ultra_tiled = ultra_tiled
GSMDE.ultra_halo = ultra_halo
