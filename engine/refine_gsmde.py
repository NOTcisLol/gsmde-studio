"""
Pipeline de refino GSMDE no padrao SD.NEXT:

    gerador (GSMDE multi-centro)  ->  upscaler (ESRGAN)  ->  hires fix (camadas)  ->  detailer

Filosofia das camadas (conforme pedido): quanto MAIOR o upscale, MENOR o denoise e
MAIS steps por camada. Subir de 768 direto p/ 1536 com denoise alto reescreve a
composicao; em camadas o modelo so preenche a frequencia alta que o ESRGAN inventou.

  camada 1: 768 -> 1152  denoise 0.45  (reconstroi textura/estrutura media)
  camada 2: 1152 -> 1536 denoise 0.28  (so detalhe fino, muitos steps)
  detailer: YOLO face/hands -> crop 768 -> denoise 0.35 -> cola com feather

Cada camada roda o motor GSMDE completo (mascaras por cross-attention + CFG
por-centro), entao os especialistas continuam mandando no seu territorio durante
todo o refino — nao e um refino "generico" por cima.

Uso:
  python src\\lora_pipeline\\refine_gsmde.py \
    --centers "person:girl,woman;scenary:park,trees,outdoors" --globals style \
    --prompt "1girl walking a dog in a park, outdoors, trees, full body, day"
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from gsmde_engine import GSMDE, NEG, parse_centers, yield_gpu

OUT = Path(r"D:\GSMDE\trainer\outputs\gsmde_gen")
UPS = {
    "anime": r"D:\Models\RealESRGAN\RealESRGAN_x4plus_anime_6B.pth",
    "real": r"D:\Models\RealESRGAN\RealESRGAN_x4plus.pth",
    "sharp": r"D:\Models\ESRGAN\ESRGAN-UltraSharp-4x.pth",
}
YOLO = {
    "face": r"D:\Models\yolo\face_yolov9c.pt",
    "hand": r"D:\Models\yolo\hand_yolov8s.pt",
}
YOLO_DIR = Path(r"D:\Models\yolo")


def yolo_path(kind: str) -> str | None:
    """Resolve o alvo do detailer. Aceita os apelidos historicos ('face','hand') E o
    NOME DO ARQUIVO em D:\\Models\\yolo (ex.: 'anzhc-head-seg-8n'), que e' o que as UIs
    listam. Antes so o dict valia: qualquer alvo novo caia no 'ausente, pulando' e o
    detailer nao rodava — silenciosamente, o que parecia bug do detector."""
    if not kind:
        return None
    p = YOLO.get(kind)
    if p and Path(p).exists():
        return p
    for cand in (Path(kind), YOLO_DIR / kind, YOLO_DIR / f"{kind}.pt",
                 YOLO_DIR / f"{kind}.onnx"):
        if cand.suffix and cand.exists():
            return str(cand)
    return None


def esrgan(img: Image.Image, model_path: str, tile=256, yield_ms=40) -> Image.Image:
    """Upscale ESRGAN via spandrel, em tiles.

    Duas coisas seguram a VRAM aqui, e as duas ja causaram travamento do PC:
      1. a saida 4x inteira na GPU (um 1152 vira 4608x4608 = 254MB em fp32, e o
         pico soma com a entrada e os tiles) -> acumulamos na RAM, tile a tile;
      2. a UNet residente durante o upscale -> quem chama deve tirar antes
         (upscale_to faz isso). VRAM sequencial: um modelo de cada vez.
    """
    import numpy as np
    from spandrel import ImageModelDescriptor, ModelLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ModelLoader().load_from_file(model_path)
    assert isinstance(m, ImageModelDescriptor)
    m.to(dev).eval()
    scale, pad = m.scale, 16
    src = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0)
    _, _, h, w = src.shape
    out = torch.zeros((1, 3, h * scale, w * scale), dtype=torch.uint8)   # RAM, nao VRAM
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(h, y1 + pad), min(w, x1 + pad)
            with torch.no_grad():
                t = m(src[:, :, ya:yb, xa:xb].to(dev))     # so o tile sobe
            oy, ox = (y0 - ya) * scale, (x0 - xa) * scale
            crop = t[:, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                (crop.clamp(0, 1) * 255).round().to(torch.uint8).cpu()   # desce na hora
            del t, crop
            yield_gpu(yield_ms)                 # cede a GPU ao desktop entre tiles
    a = out[0].permute(1, 2, 0).numpy()
    del m, src, out
    torch.cuda.empty_cache()
    return Image.fromarray(a)


def upscale_to(img: Image.Image, target_w: int, model_path: str, eng=None,
               yield_ms=40, tile=128, offload=False) -> Image.Image:
    """ESRGAN 4x e volta por Lanczos ate a largura alvo (mantem proporcao).

    O TILE decide, nao o offload. Medido a 1152 -> 4608, com a UNet residente:

        tile 256  -> pico 7.98GB numa placa de 8.0 -> DERRAMA -> 37.4s (e trava o PC)
        tile 128  -> pico 5.78GB                   -> cabe    -> 11.4s
        UNet fora -> pico 3.20GB                   -> 4.8s, mas +4.8GB de RAM

    Tile menor e' 3,3x MAIS rapido que o maior — nao por eficiencia, mas porque o
    grande estoura e o driver passa a arrastar peso pelo PCIe. E tirar a UNet so
    compra mais 6,6s ao preco de 4,8GB de RAM anonima: mau negocio quando a memoria
    ja esta apertada. Offload fica opcional, p/ quem tem RAM sobrando.
    """
    if target_w <= img.width:
        return img.resize((target_w, round(img.height * target_w / img.width)), Image.LANCZOS)
    if offload and eng is not None:
        eng.unet_offload(True)
    try:
        big = esrgan(img, model_path, tile=tile, yield_ms=yield_ms)
    finally:
        if offload and eng is not None:
            eng.unet_offload(False)
    return big.resize((target_w, round(img.height * target_w / img.width)), Image.LANCZOS)


def ultra(eng: GSMDE, img: Image.Image, prompt: str, denoise=0.3, steps=18,
          tile=768, overlap=0.25, seed=1234, gw=0.25, cb=None) -> Image.Image:
    """Ultra: SINTESE DE DETALHE em tiles, nao upscale.

    A imagem inteira em alta resolucao nao cabe nos 8GB — e mesmo que coubesse, o
    modelo desenha textura fina melhor quando o pedaco ocupa a resolucao nativa.
    Entao picota com sobreposicao e roda img2img por pedaco.

    ANTI-ALUCINACAO DE GRACA
    O problema classico do ultra e' o pedaco perder o contexto: um tile so de
    arvores, com o prompt '1girl walking a dog', ganha uma garota no meio do mato.
    A solucao usual e' rodar um tagger (WD14) por tile e injetar as tags.
    Aqui nao precisa: as MASCARAS da Secao 7.2 ja dizem de quem e' cada regiao.
    Num tile dentro do territorio do 'scenary', o 'person' simplesmente nao entra —
    o mecanismo que faz o multi-centro funcionar tambem contem a alucinacao, sem
    modelo extra e sem custo.
    """
    import torch
    W, H = img.size
    passo = max(64, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, W - tile + 1), passo)) or [0]
    ys = list(range(0, max(1, H - tile + 1), passo)) or [0]
    if xs[-1] + tile < W: xs.append(W - tile)
    if ys[-1] + tile < H: ys.append(H - tile)
    total = len(xs) * len(ys)
    print(f"[ultra] {W}x{H} -> {total} tiles de {tile}px (sobrep. {overlap:.0%}), "
          f"denoise={denoise}", flush=True)

    # mascara da imagem inteira, p/ saber quem manda em cada tile
    dono = None
    try:
        lat = eng.encode_image(img.resize((W // 8 * 8, H // 8 * 8), Image.LANCZOS))
        t0 = eng.pipe.scheduler.timesteps[0] if len(getattr(eng.pipe.scheduler, "timesteps", [])) \
            else torch.tensor(500)
        add_time = torch.tensor([[H, W, 0, 0, H, W]], device=eng.dev, dtype=torch.float16)
        dono = eng._masks(lat, t0, lat.shape[2], lat.shape[3], add_time)
    except Exception as e:
        print(f"[ultra] sem mascara global ({type(e).__name__}) — todos os centros em todo tile", flush=True)

    res = img.copy()
    feito = 0
    for y in ys:
        for x in xs:
            feito += 1
            if cb: cb(feito, total, tile)
            peca = res.crop((x, y, x + tile, y + tile))
            regional_orig = eng.regional
            if dono:                       # so quem tem territorio neste tile entra
                fx0, fy0 = x / W, y / H
                fx1, fy1 = (x + tile) / W, (y + tile) / H
                ativos = []
                for nome, palavras in regional_orig:
                    m = dono.get(nome)
                    if m is None:
                        ativos.append((nome, palavras)); continue
                    mh, mw = m.shape
                    sub = m[int(fy0 * mh):max(int(fy1 * mh), int(fy0 * mh) + 1),
                            int(fx0 * mw):max(int(fx1 * mw), int(fx0 * mw) + 1)]
                    if sub.numel() and float(sub.mean()) > 0.22:   # manda neste pedaco?
                        ativos.append((nome, palavras))
                eng.regional = ativos or regional_orig
            try:
                nova = eng.denoise(init=peca, steps=steps, seed=seed + feito,
                                   strength=denoise, mask_every=99, global_weight=gw,
                                   progress=False)
            finally:
                eng.regional = regional_orig
            # cola com feather nas bordas internas (as de fora ficam retas)
            m = Image.new("L", (tile, tile), 255)
            fw = max(4, int(tile * overlap * 0.5))
            px = Image.new("L", (tile, tile), 0)
            px.paste(255, (fw if x > 0 else 0, fw if y > 0 else 0,
                           tile - (fw if x + tile < W else 0),
                           tile - (fw if y + tile < H else 0)))
            res.paste(nova, (x, y), px.filter(ImageFilter.GaussianBlur(fw / 2)))
    return res


# ---------------------------------------------------------------------------
# Detailer: quatro correcoes medidas na bancada de 17/08 (docs/detailer.jsonl)
# ---------------------------------------------------------------------------
# A versao anterior tratava cada deteccao como um i2i independente e colava o
# RECORTE INTEIRO com feather retangular. Isso produzia tres defeitos, todos
# observados em imagem e nenhum deles visivel nas metricas (a variancia do
# laplaciano ficou em 698,5 -> 690,3, indiferente ao estrago):
#
#   1. FALSO POSITIVO virava conteudo. O detector de olhos marcou uma LOUSA de
#      32x166 px na borda e o adaptador de olhos a encheu de rabiscos.
#   2. UM OLHO AZUL, o outro castanho. O passe de rosto redesenhava os dois; o
#      passe de olhos refazia so' um, sem ver o que o outro tinha virado.
#   3. O QUE NAO E' ALVO era redesenhado junto: colando o recorte inteiro, o
#      nariz entre os olhos entrava na conta.
#
# As correcoes seguem o desenho do detailer do SD.Next (modules/postprocess/
# yolo.py): `merge()` usa a UNIAO das caixas como recorte e o MAXIMO das
# mascaras individuais como mascara — recorte com contexto, mascara com buracos.
FORMA_MAX = 5.0            # razao maxima largura/altura de uma deteccao plausivel

# Regras por CLASSE de alvo. `dentro` encadeia detectores (olho fora de rosto nao
# e' olho); `largo` exige largura > altura (olho aberto e' largo — foi o filtro que
# faltou quando uma caixa de 12x40 no queixo passou como olho); `par` recusa
# tratar um alvo sozinho quando ele so' existe em par.
REGRAS = {
    "olho": {"dentro": "rosto", "largo": True, "par": True, "conf": 0.50},
}

# A UI lista o NOME DO ARQUIVO ('Anzhc_eyes', 'face_yolov9c', 'hand_yolov8s'), nao
# uma classe. Casar a regra por chave exata falharia em silencio — o filtro nao
# rodaria e o falso positivo voltaria. Dai a classificacao por substring.
_CLASSES = (
    ("olho", ("eye", "olho", "iris")),
    ("rosto", ("face", "rosto", "head", "cabeca")),
    ("mao", ("hand", "mao", "maos")),
    ("pessoa", ("person", "pessoa", "body")),
)


def classe_do_alvo(kind: str) -> str:
    """'Anzhc_eyes' -> 'olho'. Alvo desconhecido devolve ele mesmo (sem regra)."""
    k = (kind or "").lower()
    for nome, chaves in _CLASSES:
        if any(c in k for c in chaves):
            return nome
    return k


# Centro que atende cada classe no modo AUTOMATICO. Sao os que a bancada de 17/08
# exercitou de fato — 'person' e 'maos' foram treinados no projeto; de olhos nao ha
# centro treinado, entao entra o da biblioteca. Ordem = preferencia: o primeiro que
# existir na instalacao ganha.
CENTRO_AUTO = {
    "rosto": ["person", "pele", "expression"],
    "olho": ["detailedeyes_v3", "enchantingeyesillustrious", "loraeyes_v1"],
    "mao": ["maos", "hands_v2_1"],
    "pessoa": ["person", "pose"],
}


def centro_automatico(kind: str, disponiveis=None) -> str | None:
    """Melhor centro para este alvo, ou None se nenhum candidato existir.

    None NAO e' erro: significa 'refaz a regiao com os centros que ja estao na
    cena', que e' o comportamento historico do detailer.
    """
    for c in CENTRO_AUTO.get(classe_do_alvo(kind), []):
        if disponiveis is None or c in disponiveis:
            return c
    return None


def _forma_ok(cx, regra):
    x0, y0, x1, y1 = cx
    w, h = x1 - x0, y1 - y0
    if w <= 1 or h <= 1 or max(w / h, h / w) > FORMA_MAX:
        return False
    return not (regra.get("largo") and w < h)


def _dentro(cx, caixas):
    ccx, ccy = (cx[0] + cx[2]) / 2, (cx[1] + cx[3]) / 2
    return any(a <= ccx <= c and b <= ccy <= d for a, b, c, d in caixas)


def _recorte(cx, W, H, pad, lado_min):
    """Caixa -> recorte QUADRADO com folga e lado minimo.

    Quadrado porque o SDXL nao tem bucket para 8:23 (a caixa crua de um olho).
    Lado minimo porque ampliar 44x um olho de 23px nao recupera detalhe: inventa.
    """
    x0, y0, x1, y1 = cx
    lado = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    lado = max(min(max(lado, lado_min), min(W, H)), 8)
    a, cx_, cy_ = lado / 2, (x0 + x1) / 2, (y0 + y1) / 2
    cx_ = min(max(cx_, a), W - a)
    cy_ = min(max(cy_, a), H - a)
    return (int(cx_ - a), int(cy_ - a), int(cx_ + a), int(cy_ + a))


def _agrupa(caixas, W, H, pad, lado_min):
    """Junta caixas cujos RECORTES se sobreporiam: um i2i para todas elas.

    Testar as caixas cruas com folga proporcional nao funciona em alvo pequeno —
    dois olhos de 20x9 recebem 7px de folga e nunca se tocam. Ja' os RECORTES
    desses mesmos olhos sao quadrados de 256px praticamente sobrepostos.

    Uma passada para o par e' o que impede o olho azul: as duas iris nascem no
    mesmo forward, vendo uma a' outra.
    """
    grupos = [[c] for c in caixas]
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                ri = [_recorte(c, W, H, pad, lado_min) for c in grupos[i]]
                rj = [_recorte(c, W, H, pad, lado_min) for c in grupos[j]]
                if any(not (p[2] < q[0] or q[2] < p[0] or p[3] < q[1] or q[3] < p[1])
                       for p in ri for q in rj):
                    grupos[i] += grupos[j]
                    del grupos[j]
                    mudou = True
                    break
            if mudou:
                break
    saida = []
    for g in grupos:
        xs = [c[0] for c in g] + [c[2] for c in g]
        ys = [c[1] for c in g] + [c[3] for c in g]
        saida.append(((min(xs), min(ys), max(xs), max(ys)), g))
    return saida


def _mascara(caixas, tam, desloc=(0, 0), escala=1.0):
    """Um blob por caixa — o vao entre eles fica de fora e nao e' redesenhado.

    O esfumado sai do tamanho do MENOR blob, nao da ampliacao: escalar o desfoque
    pela ampliacao dava 48px sobre blobs de 80px, e ai a mascara media 0,73 tanto
    no olho quanto no nariz — tinha deixado de discriminar.
    """
    m = Image.new("L", tam, 0)
    d = ImageDraw.Draw(m)
    dx, dy = desloc
    esc = [((x0 - dx) * escala, (y0 - dy) * escala,
            (x1 - dx) * escala, (y1 - dy) * escala) for x0, y0, x1, y1 in caixas]
    folga = max(2.0, min(32.0, 0.25 * min(min(c[2] - c[0], c[3] - c[1]) for c in esc)))
    for a, b, c, e in esc:
        d.ellipse([a - folga, b - folga, c + folga, e + folga], fill=255)
    return m.filter(ImageFilter.GaussianBlur(folga))


def detail(eng: GSMDE, img: Image.Image, kinds, conf=0.35, denoise=0.35,
           steps=20, crop=768, pad=0.30, seed=1234, lado_min=256,
           mascarado=True, par=True, centros=None) -> Image.Image:
    """YOLO detecta -> agrupa -> i2i MASCARADO no recorte ampliado -> costura.

    centros: {alvo: nome_do_centro | "auto" | "" }. Este e' o GSMDE trabalhando
        como detailer de fato: cada regiao e' refeita pelo ESPECIALISTA dela, e
        nao pelos centros que por acaso ficaram carregados da cena inteira.
        "auto" resolve pela tabela CENTRO_AUTO (rosto->person, olho->
        detailedeyes_v3, mao->maos), "" ou ausente mantem os centros da cena.

        O centro precisa ja estar carregado no motor: trocar `eng.regional` so
        escolhe entre os adaptadores que existem, nao carrega peso novo. Pedir um
        centro ausente cai no comportamento da cena, com aviso.

    mascarado=False volta ao comportamento antigo (recorte inteiro redesenhado),
    para comparacao — nao para uso.
    """
    from ultralytics import YOLO as Y
    import numpy as np
    import torch

    res = img.copy()
    W, H = res.size
    achadas = {}
    centros = centros or {}
    # os adaptadores que o motor tem: so' entre estes da' para escolher
    carregados = set(getattr(eng, "paged", {}) or {})
    reg_cena = list(eng.regional)
    for kind in kinds:
        mp = yolo_path(kind)
        if not mp:
            print(f"[detailer] modelo '{kind}' ausente, pulando", flush=True)
            continue
        cls = classe_do_alvo(kind)
        regra = REGRAS.get(cls, {})
        det = Y(mp)
        boxes = det.predict(res, conf=float(regra.get("conf", conf)), verbose=False)[0].boxes
        cru = [tuple(int(round(v)) for v in b) for b in
               (boxes.xyxy.cpu().tolist() if boxes is not None and len(boxes) else [])]

        bons, fora = [], 0
        for cx in cru:
            if not _forma_ok(cx, regra):
                fora += 1
                continue
            ref = regra.get("dentro")
            if ref and not _dentro(cx, achadas.get(ref, [])):
                fora += 1
                continue
            bons.append(cx)

        # PAR: um olho sozinho e' pior que nenhum — repintar so' um deixa as duas
        # iris de cores diferentes (medido). Tenta achar o irmao com limiar baixo
        # dentro do proprio rosto; nao achando, desiste do alvo inteiro, porque o
        # passe de rosto ja' entrega olhos bons.
        if par and regra.get("par") and regra.get("dentro"):
            for pai in achadas.get(regra["dentro"], []):
                deste = [c for c in bons if _dentro(c, [pai])]
                if len(deste) != 1:
                    continue
                try:
                    r2 = det.predict(res.crop(pai), conf=0.15, verbose=False)[0].boxes
                    extra = [(int(b[0]) + pai[0], int(b[1]) + pai[1],
                              int(b[2]) + pai[0], int(b[3]) + pai[1])
                             for b in (r2.xyxy.cpu().tolist() if r2 is not None else [])]
                except Exception:
                    extra = []
                novas = [c for c in extra if _forma_ok(c, regra) and _dentro(c, [pai])
                         and all(max(abs(c[0] - d[0]), abs(c[1] - d[1])) > 4 for d in deste)]
                if novas:
                    irmao = max(novas, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
                    bons.append(irmao)
                    print(f"[detailer] {kind}: par recuperado em {list(irmao)}", flush=True)
                else:
                    for c in deste:
                        bons.remove(c)
                    print(f"[detailer] {kind}: alvo sem par -> pulado "
                          f"(repintar so' um deixa as duas iris diferentes)", flush=True)

        achadas[cls] = bons
        grupos = _agrupa(bons, W, H, pad, lado_min) if bons else []
        print(f"[detailer] {kind} ({cls}): {len(cru)} bruta(s), {fora} descartada(s), "
              f"{len(bons)} valida(s) em {len(grupos)} passada(s)", flush=True)

        # ---- centro deste alvo ------------------------------------------------
        pedido = str(centros.get(kind) or centros.get(cls) or "").strip()
        alvo_centro = None
        if pedido == "auto":
            alvo_centro = centro_automatico(kind, carregados or None)
            if not alvo_centro:
                print(f"[detailer] {kind}: sem centro automatico disponivel "
                      f"-> usa os centros da cena", flush=True)
        elif pedido:
            if pedido in carregados or not carregados:
                alvo_centro = pedido
            else:
                print(f"[detailer] {kind}: centro '{pedido}' nao esta carregado "
                      f"-> usa os centros da cena", flush=True)
        if alvo_centro:
            print(f"[detailer] {kind}: centro '{alvo_centro}'", flush=True)

        for uniao, membros in grupos:
            cxa = _recorte(uniao, W, H, pad, lado_min)
            ow = cxa[2] - cxa[0]
            if ow < 32:
                continue
            # AMPLIA para gerar, REDUZ para costurar: o centro trabalha perto da
            # resolucao em que foi treinado, e nao num pedaco de 80px da grade.
            alvo = max(64, int(min(crop, ow * 4.0)) // 8 * 8)
            peca = res.crop(cxa).resize((alvo, alvo), Image.LANCZOS)
            kw = {}
            if mascarado:
                mpx = _mascara(membros, (alvo, alvo), desloc=(cxa[0], cxa[1]),
                               escala=alvo / ow)
                kw["mask_lat"] = torch.from_numpy(
                    np.asarray(mpx.resize((alvo // 8, alvo // 8), Image.BILINEAR),
                               np.float32) / 255.0
                ).to(eng.dev, torch.float16)[None, None]
                kw["lat_ctx"] = eng.encode_image(peca)
            # `try/finally`: se a passada estourar, a cena NAO pode continuar com
            # o centro do detailer no lugar dos dela — as etapas seguintes
            # (hires, ultra) sairiam com o especialista errado, sem erro visivel.
            masc_cena = getattr(eng, "usar_mascaras", True)
            if alvo_centro:
                eng.regional = [(alvo_centro, [])]
                # Sonda DESLIGADA: ela procura as ancoras de texto do centro no
                # prompt, e `set_prompt` foi chamado para os centros da CENA, nao
                # para este. Alem disso o recorte JA E' o territorio — nao ha o
                # que disputar —, e pular a sonda tira uma passada de UNet por
                # passo (2 em vez de 3).
                eng.usar_mascaras = False
            try:
                nova = eng.denoise(init=peca, steps=steps, seed=seed,
                                   strength=denoise, mask_every=8,
                                   **kw).resize((ow, ow), Image.LANCZOS)
            finally:
                if alvo_centro:
                    eng.regional = reg_cena
                    eng.usar_mascaras = masc_cena
            cheio = res.copy()
            cheio.paste(nova, (cxa[0], cxa[1]))
            res = Image.composite(cheio, res, _mascara(membros, res.size))
        del det
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", required=True)
    ap.add_argument("--globals", default="")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--from-image", default=None, help="pula o gerador e refina esta imagem")
    ap.add_argument("--base-size", type=int, default=768)
    ap.add_argument("--layers", default="1152:0.45:30,1536:0.28:40",
                    help="largura:denoise:steps por camada, virgula")
    ap.add_argument("--upscaler", default="anime", choices=list(UPS))
    ap.add_argument("--tile", type=int, default=128,
                    help="tile do ESRGAN. 128 cabe nos 8GB c/ a UNet residente; "
                         "256 derrama e fica 3x mais lento")
    ap.add_argument("--upscale-offload", action="store_true",
                    help="tira a UNet da placa no upscale: +2.4x no upscale, +4.8GB de RAM")
    ap.add_argument("--detailer", default="face,hand", help="'' desliga")
    ap.add_argument("--detailer-denoise", type=float, default=0.35)
    ap.add_argument("--adapter-scale", type=float, default=0.8)
    ap.add_argument("--global-weight", type=float, default=0.25)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--steps", type=int, default=26)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--yield-ms", type=int, default=40)
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--negative", default=None, help="vazio = negativo padrao")
    ap.add_argument("--out-dir", default=str(OUT), help="onde salvar (a UI usa a sua)")
    args = ap.parse_args()
    # mesmas pastas do worker: 4 imagens quase iguais por rodada tornam a pasta
    # unica inutilizavel. 'final' recebe uma; o resto vai p/ 'intermed'.
    out_dir = Path(args.out_dir) / "intermed"
    out_dir.mkdir(parents=True, exist_ok=True)
    fim_dir = Path(args.out_dir) / "final"
    fim_dir.mkdir(parents=True, exist_ok=True)
    ultimo = {}
    t0 = time.time()

    eng = GSMDE(parse_centers(args.centers),
                [g.strip() for g in args.globals.split(",") if g.strip()],
                adapter_scale=args.adapter_scale, cfg=args.cfg, yield_ms=args.yield_ms)
    eng.set_prompt(args.prompt, args.negative or NEG)

    def save(im, name):
        p = out_dir / f"gsmde_{args.tag}_{name}_s{args.seed}.png"
        im.save(p)
        ultimo["im"], ultimo["nome"] = im, name
        print(f"[refino] salvo {p.name}  ({im.width}x{im.height})  "
              f"{time.time()-t0:.0f}s", flush=True)

    # 1) gerador
    if args.from_image:
        img = Image.open(args.from_image).convert("RGB")
        print(f"[refino] partindo de {args.from_image} ({img.width}x{img.height})", flush=True)
    else:
        print("[refino] camada 0: gerador GSMDE", flush=True)
        img = eng.denoise(size=args.base_size, steps=args.steps, seed=args.seed,
                          global_weight=args.global_weight)
        save(img, "l0gen")

    # 2+3) upscaler + hires fix, camada por camada
    for i, spec in enumerate([s for s in (args.layers or "").split(",") if s.strip()], 1):
        w, dn, st = spec.split(":")
        w, dn, st = int(w), float(dn), int(st)
        print(f"[refino] camada {i}: upscale -> {w}px ({args.upscaler}), "
              f"hires denoise={dn} steps={st}", flush=True)
        img = upscale_to(img, w, UPS[args.upscaler], eng=eng, yield_ms=args.yield_ms,
                         tile=args.tile, offload=args.upscale_offload)
        img = eng.denoise(init=img, steps=st, seed=args.seed, strength=dn,
                          global_weight=args.global_weight)
        save(img, f"l{i}hires{w}")

    # 4) detailer
    kinds = [k.strip() for k in (args.detailer or "").split(",") if k.strip()]
    if kinds:
        print(f"[refino] detailer: {kinds}", flush=True)
        img = detail(eng, img, kinds, denoise=args.detailer_denoise, seed=args.seed)
        save(img, "l9detail")
    if ultimo.get("im") is not None:      # a ultima etapa que rodou E' a final
        f = fim_dir / f"gsmde_{args.tag}_s{args.seed}.png"
        ultimo["im"].save(f)
        print(f"[refino] FINAL ({ultimo['nome']}) -> {f}", flush=True)
    print(f"[refino] pronto em {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
