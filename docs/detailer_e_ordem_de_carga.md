# Centros como detailer, e quatro defeitos que só a imagem denunciou

Bancada de 16–17/08/2026. Tudo a 1024×1024, seed 3850638904, CFG 6,5, 35 passos,
offload `sequential`. Dados brutos em `docs/completo.jsonl`, `docs/qualidade.jsonl`
e `docs/detailer.jsonl` da árvore de trabalho.

## 1. A pergunta

Os centros do GSMDE são LoRAs ativados um por vez: cada um faz uma passada
completa de UNet e produz seu próprio ε sobre o latente inteiro, combinados
depois por máscara. Com 6 centros são 8 passadas por passo (6 + base + sonda).

A alternativa proposta: gerar **só com o backbone** — sem centro, sem LoRA — e
depois usar cada centro como *detailer*, onde um detector YOLO acha a região
(rosto, olhos, mãos) e o centro correspondente a refaz por img2img.

## 2. O resultado

| | 6 centros no laço | detailer |
|---|---|---|
| tempo total | ~990 s | **300 s** |
| despejo | 2,98 GB | **0,42 GB** |
| RAM | 16,8 GB | 15,7 GB |

**3,3× mais rápido e 7× menos despejo.** A geração principal passa a custar 1
passada por passo em vez de 8, e nenhum nicho precisa estar residente durante ela.

Por backbone, normalizado por região tratada:

| | original | d5u10 | d1u10 |
|---|---|---|---|
| base, s/passo | 3,16 | 2,60 | 2,30 |
| s por região | 93,6 | 72,6 | 63,1 |
| VRAM pico | 5,80 GB | 4,89 GB | 3,91 GB |
| RAM | 15,69 GB | 12,67 GB | 10,27 GB |
| nitidez da base | 698,5 | 618,2 | 498,6 |

O `d1u10` é o mais rápido e leve em tudo, mas perde 29% de nitidez — e isso
aparece a olho nu no fundo da imagem, não é só número.

## 3. Os quatro defeitos, e por que nenhuma métrica os pegou

Em todos os casos abaixo a variância do laplaciano ficou entre 690 e 698 —
indiferente ao estrago. Quem pegou foi a inspeção visual seguida de um diff de
pixels para localizar a mudança.

### 3.1 Falso positivo virava conteúdo

O detector de olhos a `conf=0.30` marcou uma **lousa de 32×166 px** na borda da
imagem. O adaptador de olhos rodou ali e a encheu de rabiscos.

Três filtros, sendo o terceiro o decisivo:
- razão de aspecto máxima (32×166 tem razão 5,2);
- olho precisa ser mais **largo** que alto (uma caixa de 12×40 no queixo passou
  pelo filtro simétrico e só foi barrada por este);
- **olho só vale dentro de um rosto detectado** — encadeamento de detectores.

### 3.2 Uma íris azul, a outra castanha

Cada região era um img2img independente. O passe de rosto redesenhava os dois
olhos; o passe de olhos refazia só um, sem ver o que o outro tinha virado.

Correção em duas partes, e a segunda é a que resolve:

- **Recorte é a união, máscara é a soma dos blobs.** Mesmo desenho do
  `merge()` do detailer do SD.Next (`modules/postprocess/yolo.py`): o recorte
  dá contexto compartilhado, a máscara tem buraco entre os alvos.
- **Regra do par:** olho só é refeito junto com o outro. Achando um só, tenta o
  irmão com limiar baixo dentro do rosto; não achando, pula o alvo — o passe de
  rosto já entrega olhos bons.

Controle que confirma o mecanismo, mesmo código variando só isto:

| backbone | olhos detectados | tratamento | resultado |
|---|---|---|---|
| original | 1 | pintado sozinho | cores diferentes |
| d5u10 | 2 | os dois numa passada | mesma cor |

### 3.3 O esfumado apagava a máscara

Primeira tentativa escalou o desfoque pela ampliação: 48 px sobre blobs de 80 px.
Medido 0,73 no ponto **entre** os olhos contra 0,73 no centro deles — a máscara
tinha deixado de discriminar, e o nariz seria redesenhado igual. O desfoque agora
sai de uma fração do menor blob: 0,00 no nariz, 1,00 nos olhos.

### 3.4 Ampliação sem teto

Uma caixa de olho tem ~23 px. Ampliar até 1024 é 44× — o modelo não recupera
detalhe nessa escala, ele inventa. Recorte mínimo de 256 px e teto de 4×.

Custo, para dimensionar: **o preço da passada vem do alvo de ampliação, não do
tamanho da região.** Um olho e um rosto inteiro custam o mesmo se ambos forem
gerados a 1024 (88 s contra 95 s medidos).

## 4. Consertos no motor, encontrados no caminho

### 4.1 A seed não determinava a imagem

`gen` semeado era usado só no latente inicial; as chamadas
`pipe.scheduler.step(...)` não recebiam `generator=`. O EulerAncestral injeta
ruído a cada passo e, sem generator, puxa do RNG global.

Duas corridas com a mesma seed e config davam nitidez **346,2 e 201,5**. Depois
do conserto: **diferença máxima de 0 pixels, imagem bit a bit idêntica.**

Consequência retroativa: toda comparação A/B feita antes disto carrega ruído de
corrida a corrida que chegou a 40% na métrica de nitidez.

### 4.2 Despejo de 2,98 GB na carga

`pipe.unet.to(dev)` arrasta os submódulos, e os nichos já estão pendurados na
árvore da UNet: a placa via 4,78 (base) + 3,81 (8 nichos) = **8,59 GB** de uma
vez, numa placa de 8,00. A estratégia documentada no motor — carregar com a UNet
na CPU para que "a placa nunca veja os dois juntos" — tinha esse furo.

O laço em si estava limpo (0,10 GB de despejo, pico de 0,39 GB). Paginar os
nichos não adiantava: a política roda depois do `.to()`, tarde demais.

Desanexando os nichos antes de subir a base: pico **8,59 → 6,51 GB**, despejo
**2,98 → 1,04 GB**. Não zerou; sobra a janela em que `aplica_offload_base` traz
todos os nichos de volta.

### 4.3 Dois gerentes de memória na mesma VRAM

`_revisa_residencia` re-executava a política AUTO a cada mudança de área — e
religava a paginação de nicho que o offload do accelerate tinha desligado de
propósito, inclusive entre as camadas do hires.

### 4.4 Aviso de pico gritando lobo

A referência de 5,5 GB foi medida com o backbone residente. Sob offload ele não
mora na placa, e o aviso virava alarme falso a 1024 — resolução que, medida,
roda sem problema nessa configuração.

## 5. Carona custa mais que ser centro

Um "carona" é um adaptador empilhado na passada de outro centro em vez de ter
passada própria. A premissa era que isso fosse mais barato.

Medido: 6 centros custam 26,52 s/passo; 6 centros + 2 caronas custam
**40,86 s/passo** — +54% para *remover* duas passadas. Com offload e paginação,
cada passada passa a mover 3 blocos de adaptador em vez de 1, vezes 6 centros,
vezes cada passo.

Estimando pelo custo por passada (26,52/8 ≈ 3,3 s), pôr os dois como centros
próprios daria ~33 s/passo — cerca de 19% **mais rápido** que como carona.
Estimativa, não medida: falta uma condição com 8 centros para confirmar.

## 6. Nota sobre o método

Três vezes nesta bancada uma métrica de *quantidade* foi usada para julgar
*identidade* ou *localização*, e errou nas três. Diferença média de pixel e
variância do laplaciano dizem quanto mudou, não o quê nem onde. O procedimento
que funcionou foi: olhar a imagem, e então usar um diff com componentes conexos
para localizar exatamente a região alterada e conferir se ela bate com o alvo
pretendido. Foi assim que a lousa apareceu.
