# Centros como modelos: o que funciona, o que não, e o caminho que sobra

Pesquisa pedida em 2026-07-28, escrita com os números medidos na mesma sessão.
Três perguntas: (1) dá para transformar os LoRAs rank-256 em modelos próprios?
(2) dá para fazê-los trabalhar **em camadas** em vez de simultaneamente?
(3) o MoE de dois modelos da mesma linguagem (estilo SD3) se aplica aqui?

---

## 1. Transformar os LoRAs em modelos: por que não encolhe

Os centros são **PEFT LoRA rank 256** sobre `to_q/k/v/out` — 1120 chaves, 0,69 GB
cada. LoRA é, por definição, um **delta**:

```
W_efetivo = W_base + (B · A) · escala
```

Sem `W_base`, o produto `B·A` é uma matriz de correção de posto 256 — uma direção
de ajuste, não um modelo. Então o backbone não é opcional; é o que o delta corrige.

Mesclar funciona (medido: um LoRA solo a 0.8 dá `|delta| 1.4e-4`, coerente), mas o
resultado **tem o tamanho da base**, porque o delta foi absorvido em `W`:

| | disco | VRAM por modelo |
|---|---|---|
| hoje (backbone + 36 deltas) | ~30 GB | 5,2 GB (um backbone) |
| 36 modelos mesclados | ~187 GB | 5,2 GB **cada** |

**O backbone é a deduplicação, não o custo.** Ele é a parte que, sem ele, seria
copiada 36 vezes. Qualquer esquema de "centros independentes" paga o substrato
compartilhado N vezes.

E há uma perda pior que a de espaço: hoje **6 centros colaboram numa imagem**, cada
um pintando sua região. Modelos independentes dariam um domínio por imagem — perde-se
a mistura, que é o que o GSMDE é.

> Conclusão: mesclar + destilar em modelos pequenos de domínio é possível, mas custa
> N treinos, N× o disco, e mata a composição. Serve para um caso fixo (um personagem
> que se usa muito), não para substituir a arquitetura.

---

## 2. Trabalhar em camadas: o caminho promissor

Aqui está o achado que muda o quadro. Medido nesta sessão:

```
custo por centro ........ 0,43 – 0,49 s/passo
forward do backbone ..... 0,38 – 0,49 s/passo
```

São o mesmo número, porque **é a mesma coisa**: a arquitetura roda **um forward
completo por centro**. Com N centros são **N+1 passadas pelo UNet a cada passo**.

Isso foi confirmado por eliminação: deixar centros residentes na VRAM (zero
transferências) não acelerou nada — 7 residentes de 10 renderam 0,8%; com 2 centros,
residente chegou a ser 10% **pior** que paginado. O gargalo nunca foi trânsito.

### A proposta do usuário, formalizada

Em vez de N forwards (um por centro, combinados por máscara de região), **um único
forward** em que o centro ativo **muda por camada**:

```
hoje:     forward(backbone) + forward(c1) + ... + forward(cN)  -> combina por região
proposta: forward único, trocando o adapter ativo a cada bloco do UNet
```

O ganho seria de `N+1` para `1` — não incremental, estrutural.

### O que precisa ser resolvido antes

**(a) A semântica de composição muda.** Hoje cada centro prediz a imagem inteira e
a máscara escolhe onde ele vale. Na versão por camadas, o centro influencia toda a
imagem, mas só em certas profundidades. Não é a mesma operação, e não há garantia
a priori de que preserve "cada centro pinta seu território".

Há uma pista favorável nos dados desta sessão: a bateria de LoRA parcial mostrou que
**zerar 57% das chaves de um centro não degradou a imagem** — o conhecimento é
redundante entre blocos. Se um centro sobrevive perdendo mais da metade das camadas,
usá-lo em apenas algumas talvez baste.

**(b) Qual centro em qual profundidade.** A distribuição das chaves sugere ordem
natural: `down_blocks` (estrutura) → `mid_block` (semântica) → `up_blocks`
(acabamento). Centros de cena caberiam embaixo, de personagem no meio, de textura em
cima. É testável sem treinar nada.

**(c) Custo de troca.** Ativar/desativar adapter por bloco tem custo próprio
(`enable/disable adapters` aparece no perfil). Com ~70 blocos e trocas por grupo,
seriam ~5 trocas por forward em vez de N forwards — provavelmente muito mais barato,
mas precisa ser medido.

### Teste barato que decide

Rodar uma imagem com o centro A ativo só em `down_blocks` e o centro B só em
`up_blocks`, comparando com a composição atual. Não exige treino, só a troca do
processador de atenção por bloco. **É o próximo experimento de maior valor.**

---

## 3. MoE estilo SD3: a analogia não é a que parece

O SD3 usa **MMDiT**: dois fluxos (texto e imagem) com pesos separados que se
encontram na atenção conjunta. Não é troca de especialistas — os dois fluxos rodam
**sempre**, não alternam. A analogia com "dois centros se revezando" não se sustenta.

O MoE que **de fato** se aplica é o esparso, estilo Mixtral. E aqui há uma
coincidência que vale registrar: no Mixtral **os experts são a FFN**, e todo o resto
(atenção, embeddings) é compartilhado. Medimos independentemente que, no nosso UNet:

```
FFN ............ 1229,5M  47,9%   <- os centros NÃO tocam
atenção ........  955,4M  37,2%   <- os centros vivem aqui
resnets ........  327,9M  12,8%
```

Ou seja: a camada que o Mixtral escolheu para especializar é exatamente a que está
livre no GSMDE, e a que os centros ocupam é justamente a que o Mixtral compartilha.
São arquiteturas **complementares**, não concorrentes:

```
tronco compartilhado = atenção + resnets + embeddings ..... ~2,5 GB (fixo)
experts paginados    = FFN por domínio .................... ~0,6 GB (um por vez)
centros (LoRA)       = atenção por território ............. 0,69 GB (um por vez)
```

Isso permitiria capacidade total distribuída em arquivos, **com um de cada por vez na
placa** — que é literalmente o que o usuário descreveu. Exige treinar as FFN-experts,
mas não quebra os centros, porque não toca na atenção.

---

## 4. Sobre "manter os latentes na memória"

Isso já acontece e não é gargalo. O latente a 1024×576 é `4×72×128` fp16 = **74 KB**.
Mesmo com dezenas deles o custo é irrelevante perto dos 5,2 GB do backbone. Não há
o que otimizar aqui.

---

## 5. Ordem recomendada

1. **Camadas** (seção 2) — teste sem treino, ataca `N+1 → 1`, maior ganho por esforço
2. **FFN-experts** (seção 3) — exige treino, mas é a única via para "capacidade
   distribuída, um por vez na placa" sem quebrar os centros
3. **Centros standalone** (seção 1) — só para casos fixos; não substitui a arquitetura

O que **não** vale a pena, com evidência desta sessão: duplo buffer e fila de centros
(o trânsito não é o gargalo), e os modos de offload do SD.Next (`balanced` travou a
placa — eles pressupõem um forward por passo, e aqui são N+1).
