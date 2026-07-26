# Protocolo de treino dos centros GSMDE

Regras da receita de treino, separadas da lista de quais nichos treinar
(essa fica em `proximo_batch_centros.md`).

## Currículo: 3 passadas cheias antes do teste de convergência

**A partir do próximo batch, o padrão é:**

- **Passadas 1, 2 e 3 = aquecimento.** A fila roda inteira, imagem por imagem,
  e *ninguém é aprovado* — toda imagem volta para o fim da fila
  independentemente do loss. Serve só para o centro ver o dataset completo
  três vezes.
- **A partir da passada 4** começa o teste de convergência: a imagem sai da
  fila quando `loss < 0.01`, e volta para o fim se não passar.

Na linha de comando:

```
--passadas-aquecimento 3 --loss-threshold 0.01 --max-steps 4000 --max-passes 12
```

### `--max-steps` conta a partir do fim do aquecimento

O aquecimento é **custo fixo do dataset**: 3 passadas cheias × n_imagens, sempre,
sem exceção. Não faz sentido ele consumir o orçamento da convergência, então o
`--max-steps` orçamenta **apenas a fase de convergência**:

```
total real = passadas_aquecimento × n_imagens  +  max_steps
```

Com 2.500 imagens e `--max-steps 4000`: 7.500 de aquecimento + 4.000 de
convergência = 11.500 steps. O aquecimento roda inteiro mesmo que `--max-steps`
seja pequeno.

**`--max-passes` NÃO segue essa regra** — ele conta passadas totais. Se for
menor ou igual ao aquecimento, o treino para assim que o aquecimento acaba e a
convergência nunca roda. O trainer avisa em voz alta nesse caso, mas o certo é
deixar `--max-passes` bem acima do aquecimento (ex.: 12 para 3 de aquecimento).

O `train_report.json` passou a separar `steps_aquecimento` de
`steps_convergencia`, para não repetir a leitura ambígua do c6.

### Por que 3 passadas antes, e não julgar desde a primeira

O loss de diffusion é medido num timestep **sorteado a cada step**. Uma mesma
imagem pode dar 0.03 num timestep fácil e 0.25 num difícil, sem que o modelo
tenha mudado. Julgar aprovação numa única amostra aprova imagem por sorte do
sorteio, não por domínio. As três passadas cheias garantem que o julgamento
começa sobre um centro que já viu tudo, e não sobre ruído.

### Tensão conhecida com o orçamento de tempo

São duas regras do projeto que não cabem juntas sem ajuste:

1. ~10h de treino por cluster.
2. Requeue até dominar (`loss < 0.01`).

Com 2.500 imagens por centro, 3 passadas cheias já custam 7.500 steps — e o
teste de convergência só *começa* aí. A `--max-steps` precisa subir junto,
senão o treino para no teto antes do currículo operar. Ver "o que aconteceu
no c6/c7" abaixo.

Além disso, `train_specialist.py` avisa (e o aviso procede) que **0.01 é
praticamente inatingível** em MSE de diffusion: o loss típico fica entre 0.05
e 0.3. Na prática o treino vai parar por `--max-passes`/`--max-steps`, não por
fila vazia. Isso é aceitável desde que seja uma decisão consciente: o limiar
baixo funciona como "nunca aprove, continue moendo até o teto", e o teto vira
o verdadeiro critério de parada.

## O que aconteceu no c6 e no c7 (2026-07-25/26)

Os dois clusters rodaram **antes** desta regra existir, com:

```
--max-steps 4000 --max-passes 3   (threshold auto = quantil 35% da 1a passada)
```

Resultado nos 5 centros do c6: todos pararam por `max_steps` com **1,6 passada**
— a primeira inteira (2.500 steps, que só coleta a distribuição) mais 60% da
segunda. O mecanismo de requeue mal chegou a operar. As ~547 imagens
"aprovadas" de 2.500 vieram de uma passada parcial e de sorteio de timestep
favorável, não de domínio.

Medido: **1,68 s/step**, ~100 min por centro (o `pele`, rank 192, levou 163),
09h05 pelo cluster de 5.

**Custo da regra nova**, a 1,68 s/step medido:

| dataset | aquecimento (3×n) | convergência | total | por centro | cluster de 5 |
|---|---|---|---|---|---|
| 2.500 imgs | 7.500 steps | 4.000 | 11.500 | ~3h20 | ~17h |
| 1.200 imgs | 3.600 steps | 4.000 | 7.600 | ~2h15 | ~11h |

Se o orçamento de 10h/cluster for firme, reduzir o dataset para ~1.200 imagens
mantendo a estratificação por tag (`--cap 1200`) cabe quase exato. A cota por
tag cai de 125 para 60, o que continua bem acima do piso.

Decidir depois de testar os centros do c6 no i2i: se 1,6 passada já entrega
resultado visível, a discussão é acadêmica; se não, temos a explicação pronta.

## Amostragem do dataset

`build_specialist_datasets.py --estratificar` (ver comentários no arquivo).
Cota igual por tag, âncora na tag mais rara da imagem, round-robin na montagem,
teto por assinatura de tags e teto por tar **por tag**. Sem isso a tag dominante
engole o dataset.

## Rank por nicho

Sai da dispersão de assinaturas do próprio nicho, não de um `--rank` global.
Escopo pequeno converge com pouco (`unhas` = 64, ~186MB); escopo largo precisa
de mais (`bijuteria`/`pele` = 192, ~557MB).

## Fila da GPU (anti-TDR)

No ROCm/Windows o contador `Compute` não tem instância: **tudo despacha na fila
3D**, a mesma do compositor. Sem limite a fila cresce, o 3D crava 100%, a
máquina congela e o watchdog TDR reinicia o driver, derrubando o treino.
`--gpu-pausa 0.01` sincroniza ao fim de cada step (fila de 1 step) e devolve uma
fatia ao dwm. Medido no c6: 3D entre 3% e 25%, zero eventos de TDR em 9h.
