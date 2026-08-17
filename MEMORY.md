# MEMORY.md — estado do projeto e continuidade

Documento de handoff. Escrito em **2026-07-26**, ao fim de uma sessão longa, para
que outra conta/assistente consiga continuar sem redescobrir o que já foi medido.

Regra que vale para quem continuar: **quase tudo aqui foi medido, não presumido.**
Onde houver número, ele saiu de execução real nesta máquina. Onde houver dúvida,
está marcado como dúvida. Não troque medição por heurística.

---

---

# Relatório da bancada — CONCLUÍDO em 2026-08-12

54 medições (9 topologias × 3 cenas × com/sem LoRA), 1024×1024. Entregue em
`D:/GSMDE/docs`: `relatorio.md`, `ambiente_e_metodo.md`, 11 gráficos em `graficos/`,
`bench.jsonl`, `vram_historico.jsonl`, 54 imagens em `bench_imagens/`. Os documentos
antigos (`artigo_civitai.md`, `protocolo_treino.md`) foram atualizados.

**Achado principal:** a UNet **original é a única que despeja** (0,30 GB); todas as
reduções ficam em 0,08 GB, o piso. A redução não é conforto — é o que tira a geração
da zona de transbordo em 8 GB. A `d5u10` é 27% mais rápida e usa 1 GB a menos.

**Onde o corte custa:** detalhe de material e objeto pequeno (`down_blocks.2`). Em cena
com pessoa, iguala ou supera a original — inclusive na cena feita para cobrar os 79%
de compatibilidade.

**Não descartar LoRA por causa da razão de vazamento.** A métrica é diferença média de
pixel: confiável para vazamento (coluna "alheias"), fraca para efeito próprio. `mao`
0,6× é provavelmente a métrica cega, não a LoRA morta.

**Toda medição de VRAM anterior a 12/08/2026 é inválida** — lia o `snapshot.json` sem
checar a idade, e ele estava parado havia 11,5 dias.

---

## 1. O que é o GSMDE

**Graph-Structured Mixture of Denoising Experts.** Em vez de um UNet monolítico,
uma prateleira de **centros** (LoRAs especialistas). O roteador lê o prompt, casa
palavras contra o léxico de tags de cada centro, e só os escolhidos sobem para a
VRAM — um por vez, paginados a partir da RAM.

O achado que simplifica o desenho: **a palavra do prompt que casou com a tag já é
a âncora espacial da máscara**. Rotear e mascarar são o mesmo ato.

## 2. Máquina

- **RX 7600, 8 GB**, ROCm nativo no Windows 11, torch 2.10+rocm7.14
- 40 GB de RAM, 16 threads
- Backbone: `D:\Models\Stable-diffusion\IDN_Illustrious_V10_B.safetensors` (SDXL)
- Biblioteca de modelos: `D:\Models` (padrão SD.Next, 569 GB, 628 arquivos)

**Peculiaridade crítica do build:** o torch ROCm vem **sem MKL**. fp32 na CPU cai
num caminho ingênuo e fica **230x** mais lento que o melhor caminho. Nenhuma
tabela de "placa X prefere dtype Y" preveria isso — é propriedade do build.

## 3. Onde cada coisa vive

**Unificado em 2026-07-28.** Antes o projeto estava espalhado por quatro raízes
(`D:\Trainer_v13`, `D:\Trainer_outputs`, `D:\gsmde-studio`, `G:\SD.NEXT`) e trocar
um disco exigia editar ~90 arquivos. Agora a âncora é **uma**: `GSMDE_ROOT`
(variável de ambiente) ou a detecção que sobe do próprio arquivo até achar
`ckpts/` + `trainer/`. Ver `launcher/caminhos.py`.

| o quê | onde |
|---|---|
| **raiz de tudo** | `D:\GSMDE` |
| repositório publicado | `D:\GSMDE\studio` → github.com/NOTcisLol/gsmde-studio |
| motor + pipeline de treino | `D:\GSMDE\trainer` (**não é repo git ainda**) |
| estado de treino (`ckpt.pt`) | `D:\GSMDE\ckpts\models` (35 ckpts — c7 retoma daqui) |
| UI do Studio | `D:\GSMDE\ui` |
| banner / ícones | `D:\GSMDE\assets` |
| launcher pessoal (dev) | `G:\SD.NEXT\launcher` (mistura SD.Next — **não publicar**) |
| atalho de inicialização | `G:\SD.NEXT\GSMDE.bat` |
| centros treinados | `D:\Models\gsmde\specialists` (36 centros, 16,9 GB) |
| LoRAs da biblioteca | `D:\Models\Lora` (389 arquivos) |
| saídas de treino | `D:\GSMDE\outputs\treino` |
| **imagens geradas** | `G:\SD.NEXT\outputs\gsmde` |

Duas coisas ficam **fora** da raiz de propósito:

- **`D:\Models`** — a biblioteca é compartilhada com o SD.Next e segue o padrão
  dele. Unificar o código não implica mover 569 GB de pesos.
- **`G:\SD.NEXT\outputs\gsmde`** — G: é NVMe (D: é SATA): gravar imagem ali é mais
  rápido, e as saídas ficam visíveis junto das do SD.Next.

Os locais antigos foram renomeados para `_OLD_*` em vez de apagados: se algum
caminho escapou do refactor, ele **falha alto** em vez de ler dado obsoleto em
silêncio (a armadilha da §8). `G:\_OLD_Trainer_v13` são 42,7 GB de estado de treino
velho — 30 centros, todos já no canônico, todos os `ckpt.pt` mais antigos que os de
`D:\GSMDE\ckpts`. Pode apagar.

`D:\GSMDE\studio` é uma **cópia** dos arquivos, não um link. Ao editar em
`D:\GSMDE\trainer\forks` ou `G:\SD.NEXT\launcher`, é preciso copiar de novo para o
repo antes de commitar. Isso é dívida técnica conhecida.

## 4. Estado da publicação

| destino | estado |
|---|---|
| **GitHub** | ✅ github.com/NOTcisLol/gsmde-studio — público |
| **HuggingFace** | ✅ huggingface.co/CairoAOGG4/gsmde-centros — 36 centros, 74 arquivos, 16,9 GB (upload em 1h15) |
| **Artigo CivitAI** | 🟡 texto pronto em `docs/artigo_civitai.md`, **não publicado** |
| **Space de demo** | ⬜ não começado |

Identidades: git `connectcairo@gmail.com` / `NOTcisLol`; HF `CairoAOGG4`.

No repo da HF estão os pesos, o model card e o `brain_registry.public.json`
(registro sem caminhos absolutos). Os `ckpt.pt`/`latents.pt` ficaram de fora —
519 MB de estado de treino que não geram nada.

**Título sugerido para o artigo:** "GSMDE: a arquitetura de difusão para placas
fracas". O ângulo escolhido para o CivitAI é o **scan & adapt** — não existe
"modelo GSMDE" para baixar, mas todo LoRA SDXL já é um centro em potencial.

### Armadilhas de publicação (custaram tempo)

- O `--include` do typer é **flag repetida**, não valores separados por espaço.
  Mais simples usar `--exclude`.
- `hf auth login` **sem `--force`** não substitui um token já guardado: responde
  "already logged in" e o token novo nunca entra. Sintoma: 401 em tudo.
- `hf repo` virou `hf repos`.
- Criar o repo no GitHub **com** README/license gera um commit inicial e o push
  local é recusado por não ter ancestral comum.
- PowerShell 5.1 não tem `&&` — use `;`.

**Nunca versionar:** `control.token`, `civitai_auth.bin` (token OAuth cifrado),
`prefs.json`, e `broad_specialists_local.json` (lista os 389 LoRAs pessoais).
O `.gitignore` do repo já cobre todos.

## 5. Estado do treino

Clusters **c6** (texturas) e **c7** (anatomia fina) foram construídos nesta sessão.

- **c6 completo**: `pele` `pelagem` `tecido` `molhado` `pedra` — 9h05 o cluster
- **c7 parcial**: `maos` ✅ `bijuteria` ✅ · `unhas` **em treino** · faltam `pes` `boca` `maquiagem`
- Total: **36 centros treinados**

O treino está **pausado** (supervisor e fila encerrados para testes). `unhas` tem
checkpoint no step 250 e retoma de lá.

```powershell
Start-Process powershell -ArgumentList '-NoExit','-File','D:\GSMDE\trainer\supervisor_clusters.ps1' -WindowStyle Minimized
```

O supervisor encadeia c6→c7 e ressuscita a fila se ela cair. **Mate o supervisor
ANTES da fila** ao parar, senão ele reinicia o que você acabou de matar.

### Protocolo de treino — mudança importante não aplicada ainda

c6 e c7 rodaram com `--max-steps 4000`, o que dá **1,6 passada** pelo dataset de
2.500 imagens: a primeira inteira (que só coleta a distribuição de loss) e 60% da
segunda. **O currículo de requeue mal chegou a operar.**

A regra decidida para o próximo batch, já implementada mas com padrão desligado:

```powershell
--passadas-aquecimento 3 --loss-threshold 0.01 --max-steps 4000 --max-passes 12
```

3 passadas cheias sem aprovar ninguém, e só então o teste de convergência. O
`--max-steps` passou a contar **a partir do fim do aquecimento** (aquecimento é
custo fixo: N × n_imagens). `--max-passes` NÃO segue essa regra — se for ≤ ao
aquecimento, a convergência nunca roda; o trainer avisa. Detalhes em
`docs/protocolo_treino.md`.

## 6. Números medidos (não re-derivar)

| medição | valor |
|---|---|
| geração, 4 centros | 45 → **11 s/it** (evitando spill) |
| política de paginação | auto **10,4 s/passo** · residente 53,9 · paginado 15,0 |
| custo de paginar | 0,12 s/passo/nicho |
| encode do prompt, 6 centros | CPU fp32 **212 s** · CPU bf16 8,1 · CPU fp16 5,3 · **GPU fp16 sequencial 0,92** |
| divergência fp16 vs fp32 | cosseno 1e-05, erro relativo 4e-03 |
| divergência bf16 vs fp32 | cosseno 6e-05, erro relativo **4e-02** (10x pior) |
| cache do negativo por centro | 31 → **16,8 s** (46%) |
| treino | **1,68 s/step**, ~100 min por centro |
| centros | mediana **0,56 GB** · LoRAs da biblioteca 0,14 GB |
| tamanho do UNet SDXL | 2,57 B params, 5,14 GB |
| paginação, 5 centros | módulo a módulo 1,376 s/passo → **bloco contíguo 0,174** (7,9x) |

**fp16 domina bf16 nos dois eixos** (mais rápido E mais preciso — bf16 tem 7 bits
de mantissa contra 10). O padrão da CPU foi trocado de bf16 para fp16.

Na GPU as três precisões custam ~1 s: **ali não há motivo para baixar precisão**.
Resposta à pergunta "numa H100 justifica?" — não, o ganho é de décimos de segundo
numa fase que roda uma vez.

## 6b. Pipeline de alta resolução (medido em 2026-07-28)

**O hires em camadas era o culpado do spill, não a resolução.** Ele processa a
imagem INTEIRA de uma vez; o ultra tiled processa pedaços, e o tile limita a VRAM
pela própria dimensão — a resolução final deixa de importar.

Trocando `gen → hires → ultra` por **`gen 768 → ultra_halo` (sem hires)**:

| | resolução | tempo | folga de VRAM |
|---|---|---|---|
| caminho antigo, 1280x720 | 0,92 MP | 84 s/it (**spill**) | 6% |
| gen → ultra x3.0 core384 | **2074x2074 = 4,3 MP** | **5,1 min** | 0,75-0,88 GB, sem spill |

Quase 5x a área, sem derramar, com o teacher **inteiro** (4,78 GB) — não é preciso
backbone reduzido para chegar lá. O pipeline antigo com ultra levava mais de uma
hora por imagem.

**O prompt do tile tem de ser SÓ TIPO DE DETALHE.** Passar o prompt do assunto
(`1girl, cat ears, green dress...`) faz cada tile de fundo obedecer e desenhar uma
personagem em miniatura — a imagem enche de cópias pequenas. Tiles menores
**pioram** (mais pedaços recebendo a mesma ordem). Use algo como `masterpiece,
best quality, highly detailed, intricate details, fine textures, sharp focus`. O
conteúdo já existe na imagem; denoise 0.35 preserva.

Parâmetros validados: `scale=3.0, core=384, pad=64, overlap=80, denoise=0.35,
steps=26`. Regra do usuário: **steps ≥ 25 e denoise ≤ 0.35**, senão a imagem derrete.

## 6c. Redução do backbone — mapa medido, investigação arquivada

Investigado a fundo e **arquivado**: o pipeline tiled já resolve a resolução, então
destilar deixou de ser prioridade. O mapa fica registrado para quando fizer sentido.

Os centros são **LoRA rank 256** (`to_q/k/v/out`, 1120 chaves), logo o backbone é
matematicamente obrigatório (`W' = W + B@A` não existe sem `W`). E ele é a
**deduplicação**, não o custo: sem ele o substrato compartilhado seria copiado 36
vezes (~30 GB hoje contra ~144 GB de especialistas independentes).

Repartição do UNet: **FFN 47,9%** (os centros não tocam) · atenção 37,2% (onde eles
vivem) · resnets 12,8%.

| corte | GB | s/passo | qualidade (sem treino) |
|---|---|---|---|
| teacher | 4,78 | 1,17 | referência |
| FFN ff2.0 | 3,64 | 0,42 | boa |
| **blocos 6/10** | **3,23** | **0,38** | **boa — melhor por GB** |
| 8/10 + ff2.0 | 3,08 | 0,36 | degradada (os estragos **somam**) |
| blocos 4/10 | 2,45 | 0,32 | quebrada |

Cortar a FFN preserva 100% dos LoRAs. E os centros aguentam **perder 57% das
chaves** sem degradar (`up_blocks.0` + `mid_block` zerados): são redundantes entre
blocos — foi isso que liberou remover blocos inteiros.

O podado 6/10 faz o mesmo 2074x2074 em 2,5 min (2,2x mais rápido) **mas perde o
fundo** (vila e ponte viram bokeh). Serve como modo **rápido** para iterar prompt,
não como padrão.

**Erro metodológico registrado:** medir "o centro ainda age?" por distância de
pixel **não funciona**. Deu ~100% para todos os cenários, inclusive nos pares
parcial-vs-completo — a difusão em seed fixa é caótica, então qualquer perturbação
gera outra imagem. Precisa ser CLIP ou inspeção visual.

## 7. Construído nesta sessão

**Clusters c6/c7** — `build_cluster6.py`, `build_cluster7.py`. Amostragem
**estratificada**: âncora na tag mais rara da imagem + round-robin, cota igual por
tag. Sem isso a tag dominante engole o dataset (`jewelry` com 67k contra `brooch`
com 6k). Rank derivado da dispersão de assinaturas, não fixo.

**Hash de ideia** — `hash_ideia` (tags-âncora) e `hash_pesos` (tensores). Dois
centros com o mesmo `hash_ideia` disputam o mesmo território mesmo com tamanhos
diferentes. A fila avisa antes de gastar horas.

**Anti-TDR no treino** — `--gpu-pausa 0.01`. No ROCm/Windows o contador `Compute`
não tem instância: **tudo despacha na fila 3D**, a mesma do compositor. Sem limite
a fila cresce, o 3D crava 100%, e o watchdog reinicia o driver. Medido: 3D entre
3% e 25%, zero TDR em 9h.

**Centros externos** — `centros_catalogo.py` lê `ss_tag_frequency` (metadado do
kohya, presente em ~72% dos 389 LoRAs locais) e extrai o domínio por **TF-IDF**
contra as frequências globais do Danbooru. Sem TF-IDF o domínio de todo LoRA seria
`1girl, solo, looking_at_viewer` — idêntico entre todos, inútil para rotear.
Resultado: 425 centros disponíveis (36 treinados + 389 externos).

**Hierarquia no roteador** — treinado aqui vem **sempre** antes de baixado, como
primeira chave da ordenação. Medido: sem isso, "hands wearing jewelry, wet skin"
escolhia 6 LoRAs externas casando em tags genéricas e ignorava `bijuteria`,
`maos`, `pele`, `molhado`.

**Sinônimos c6/c7** — os centros foram montados com tags danbooru precisas
(`own_hands_together`, `shiny_skin`, `brick_wall`) e ninguém escreve prompt assim.
`maos`, `pele` e `pedra` existiam e ficavam mudos.

**Autenticação** — OAuth PKCE no navegador padrão (RFC 8252) para CivitAI e HF.
Tokens cifrados com DPAPI. **Nunca** webview embutido: o Google bloqueia
(`disallowed_useragent`) e o gerenciador de senhas do usuário não alcança.

**Caminhos** — `caminhos.py`, fonte única. Antes havia três resoluções
independentes que podiam discordar — e discordavam.

**Buffer de RAM** — o usuário escolhe o tamanho, sem veto. Níveis 1 (passou) e 2
(passou 20% além) só incomodam: barra colorida, aba piscando, alarme sonoro,
balão de ajuda forçado. **Só o teto de 83% do sistema bloqueia** a geração.

**Text encoder** — modos `cpu` / `sequencial` / `residente`. O sequencial sobe o
TE só para codificar e desce antes do denoise; as duas fases não competem no
tempo. Medido 8,8x mais rápido que a CPU, sem OOM em 8 GB.

**Scan & adapt** — qualquer LoRA SDXL vira centro se tiver tags legíveis. 61% da
biblioteca local qualifica.

**Paginação por bloco contíguo** (2026-07-26, tarde) — os pesos de cada centro
viram **um tensor plano em memória pinada**, e os parâmetros são *views* dentro
dele. Paginar passou a ser uma cópia por centro em vez de uma por módulo: com 7
centros, de 7.840 transferências por passo para 7.

Medido: **1,376 → 0,174 s/passo** com 5 centros (7,9x). Numa geração de 25
passos, 34 s → 4 s só de paginação.

Quatro pontos de movimentação foram roteados pelo bloco, não só o `_page`: o
`_politica_vram` também move nichos direto, e um `.to()` por módulo apontaria os
parâmetros para fora do buffer, deixando as views órfãs. Fallback para o caminho
antigo se um centro tiver dtypes misturados.

Verificado com `allclose` que os pesos sobrevivem à ida e volta — se as views
desalinhassem, o modelo geraria lixo **sem dar erro**.

## 8. Armadilhas que custaram tempo (não repetir)

**Validar JS contando chaves não é validar sintaxe.** Reportei "OK" várias vezes
sobre um `app.js` quebrado. O `node --check` está instalado e é o parser de
verdade — **use ele**.

**A camada de shell come barras invertidas.** Escrever `"\n"` em JS via script
Python pelo shell produz uma quebra de linha **literal** dentro da string, que é
erro de sintaxe. Aconteceu três vezes. Solução: `String.fromCharCode(10)` ou
`chr(92)` para emitir a barra.

**Trava de segurança tem que falhar FECHADA.** Escrevi uma que usava `wmic`
(removido do Windows 11) e devolvia lista vazia no `except` — erro de detecção
virava "pode prosseguir". Resultado: movi 36 centros com o treino rodando. Foi
revertido sem perda porque os `.pt` não são movidos por design.

**PowerShell 5.1 não tem `&&`.** E lê `.ps1` como ANSI: um caractere não-ASCII
quebra o parser longe de onde ele está. Scripts `.ps1` **só em ASCII**, validados
com `Parser::ParseFile`.

**Ordem de resolução importa mais que o conteúdo.** Depois de migrar os centros
para `D:\Models\gsmde`, `G:\Trainer_v13` continuava **primeiro** na lista de
candidatos em três arquivos — o app seguiria lendo 29 centros velhos e ignorando
7 novos, sem erro nenhum.

**Nem todo LoRA baixado carrega.** Alguns tocam camadas fora da atenção que o
conversor do diffusers não mapeia. Cada carga fica num `try`: pula com aviso e
segue. Um LoRA de terceiro não pode derrubar a rodada.

**`load_lora_weights` chama o carregador do text encoder mesmo sem chaves de TE**,
e o `rank_dict` vazio estoura `IndexError`. Use `unet.load_lora_adapter()`.

**`hipErrorLaunchFailure` sob paginação módulo a módulo — RESOLVIDO.** A GPU
travava (o próprio Gerenciador de Tarefas parava de renderizar) e o erro saía
como *unspecified launch failure* num `conv2d` qualquer — assíncrono, então o
lugar do rastro não era o lugar da falha.

Causa confirmada, não hipótese: ao tentar reproduzir num benchmark, o caminho
antigo deu `hipErrorOutOfMemory` movendo 7 centros módulo a módulo. As cópias
assíncronas se acumulam mais rápido do que são liberadas. Em produção o driver
nem chegava a reportar OOM limpo — devolvia launch failure e travava.

Sintoma associado: **9,5 GB de memória compartilhada** com apenas 4,87 GB
alocados pelo torch. Não éramos nós inflando; era o WDDM reagindo à fragmentação
causada pela enxurrada de alocações pequenas.

E explicava os **86% do tempo em `yield_gpu`** que o `[perf]` reportava: o
`synchronize()` esperava uma fila de milhares de cópias minúsculas drenar.

**Depois de um `hipErrorLaunchFailure` o contexto HIP é irrecuperável.** O worker
captura o traceback e segue no laço, mas toda chamada seguinte falha — ele fica
segurando VRAM e RAM sem gerar nada. Mate o processo antes de tentar de novo.

## 9. Pendências, por ordem de valor

1. **Terminar o upload da HF** (comando na seção 4) — destrava Space e artigo
2. **Retomar o treino do c7** — faltam `unhas` (em curso), `pes`, `boca`, `maquiagem`
3. **Extrair o launcher do Studio** do `app.py`, que mistura SD.Next e GSMDE.
   Hoje o repo é biblioteca, não aplicativo executável. É o maior bloqueio para
   alguém de fora usar.
4. **Fechar o circuito dos caminhos** — `caminhos.py` é fonte única do launcher,
   mas o motor e o trainer ainda têm listas próprias
5. **Ligar o download** de centros pela UI (busca e catálogo já funcionam)
6. **Space na HF** com scan & adapt e geração
7. **Artigo no CivitAI** apontando para GitHub e HF
8. **Aba de benchmark** com compartilhamento de resultados por hardware —
   `bench_te.py` já grava JSON agregável. Precisa ser **opt-in**: o JSON
   identifica a máquina.
9. **Medir o ganho da paginação por bloco numa geração completa.** O 7,9x é
   da paginação isolada; quanto isso melhora o total depende de que fração do
   tempo era paginação. Procurar no log a linha
   `[gsmde] blocos contiguos: N centros ... 1 transferencia por centro`
9b. **Validar o i2i** em imagem de alta resolução — nunca foi testado de verdade
10. **Sobreposição de transferência** (duplo buffer): copiar o centro *i+1*
    enquanto o *i* calcula. Ficou mais viável depois da paginação por bloco — os
    pesos já estão em memória pinada, que era um dos dois pré-requisitos. Falta
    confirmar se o ROCm/Windows honra stream de cópia paralela (o contador
    `Copy` nunca mostrou atividade nesta placa). **Medir o teto antes**: a
    paginação caiu de 34 s para 4 s numa geração de 25 passos, então o que
    sobra para esconder é bem menor do que era
11. **madeira** (1.588 imgs, abaixo do piso de 2.000) e **texto** (precisa de
    LoRA teacher) seguem sem centro
12. Atualizar **github.com/NOTcisLol/IDK-Trainer** com o pipeline de treino

## 10. Decisões de projeto (não reverter sem motivo)

- **i2i não é re-noise.** Halo de três camadas: externa = contexto, média =
  blending, núcleo = geração do zero.
- **Supersampling: liberado (revisto em 2026-07-28).** A regra antiga era "sem
  tiling e sem supersampling", pelo receio de que reescalar só a região mascarada
  criasse descasamento de ruído ("photoshop mal feito"). O receio era *a priori*;
  testes exaustivos no ultra depois disso mostraram que a costura **aguenta sem
  gerar artefato**. Fazer upscale da região antes de gerar dá ao modelo mais pixels
  para trabalhar e ele **alucina menos**. Vale para o caminho do detailer
  (YOLO define a máscara → `ultra_halo` regenera com padding + halo).
- **Quatro contextos no i2i**: global (IP-Adapter), vizinho (halo), costura
  (feather) e prompt.
- **O agendamento mora no CivitAI**, não aqui. Nada de agendador local: o PC pode
  estar desligado na hora marcada. (A API oficial não expõe rota de posts —
  medido: `posts`, `user/posts`, `me/posts` dão 404.)
- **Centros baixados ficam separados por procedência** em disco. Não é
  organização, é responsabilidade: foram treinados contra outro base.
- **Alertas de buffer não cortam nem bloqueiam.** Cortar centros pelas costas
  daria imagem pior sem explicação.
- **Um centro por vez no treino**, uma imagem por passo, requeue por loss, sem
  épocas.

---

## Variantes de redução do backbone — como cada uma nasceu

Escrito em **2026-08-08**. Todos os números saíram de execução nesta máquina.

O objetivo é o mesmo desde o início: **caber nos 8 GB sem despejo**. A UNet do IDN
ocupa 4,78 GB em fp16, e o despejo dessa ordem foi o que travou a máquina várias
vezes — inclusive derrubando o driver para o genérico da Microsoft.

Há **duas** estratégias implementadas, não três. O que parecia uma terceira era o
parâmetro da primeira.

### 1. Corte de camadas transformer  (`treina_backbone.py`, `cfg_student()`)

Reduz `transformer_layers_per_block` do nível mais profundo. O teacher é `[1,2,10]`;
o student em uso é `(1,2,6)`.

O corte NÃO escolhe "up" ou "down": o terceiro valor é o nível de 1280 canais, que
aparece em **três lugares ao mesmo tempo** — `down_blocks.2`, `mid_block` e
`up_blocks.0`.

| camadas | fp16 | redução | LoRAs encaixam |
|---|---|---|---|
| 10 (teacher) | 4,78 G | — | 560/560 (100%) |
| 8 | 4,01 G | 16,2% | 464/560 (83%) |
| 6 (o student) | 3,23 G | 32,5% | 368/560 (66%) |
| 4 | 2,45 G | 48,7% | 272/560 (49%) |
| 2 | 1,67 G | 65,0% | 176/560 (31%) |

**Corte assimétrico é possível** via `reverse_transformer_layers_per_block`, e
`down=10/up=6` tem exatamente o mesmo tamanho que `down=6/up=10` (2,150B, 4,01 GB) —
mesma economia, capacidade em lados opostos da U.

### 2. Encolhimento da FFN  (`destila_backbone.py`)

Reduz só a rede feed-forward, medida em 47,9% dos pesos da UNet. **As formas da
atenção ficam idênticas**, então LoRAs e centros encaixam por inteiro.

| ff_mult | fp16 | redução | LoRAs |
|---|---|---|---|
| 4 (teacher) | 4,78 G | — | 100% |
| 1.5 (o construído) | 3,35 G | 29,9% | **560/560 (100%)** |
| 1.0 | 3,06 G | 35,9% | 100% |
| 0.5 | 2,78 G | 41,9% | 100% |

A FFN é truncada por **norma** — ficam as linhas de maior magnitude. Atenção,
resnets e embeddings são copiados bit a bit do teacher.

### 3. Corte ASSIMETRICO da descida — a estrategia escolhida (10/08/2026)

`reverse_transformer_layers_per_block` permite cortar a descida e a subida em
quantidades diferentes. Medido em 09-10/08 com heranca de pesos do teacher e **sem
nenhuma destilacao**, as quatro LoRAs assadas a 0,25, mesmo prompt e mesma seed:

| variante | params | fp16 | menor | LoRAs encaixam | imagem |
|---|---|---|---|---|---|
| teacher | 2,567 B | 4,78 G | — | 100% | referencia |
| d6u10 | 2,150 B | 4,01 G | 16,2% | 83% | boa, indicador funde na xicara |
| **d5u10 — ESCOLHIDA** | **2,046 B** | **3,81 G** | **20,3%** | **79%** | **boa** |
| d4u10 | 1,942 B | 3,62 G | 24,4% | 74% | boa |
| d3u10 | 1,838 B | 3,42 G | 28,4% | 70% | boa, maos limpas |
| d2u10 | 1,733 B | 3,23 G | 32,5% | 66% | boa |
| d1u10 | 1,629 B | 3,03 G | 36,5% | 61% | boa, maos limpas |

**NENHUMA quebrou, nem a d1u10.** A descida aguentou o corte ate 1 camada. O ponto de
deterioracao que se procurava nao existe dentro dessa faixa.

**O que isso ensina:** o que destruia o corte simetrico (1,2,6) nao era perda de
capacidade — era cortar a SUBIDA, onde o detalhe e' reconstruido. A descida so'
codifica e tem folga enorme. O corte simetrico destilado por 20500 passos sai PIOR que
qualquer assimetrica sem treino nenhum.

**Escolha do usuario: d5u10.** Um pouco maior que o FFN 1.5 (3,81 contra 3,35 GB) mas
com qualidade de imagem muito superior, e ~1 GB abaixo do teacher. **Todas as outras
ficam no disco para teste posterior**, justamente porque nenhuma saiu quebrada.

Arquivos: `D:/GSMDE/outputs/treino/students_cur/var_<nome>.pt` (8 variantes).
Gerador: `D:/GSMDE/auto/varre_down.py` e `variantes_corte.py`.

**RESSALVA NAO RESOLVIDA:** o quadro tem UMA seed e UM prompt — close de rosto com
estante ao fundo, que quase nao convoca as LoRAs. O custo dos 79% de compatibilidade
nao apareceu porque a cena nao pedia os temas. Antes de fixar a d5u10 em producao,
repetir com prompt que chame material, cena e mao ao mesmo tempo.

### A comparação que importa

Corte a 6 camadas dá 32,5% de economia com 66% de compatibilidade; FFN a 1.5 dá
29,9% com 100%. **Praticamente a mesma VRAM, e a FFN preserva a compatibilidade
inteira.** Verificado na prática: 560 de 560 módulos presentes, zero formas erradas.

### Onde os arquivos estão

```
D:/Models/gsmde/backbone_student/student_raw.pt                 FFN 1.5, herdado, NAO destilado
D:/GSMDE/outputs/treino/students_cur/preservados/               corte (1,2,6), destilado ate 20500
D:/GSMDE/outputs/treino/students_cur/var_*.pt                   variantes assimetricas (d10u6, d6u10, d8u8)
D:/GSMDE/outputs/treino/students_cur/student_ffn_mesclado.pt    FFN + 4 LoRAs @ 0.25
D:/GSMDE/outputs/treino/students_cur/student_corte_mesclado.pt  corte + 4 LoRAs @ 0.25
```

### O acidente que custou o melhor checkpoint

O student de corte chegou a **loss 0,0061 no passo 19329** e gerava imagem boa. Ao
retomar o treino ele caiu para 0,026 em 200 passos e estacionou. Causa: **o
checkpoint nunca gravou o estado do Adafactor** — só `state_dict` e `cfg`. Cada
retomada reinicia os segundos momentos.

Somado a `--manter 3`, que podou os checkpoints antigos, e ao rastreio de "melhor"
que reinicia por sessão e sobrescreveu `melhor.pt` com um pior, **os pesos de 19329
se perderam**. Antes de retomar qualquer destilação, gravar o estado do otimizador.

---

# Bancada de 16–17/08/2026 — detailer, e quatro defeitos que a métrica não pegou

Detalhes e números em `docs/detailer_e_ordem_de_carga.md`.

**Centros como detailer funciona.** Gerar só com o backbone e refazer as regiões
achadas por YOLO custa **300 s contra ~990 s** dos 6 centros no laço, com despejo
de 0,42 contra 2,98 GB. A geração principal passa a ser 1 passada por passo.

**A seed não determinava a imagem** — `pipe.scheduler.step()` não recebia
`generator=`, então o ruído ancestral do EulerAncestral vinha do RNG global. Duas
corridas iguais davam nitidez 346 e 201. Corrigido: diferença de 0 pixels.
**Toda comparação A/B anterior a 17/08 carrega esse ruído.**

**O despejo era na CARGA, não no laço.** `pipe.unet.to(dev)` arrasta os nichos
pendurados na árvore: 4,78 + 3,81 = 8,59 GB de uma vez numa placa de 8. O laço
sempre esteve limpo (0,10 GB). Desanexando antes de subir: 6,51 GB e 1,04 GB de
despejo. Ainda não zerou — falta a janela do `aplica_offload_base`.

**Carona custa MAIS que ser centro:** +54% de tempo para remover 2 passadas,
porque cada passada passa a mover 3 blocos em vez de 1. Confirmar com 8 centros.

**Regra do par:** nunca repintar um olho sozinho — sai uma íris azul e outra
castanha. Medido nos dois sentidos (1 olho → diverge; 2 olhos numa passada →
converge).

**Método:** três vezes seguidas uma métrica de quantidade (diferença média de
pixel, variância do laplaciano) foi usada para julgar identidade ou localização e
errou. O que funciona é olhar a imagem e depois usar diff com componentes conexos
para localizar a mudança e conferir se ela bate com o alvo.
