# MEMORY.md — estado do projeto e continuidade

Documento de handoff. Escrito em **2026-07-26**, ao fim de uma sessão longa, para
que outra conta/assistente consiga continuar sem redescobrir o que já foi medido.

Regra que vale para quem continuar: **quase tudo aqui foi medido, não presumido.**
Onde houver número, ele saiu de execução real nesta máquina. Onde houver dúvida,
está marcado como dúvida. Não troque medição por heurística.

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

| o quê | onde |
|---|---|
| repositório publicado | `D:\gsmde-studio` → github.com/NOTcisLol/gsmde-studio |
| motor + pipeline de treino | `D:\Trainer_v13` (**não é repo git ainda**) |
| launcher pessoal (dev) | `G:\SD.NEXT\launcher` (mistura SD.Next — **não publicar**) |
| centros treinados | `D:\Models\gsmde\specialists` (36 centros, 16,9 GB) |
| LoRAs da biblioteca | `D:\Models\Lora` (389 arquivos) |
| saídas de treino | `D:\Trainer_outputs` |

`D:\gsmde-studio` é uma **cópia** dos arquivos, não um link. Ao editar em
`D:\Trainer_v13\forks` ou `G:\SD.NEXT\launcher`, é preciso copiar de novo para o
repo antes de commitar. Isso é dívida técnica conhecida.

## 4. Estado da publicação

| destino | estado |
|---|---|
| **GitHub** | ✅ `5d70cd1`, 33 arquivos, 9.253 linhas. Público. |
| **HuggingFace** | 🟡 repo `CairoAOGG4/gsmde-centros` criado e vazio (2 arquivos auto). Upload dos 16 GB **não feito**. |
| **Space de demo** | ⬜ não começado — depende do upload acima |
| **Artigo CivitAI** | ⬜ não começado — é o último, aponta para os dois anteriores |

Identidades: git `connectcairo@gmail.com` / `NOTcisLol`; HF `CairoAOGG4`.

### O comando do upload que falta

O `--include` do typer é **flag repetida**, não valores separados por espaço —
errei isso duas vezes. Mais simples listar o que fica de fora:

```powershell
& "G:\SD.NEXT\sdnext\venv\Scripts\hf.exe" upload-large-folder CairoAOGG4/gsmde-centros "D:\Models\gsmde\specialists" --repo-type model --exclude "*.pt" --exclude "logs/*" --exclude "brain_registry.json"
```

`upload-large-folder` (e não `upload`) porque retoma se cair — são 16 GB.

**Já preparados, prontos para subir:**
- `D:\Models\gsmde\specialists\README.md` — model card com tabela dos 36 centros
- `D:\Models\gsmde\specialists\brain_registry.public.json` — registro sem caminhos absolutos

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
Start-Process powershell -ArgumentList '-NoExit','-File','D:\Trainer_v13\supervisor_clusters.ps1' -WindowStyle Minimized
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

**fp16 domina bf16 nos dois eixos** (mais rápido E mais preciso — bf16 tem 7 bits
de mantissa contra 10). O padrão da CPU foi trocado de bf16 para fp16.

Na GPU as três precisões custam ~1 s: **ali não há motivo para baixar precisão**.
Resposta à pergunta "numa H100 justifica?" — não, o ganho é de décimos de segundo
numa fase que roda uma vez.

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
9. **Validar o i2i** em imagem de alta resolução — nunca foi testado de verdade
10. **Sobreposição de transferência** (duplo buffer): copiar o centro *i+1*
    enquanto o *i* calcula. Requer memória pinada e streams paralelas — medir o
    teto antes de implementar
11. **madeira** (1.588 imgs, abaixo do piso de 2.000) e **texto** (precisa de
    LoRA teacher) seguem sem centro
12. Atualizar **github.com/NOTcisLol/IDK-Trainer** com o pipeline de treino

## 10. Decisões de projeto (não reverter sem motivo)

- **i2i não é re-noise.** Halo de três camadas: externa = contexto, média =
  blending, núcleo = geração do zero. Sem tiling e sem supersampling — reescalar
  só a região mascarada cria descasamento de ruído ("photoshop mal feito").
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
