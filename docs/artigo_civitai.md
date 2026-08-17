# GSMDE: e se o UNet não precisasse ser um bloco só?

*Difusão modular por especialistas, rodando em 8 GB — e um jeito de transformar
os LoRAs que você já tem em parte da arquitetura.*

---

## O problema

Toda geração em SDXL carrega o UNet inteiro na VRAM e submete cada token ao
processamento de todos os parâmetros. Não importa se o seu prompt fala de mãos,
de textura de pano ou de arquitetura: os mesmos 2,6 bilhões de pesos passam por
tudo.

Isso contrasta com como a cognição biológica funciona. Ao evocar um conceito
visual específico, o cérebro não ativa a rede toda nem revisa as memórias de uma
vida — ele acessa o nicho especializado naquela ontologia, mantendo os vizinhos
em ativação mínima só para preservar contexto.

O **GSMDE** (*Graph-Structured Mixture of Denoising Experts*) tenta essa segunda
forma: em vez de um bloco monolítico, uma prateleira de **centros** — especialistas
pequenos, cada um dominando um domínio. O codificador de texto deixa de ser um
vetor de força bruta e passa a ser um **mecanismo de roteamento**: ele lê o prompt
e localiza quais nichos importam. Só esses sobem para a placa.

## O achado que simplificou o desenho

A parte que parecia mais difícil se resolveu sozinha.

Para o especialista de mãos pintar as mãos e não o fundo, ele precisa de uma
máscara — precisa saber *onde* na tela está aquilo que é dele. Eu esperava ter
que construir um mecanismo separado para isso.

Mas o roteador já resolve: **a palavra do prompt que casou com a tag do centro é,
ela mesma, a âncora espacial**. Se "jewelry" ativou o centro de bijuteria, é
"jewelry" que aponta onde na imagem a bijuteria está, via atenção cruzada.
Rotear e mascarar são o mesmo ato. O roteador devolve os dois de uma vez, sem
inventar nada.

## O que isso muda na prática

Numa RX 7600 de 8 GB, com ROCm nativo no Windows:

| | antes | depois |
|---|---|---|
| geração com 4 centros | 45 s/it | **11 s/it** |
| política de paginação | 53,9 s/passo (tudo residente) · 15,0 (tudo paginado) | **10,4** (automática) |
| codificação do prompt, 6 centros | 212 s | **0,92 s** |

O ganho não vem de otimização heroica. Vem de **parar de derramar**: quando os
especialistas não cabem todos na placa, o driver começa a paginar sozinho, ao
acaso, pelo PCIe. A paginação explícita — nichos moram na RAM, só o ativo sobe —
troca o caos do driver por trânsito determinístico.

E o teto de 77 tokens do CLIP deixa de ser global e passa a ser **por centro**.
Com seis centros são ~460 tokens úteis, cada um gastando o seu só no que é dele.
De quebra, isso ataca o *attribute bleeding* na origem: se o centro do vampiro
nunca lê a palavra "lobisomem", não há o que vazar.

## A parte que interessa a quem já tem LoRAs

Aqui está o ponto que me parece mais útil para esta comunidade.

**Não existe "modelo GSMDE" para baixar.** Um centro é um LoRA de rank alto sobre
o UNet do SDXL, atacando `to_k / to_q / to_v / to_out.0`. Como todo checkpoint
SDXL compartilha essa arquitetura, esses módulos existem em qualquer um — o que
significa que **os LoRAs que você já tem são centros em potencial**.

O que falta num LoRA baixado não é compatibilidade de peso. É o sistema saber
**qual domínio ele cobre**, senão o roteador nunca o escolhe. E essa informação
costuma estar dentro do próprio arquivo: LoRAs treinados com Kohya gravam
`ss_tag_frequency`, a contagem de tags do dataset de treino. Medi na minha
biblioteca: **72% dos 389 LoRAs locais têm esse metadado**.

Mas ler as tags não basta. As mais frequentes de qualquer LoRA de personagem são
`1girl`, `solo`, `looking_at_viewer` — presentes em metade do Danbooru, idênticas
entre todos, inúteis para decidir. O que define o domínio é a tag que é comum
*ali* e rara *lá fora*. Cruzando com as frequências globais do dump (TF-IDF),
um LoRA que parecia genérico revela sua identidade:

```
cru:      748cmstyle, 1girl, looking_at_viewer, long_hair, jewelry
domínio:  748cmstyle (16.0) · ear_piercing (2.68) · blunt_bangs (1.48) · jewelry (1.25)
```

A *trigger word* sobe ao topo sozinha, porque tem frequência zero no dump — é
exatamente o que identifica aquele LoRA. Nenhuma curadoria manual envolvida.

Há um botão de **scan & adapt** que faz isso: você aponta um `.safetensors`, ele
lê rank, imagens de treino, sobre qual base foi treinado, e o domínio extraído. Se
houver domínio, instala como centro. **61% da minha biblioteca qualificou.**

Uma ressalva honesta: os centros esperam o roteador e a máscara de território.
Aplicados soltos, com peso 1.0 na imagem toda, tendem a inundar a cena — o
desenho supõe que cada um pinta só a região dele.

## O que a medição ensinou

Duas coisas que eu teria errado se tivesse confiado em intuição.

**Hierarquia importa.** Com 389 LoRAs no vocabulário, o prompt *"hands wearing
jewelry, wet skin, stone wall"* escolhia seis LoRAs externas casando em tags
genéricas — e deixava de fora os especialistas de mãos, joias, pele e molhado,
construídos exatamente para aquilo. São 389 contra 36: qualquer empate tende para
a maioria. A correção foi tornar a procedência a primeira chave da ordenação. Um
LoRA de terceiro pode somar quando há vaga; não tem por que deslocar um centro
feito para a tarefa.

**Tabela de hardware não substitui medir.** Eu ia embutir uma heurística do tipo
"placa X prefere precisão Y". Medi antes, e o resultado dispensou a tabela:

| dispositivo | precisão | segundos |
|---|---|---|
| GPU | fp16 | **0,92** |
| CPU | fp16 | 5,33 |
| CPU | bf16 | 8,10 |
| CPU | fp32 | **212,19** |

O fp32 na CPU é 230x pior que o melhor caminho — não por causa da placa, mas
porque este build do PyTorch para ROCm vem sem MKL e cai num caminho ingênuo.
Nenhuma tabela por modelo de GPU preveria isso; é propriedade do build.

E o padrão que eu tinha escrito no código (bf16 na CPU) estava errado nos dois
eixos: fp16 é 1,5x mais rápido **e** mais preciso — bf16 tem 7 bits de mantissa
contra 10 do fp16, então perde dez vezes mais na comparação com fp32.

Sobre baixar precisão: só justifica quando o segundo economizado vale algo. Na CPU
aqui, 212 s → 5 s decide tudo. Numa placa rápida, a diferença entre fp32 e fp16 é
de décimos de segundo numa fase que roda uma vez por geração — não paga pensar no
assunto.

## Estado do projeto

Funcional e em uso diário, **não é release**. São 36 centros treinados, cobrindo
sujeito, cena, iluminação, câmera, e dois clusters novos de detalhe: texturas
(pele, pelagem, tecido, molhado, pedra) e anatomia fina (mãos, bijuteria).

O que está aberto, sem enfeite:

- **Falta o ponto de entrada isolado.** O launcher que sobe a janela ainda mistura
  gerência do SD.Next com o GSMDE. Hoje o repositório é biblioteca, não aplicativo
  clicável — é o maior obstáculo para alguém de fora usar.
- Quatro centros do cluster de anatomia (unhas, pés, boca, maquiagem) ainda não
  foram treinados.
- Os dois clusters novos rodaram com 1,6 passada pelo dataset, não com o currículo
  completo. Funcionam, mas não estão no ponto que o protocolo pede.
- Nem todo LoRA carrega: alguns tocam camadas fora da atenção que o conversor do
  diffusers não mapeia. Esses são pulados com aviso.

## Links

- **Código:** https://github.com/NOTcisLol/gsmde-studio
- **Centros treinados:** https://huggingface.co/CairoAOGG4/gsmde-centros

O `MEMORY.md` do repositório traz o estado detalhado, os números medidos e as
decisões de projeto — inclusive os erros que custaram tempo, com o mecanismo de
cada um.

Se você tem uma placa apertada e uma biblioteca grande de LoRAs, essa combinação é
exatamente o caso de uso. E se testar, me diga onde quebrou — a lista de
limitações acima saiu toda de coisa quebrando.
