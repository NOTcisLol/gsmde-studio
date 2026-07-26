# GSMDE Studio

**Graph-Structured Mixture of Denoising Experts** — difusão modular por especialistas,
com roteamento semântico e paginação de centros, rodando em placa de 8 GB.

Em vez de um UNet monolítico que carrega tudo a cada passo, o GSMDE mantém uma
prateleira de **centros** (LoRAs especialistas), lê o prompt, decide quais são
relevantes e sobe só esses para a placa. O codificador de texto atua como
mecanismo de roteamento topológico, não como vetor de força bruta.

> Formulação teórica: *Formulação Matemática e Arquitetural de Modelos de Difusão
> Baseados em Grafos e Mistura de Especialistas*.

---

## Estado

Funcional e em uso diário numa **RX 7600 (8 GB, ROCm nativo no Windows)**. Não é
release: a API muda, e há partes marcadas como pendência abaixo.

## O que há aqui

```
engine/     motor de difusão, roteador semântico, catálogo de centros, benchmarks
launcher/   Studio (UI em pywebview) e integrações CivitAI / HuggingFace
config/     definição dos clusters de centros (tags por nicho)
docs/       protocolo de treino e resultados de benchmark
```

## Números medidos nesta máquina

Não são estimativas — saíram de medição, e a metodologia está nos comentários
do código.

| | antes | depois |
|---|---|---|
| geração com 4 centros | 45 s/it | 11 s/it |
| política de paginação | 53,9 s/passo (tudo residente) · 15,0 (tudo paginado) | **10,4** (auto) |
| encode do prompt, 6 centros | 212 s (CPU fp32) | **0,92 s** (GPU fp16, sequencial) |
| encode, cache do negativo | 31 s | **16,8 s** |

O caso do encode ilustra por que este projeto mede em vez de assumir: o build
ROCm do torch vem sem MKL, então fp32 na CPU cai num caminho ingênuo e fica
**230x** mais lento que o melhor caminho. Nenhuma tabela de "placa X prefere
dtype Y" preveria isso — é propriedade do build, não do hardware.

## Conceitos

**Centro** — um LoRA especialista num domínio (pele, mãos, pelagem, joias). Sobe
para a VRAM só quando o roteador o escolhe.

**Roteamento** — o prompt é lido contra o léxico de tags de cada centro. A palavra
que casou já é a âncora espacial da máscara: rotear e mascarar são o mesmo ato.

**Hierarquia** — centros treinados para a tarefa vêm sempre antes de LoRAs
baixados. Um LoRA de terceiro foi treinado contra outro base e não passou por
curadoria; pode somar quando há vaga, não deslocar.

**Paginação** — os centros moram na RAM e sobem um por vez. Quem limita quantos
cabem é a RAM, não a VRAM.

## Requisitos

- Python 3.11, PyTorch com CUDA ou ROCm
- `diffusers`, `transformers`, `safetensors`, `peft`, `pywebview`, `psutil`
- Um checkpoint SDXL como backbone (testado com Illustrious)

Os caminhos são configuráveis na UI (aba Configuração › Pastas); a detecção
automática procura uma biblioteca no padrão SD.Next.

## Pendências conhecidas

- **Entrada do Studio ainda não extraída.** O `app.py` que sobe a janela vive
  no launcher pessoal e mistura gerência do SD.Next com o GSMDE. Os módulos aqui
  são independentes, mas falta o ponto de entrada isolado.
- **Agenda do CivitAI** lê só o que já foi publicado: a API oficial não expõe
  rota de posts (medido — `posts`, `user/posts` e `me/posts` retornam 404),
  embora os escopos `MediaRead`/`MediaWrite` mencionem posts.
- **Download de centros** pela UI ainda não está ligado; a busca e o catálogo já
  funcionam.
- **Nem todo LoRA baixado carrega**: alguns tocam camadas fora da atenção que o
  conversor do diffusers não mapeia. Esses são pulados com aviso.

## Licença

A definir.
