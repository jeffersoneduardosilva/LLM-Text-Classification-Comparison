# 🤖 Comparação de Técnicas de Classificação de Texto em LLMs: Zero-Shot, Fine-Tuning e RAG

Este projeto implementa e compara três estratégias de Modelos de Linguagem (LLMs) para a tarefa de **Classificação de Texto Zero-Shot** em um domínio técnico e academicamente denso: artigos científicos do arXiv.

O principal objetivo é avaliar o **trade-off entre o custo de treinamento/adaptação e o desempenho** de cada abordagem na distinção entre tópicos altamente correlacionados, como **Inteligência Artificial (AI)** e **Aprendizado de Máquina (ML)**, utilizando as categorias `cs.AI` e `cs.LG`.

---

## 🔗 URL do Youtube / Apresentação do Projeto

| Tipo de Conteúdo | Status | Link |
| :--- | :--- | :--- |
| **Apresentação do Projeto** | *Pendente* | [INSERIR_LINK_AQUI] |

---

## 🚀 Implementação e Estratégias

O experimento utiliza um dataset balanceado de 1000 resumos de artigos científicos (500 de `cs.AI` e 500 de `cs.LG`) coletados via API do arXiv.

### 1. Zero-Shot Classification (Modelo Base: BART-MNLI)

Esta abordagem serve como linha de base.

* **Modelo Utilizado:** `facebook/bart-large-mnli`. Este modelo é treinado para a tarefa de *Inferência de Linguagem Natural* (NLI) no corpus MNLI e transfere essa capacidade para classificação, inferindo a relação entre o texto de entrada e o rótulo candidato.
* **Mecanismo:** O classificador avalia quão bem o resumo do artigo (`TEXTO ALVO`) implica ou contradiz o rótulo (`AI` ou `ML`), sem a necessidade de qualquer dado rotulado de treino específico do domínio.
* **Resultado do Notebook:** Apresentou a acurácia mais baixa ($\approx 49.5\%$), indicando que o modelo NLI pré-treinado tem dificuldade em diferenciar subdomínios técnicos com vocabulário muito sobreposto.

### 2. Fine-Tuning (Adaptação Supervisionada: SciBERT)

Esta é a abordagem supervisionada padrão, que estabelece o teto de desempenho (benchmarking).

* **Modelo Utilizado:** `allenai/scibert_scivocab_uncased`. Foi escolhido por ser um modelo BERT otimizado e pré-treinado especificamente em uma grande coleção de artigos científicos, garantindo que o vocabulário técnico seja compreendido de forma mais eficaz.
* **Mecanismo:** O modelo é ajustado (fine-tuned) por 3 épocas em 800 exemplos rotulados, aprendendo a mapear as características de texto para as classes `AI` (0) e `ML` (1).
* **Resultado do Notebook:** Alcançou o melhor desempenho ($\approx 62.0\%$ de acurácia), o que era esperado devido ao ajuste direto à tarefa e à especialização do modelo base (SciBERT) no domínio científico.

### 3. RAG - Retrieval-Augmented Classification (Híbrido)

Esta abordagem busca melhorar a classificação Zero-Shot, adicionando contexto sem a necessidade de Fine-Tuning supervisionado.

* **Componentes:**
    1.  **Embeddings:** `all-mpnet-base-v2` (Sentence-Transformers) para codificar o corpus.
    2.  **Índice Vetorial:** **FAISS** (`IndexFlatL2`) para busca rápida de vizinhos.
    3.  **Classificador:** O mesmo Zero-Shot **BART-MNLI**.
* **Mecanismo:**
    1.  Para cada texto de teste, os **K=5** artigos mais semanticamente similares são recuperados do corpus indexado.
    2.  O texto original é enriquecido com o título, resumo parcial e o rótulo dos vizinhos mais próximos.
    3.  Essa *entrada aumentada* é fornecida ao classificador BART-MNLI para que ele use as informações de contexto (que já contêm o rótulo verdadeiro de artigos similares) na sua decisão.
* **Resultado do Notebook:** Obteve desempenho intermediário ($\approx 57.5\%$ de acurácia), demonstrando que a **recuperação semântica é eficaz** para aumentar a precisão da classificação Zero-Shot em domínios correlacionados.

---

## 📊 Resumo Comparativo das Métricas

O `F1 Macro Score` é a métrica principal, pois considera a precisão e o recall para ambas as classes (`AI` e `ML`), sendo mais robusta para avaliação.

| Método | Accuracy | F1 Macro Score |
| :--- | :--- | :--- |
| **Fine-Tuning (SciBERT)** | **0.6200** | **0.6153** |
| **RAG (Embeddings + Zero-Shot)** | 0.5750 | 0.5631 |
| **Zero-Shot (BART-MNLI)** | 0.4950 | 0.4826 |

---

## ⚙️ Detalhes de Implementação

### Dataset e Pré-processamento

* **Fonte:** API do arXiv, categorias `cs.AI` e `cs.LG`.
* **Tamanho Total:** 1000 artigos (500 AI, 500 ML).
* **Divisão:** 800 para treino/corpus, 200 para teste.
* **Entrada do Modelo:** Título e Abstract concatenados (`title + " - " + abstract`).

### Dependências (Instalação)

```bash
# Necessário rodar no notebook, idealmente em ambiente com GPU (para Fine-Tuning)
!pip install transformers datasets sentence-transformers faiss-cpu arxiv accelerate scikit-learn -q
!pip install --upgrade transformers accelerate datasets -q

🛠️ Como Rodar o Notebook
O arquivo Trabalho_Final_Prof_Rogerio.ipynb contém a implementação e a comparação de três métodos de classificação de texto em LLMs (Zero-Shot, Fine-Tuning e RAG).

NOTA: Para garantir a execução bem-sucedida da etapa de Fine-Tuning e dos componentes de embeddings (RAG), é altamente recomendável usar um ambiente com GPU (Google Colab ou uma máquina local com setup CUDA) para reduzir drasticamente o tempo de processamento.

💻 Opção 1: Rodar Localmente via VS Code
Esta opção é ideal se você possui um ambiente Python configurado e, preferencialmente, acesso a uma GPU local.

Pré-requisitos
Python: Tenha o Python (3.8+) instalado.

VS Code: Tenha o Visual Studio Code instalado.

Extensões do VS Code: Instale as seguintes extensões:

Jupyter

Python

Passos de Execução
Configurar Ambiente Virtual (Recomendado):

Bash

python -m venv venv
# Ativar no macOS/Linux:
source venv/bin/activate
# Ativar no Windows:
.\venv\Scripts\activate
Abrir e Conectar o Kernel:

Abra o arquivo Trabalho_Final_Prof_Rogerio.ipynb no VS Code.

Clique em "Select Kernel" (Canto superior direito) e escolha o ambiente virtual que você acabou de criar/ativar.

Instalar as Dependências:

Execute a primeira célula do notebook (Seção 0 - INSTALAÇÕES INICIAIS) para garantir que todas as bibliotecas necessárias estejam instaladas no ambiente.

Executar o Projeto:

Execute as células restantes em ordem sequencial (Seções 1 a 13) para:

Importar bibliotecas.

Baixar os artigos do arXiv.

Realizar as três abordagens de classificação (Zero-Shot, Fine-Tuning e RAG).

Exibir o relatório de conclusão.

☁️ Opção 2: Rodar na Nuvem via Google Colab (Recomendado)
Esta é a opção mais simples e garante acesso a recursos de GPU para otimizar o tempo de execução.

Passos de Execução
Acessar o Colab: Abra o Google Colab (https://colab.research.google.com/).

Fazer Upload do Notebook:

Clique em "File" (Arquivo) > "Upload notebook" (Fazer upload de notebook).

Selecione e carregue o arquivo Trabalho_Final_Prof_Rogerio.ipynb.

Ativar a GPU (Passo Obrigatório para Fine-Tuning):

Vá em "Runtime" (Ambiente de execução) no menu superior.

Selecione "Change runtime type" (Alterar tipo de ambiente de execução).

Em "Hardware accelerator", escolha GPU.

Clique em "Save" (Salvar).

Executar Todas as Células:

Vá em "Runtime" (Ambiente de execução) no menu superior.

Selecione "Run all" (Executar tudo).

O Colab irá instalar as dependências, baixar os dados do arXiv e executar todas as etapas da comparação de modelos. O processo de Fine-Tuning (Seção 8) será o mais demorado, mesmo com a GPU ativa.
