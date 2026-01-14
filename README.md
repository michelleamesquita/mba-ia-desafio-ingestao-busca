# Desafio RAG: Ingestão e Busca de PDF com pgVector

Este projeto implementa uma solução de RAG (Retrieval-Augmented Generation) que permite ingerir documentos PDF em um banco de dados vetorial PostgreSQL (com a extensão pgVector) e realizar perguntas sobre o conteúdo via CLI.

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar a aplicação em seu ambiente local.

### 1. Pré-requisitos

*   **Python 3.10+**
*   **Docker** e **Docker Compose**
*   Chave de API da **OpenAI** ou **Google Gemini**

### 2. Configuração do Ambiente

Primeiro, clone o repositório e navegue até a pasta do projeto. Em seguida, crie e ative um ambiente virtual:

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar no macOS/Linux:
source .venv/bin/activate

# Ativar no Windows:
# .venv\Scripts\activate
```

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

### 3. Configuração das Variáveis de Ambiente

Crie um arquivo chamado `.env` na raiz do projeto e preencha com suas configurações:

```env
# Chaves de API
OPENAI_API_KEY=sua_chave_openai
GOOGLE_API_KEY=sua_chave_google

# Configurações do PDF e Banco
PDF_PATH=document.pdf
POSTGRES_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# Escolha do Provedor (openai ou google)
LLM_PROVIDER=openai

# Modelos (Opcional - já configurados com valores padrão seguros)
# LLM_MODEL_OPENAI=gpt-4o-mini
# LLM_MODEL_GOOGLE=gemini-1.5-flash
```

### 4. Iniciando o Banco de Dados

Utilize o Docker Compose para subir o PostgreSQL com a extensão pgVector:

```bash
docker-compose up -d
```

### 5. Ingestão de Dados

Certifique-se de que o arquivo PDF (especificado em `PDF_PATH`) está na raiz do projeto. Execute o script de ingestão para processar o PDF e salvar os vetores no banco:

```bash
python src/ingest.py
```

### 6. Execução do Chat (Busca)

Após a ingestão, você pode iniciar o chat interativo via linha de comando:

```bash
python src/chat.py
```

---

## 🛠️ Tecnologias Utilizadas

*   **LangChain**: Framework principal para orquestração do RAG.
*   **PostgreSQL + pgVector**: Armazenamento de vetores e busca por similaridade.
*   **OpenAI/Gemini**: Modelos de Embeddings e LLM para geração de respostas.
*   **Docker**: Conteinerização do banco de dados.

## 📝 Regras de Resposta

O sistema foi configurado para:
1. Responder **apenas** com base no conteúdo do PDF.
2. Se a informação não for encontrada, responder: *"Não tenho informações necessárias para responder sua pergunta."*
3. Nunca utilizar conhecimento externo ou inventar informações.
