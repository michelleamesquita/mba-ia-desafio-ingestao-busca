import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def get_embeddings():
    """Returns the embedding model based on the provider."""
    if LLM_PROVIDER == "openai":
        return OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL_OPENAI", "text-embedding-3-small"))
    elif LLM_PROVIDER == "google":
        return GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL_GOOGLE", "models/embedding-001"))
    else:
        raise ValueError(f"Provider {LLM_PROVIDER} not supported.")

def get_llm():
    """Returns the LLM based on the provider."""
    if LLM_PROVIDER == "openai":
        # Note: using gpt-4o-mini as default
        return ChatOpenAI(model=os.getenv("LLM_MODEL_OPENAI", "gpt-4o-mini"), temperature=0)
    elif LLM_PROVIDER == "google":
        # Note: using gemini-1.5-flash as default
        return ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL_GOOGLE", "gemini-1.5-flash"), temperature=0)
    else:
        raise ValueError(f"Provider {LLM_PROVIDER} not supported.")

def format_docs(docs):
    """Concatenates the content of retrieved documents."""
    context = "\n\n".join(doc.page_content for doc in docs)
    # print(f"\n--- DEBUG: Contexto recuperado ({len(docs)} chunks) ---")
    # print(context[:500] + "...") 
    return context

def search_prompt():
    """Configures and returns the RAG chain."""
    try:
        embeddings = get_embeddings()
        llm = get_llm()
        
        # Initialize Vector Store
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name="pdf_docs",
            connection=POSTGRES_URL,
            use_jsonb=True,
        )
        
        # Configure retriever with k=10 as per Agents.md
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Define prompt
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Create the RAG chain
        chain = (
            {"contexto": retriever | format_docs, "pergunta": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    except Exception as e:
        print(f"Error initializing search chain: {e}")
        return None
