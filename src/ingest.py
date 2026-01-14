import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Load environment variables from .env file
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

def get_embeddings():
    """Returns the embedding model based on the provider configured in .env"""
    if LLM_PROVIDER == "openai":
        return OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL_OPENAI", "text-embedding-3-small"))
    elif LLM_PROVIDER == "google":
        return GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL_GOOGLE", "models/embedding-001"))
    else:
        raise ValueError(f"Provider {LLM_PROVIDER} not supported. Use 'openai' or 'google'.")

def ingest_pdf():
    """Reads a PDF, splits it into chunks, and stores them in PGVector."""
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return

    # 1. Load PDF
    print(f"Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    # 2. Split PDF into chunks (1000 chars, 150 overlap)
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(documents)
    
    # 3. Initialize Embeddings and PGVector
    embeddings = get_embeddings()
    collection_name = "pdf_docs"
    
    print(f"Connecting to PostgreSQL and storing vectors (Provider: {LLM_PROVIDER})...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=POSTGRES_URL,
        use_jsonb=True,
    )
    
    # 4. Add documents to vector store
    vector_store.add_documents(docs)
    print(f"Successfully ingested {len(docs)} chunks into PostgreSQL.")

if __name__ == "__main__":
    ingest_pdf()
