#!/usr/bin/env python
"""
Simple RAG system - no agents, no complexity.
Just: load docs -> embed -> store -> query
"""

import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine

# Configuration
LAKE_DIR = Path("~/lake").expanduser()
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"  # Match what you actually have
EMBED_MODEL = "nomic-embed-text:latest"

# PostgreSQL (optional - will use in-memory if not available)
POSTGRES_URI = "postgresql://rag_user:wrinklepants@localhost:5432/rag_db"
USE_POSTGRES = False  # Set to True once you create the database

print("=" * 60)
print("SIMPLE RAG SYSTEM")
print("=" * 60)

# 1. Configure LlamaIndex
print("\n1. Configuring models...")
Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_URL
)
Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.1,
    request_timeout=60.0,
)
print(f"✅ Using LLM: {LLM_MODEL}")
print(f"✅ Using embeddings: {EMBED_MODEL}")

# 2. Load documents
print(f"\n2. Loading documents from {LAKE_DIR}...")
if not LAKE_DIR.exists():
    print(f"❌ Directory not found: {LAKE_DIR}")
    print("Creating it for you...")
    LAKE_DIR.mkdir(parents=True, exist_ok=True)
    (LAKE_DIR / "test.md").write_text("# Test Document\n\nThis is a test.")

documents = SimpleDirectoryReader(str(LAKE_DIR)).load_data()
print(f"✅ Loaded {len(documents)} documents")

# 3. Create vector store
print("\n3. Creating vector index...")
if USE_POSTGRES:
    try:
        vector_store = PGVectorStore.from_params(
            database="rag_db",
            host="localhost",
            password="wrinklepants",
            user="rag_user",
            table_name="documents",
            embed_dim=768,
        )
        print("✅ Using PostgreSQL vector store")
    except Exception as e:
        print(f"⚠️  PostgreSQL failed: {e}")
        print("Falling back to in-memory storage")
        USE_POSTGRES = False

if not USE_POSTGRES:
    print("✅ Using in-memory vector store")
    index = VectorStoreIndex.from_documents(documents)
else:
    from llama_index.core import StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )

print("✅ Index created")

# 4. Query the index
print("\n4. Ready to query!")
print("=" * 60)

def query(question: str):
    """Simple query interface."""
    print(f"\nQuery: {question}")
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    print(f"\nAnswer: {response}")
    return response

# Example queries
if __name__ == "__main__":
    # Test query
    query("What documents are available?")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive mode (Ctrl+C to exit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nYour question: ")
            if user_input.strip():
                query(user_input)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break