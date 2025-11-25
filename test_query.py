#!/usr/bin/env python
"""
Quick test script to verify data is in the database and queryable.
Run with: uv run python test_query.py
"""

from pathlib import Path
from sqlalchemy import create_engine, text
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore

# Configuration
METADATA_DB_URI = "postgresql://rag_user:wrinklepants@localhost:5432/metadata_catalog"
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text:latest"

print("=" * 60)
print("DATABASE & QUERY TEST")
print("=" * 60)

# 1. Check metadata database
print("\n1. Checking metadata_catalog...")
metadata_engine = create_engine(METADATA_DB_URI)
with metadata_engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM document_metadata_catalog"))
    count = result.scalar()
    print(f"✅ Found {count} documents in metadata catalog")
    
    if count > 0:
        result = conn.execute(text("SELECT id, topic, doc_path FROM document_metadata_catalog LIMIT 5"))
        for row in result:
            print(f"   - {row[0]}: {row[1]} ({row[2]})")

# 2. Check vector database
print("\n2. Checking vector store...")
vector_store = PGVectorStore.from_params(
    database="rag_db",
    host="localhost",
    port="5432",
    password="wrinklepants",
    user="rag_user",
    table_name="document_vectors",
    embed_dim=768,
)

# Count vectors - use separate engine
vector_engine = create_engine("postgresql://rag_user:wrinklepants@localhost:5432/rag_db")
with vector_engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM document_vectors"))
    count = result.scalar()
    print(f"✅ Found {count} vectors in database")

# 3. Setup query engine
print("\n3. Setting up query engine...")
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

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()
print("✅ Query engine ready")

# 4. Test query
print("\n4. Running test query...")
print("-" * 60)
test_query = "What is this document about?"
print(f"Query: {test_query}\n")

response = query_engine.query(test_query)
print(f"Answer: {response}\n")
print("-" * 60)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("✅ Data is in the database and queryable")
print("\nYou can now use your playground to query this data.")