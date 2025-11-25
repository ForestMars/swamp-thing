#!/usr/bin/env python
"""
/scripts/test_query.py

Quick test script to verify data is in the database and queryable.
Used for smoke testing and basic validation.

Author: Forest Mars
Version: 0.2

Run with: 
  uv run python scripts/test_query.py                          # Query all docs
  uv run python scripts/test_query.py --doc <doc_id> <query>  # Query specific doc
  uv run python scripts/test_query.py "your question"          # Query all docs
"""
__version__ = '0.2'
__author__ = 'Forest Mars'

import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
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
        print("\nAvailable documents:")
        result = conn.execute(text("SELECT id, topic, doc_path FROM document_metadata_catalog"))
        for row in result:
            filename = Path(row[2]).name if row[2] else "unknown"
            print(f"   [{row[0]}] {filename} - {row[1]}")
    else:
        print("⚠️  No documents found. Run: uv run python ingest_documents.py")
        sys.exit(1)

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
print("✅ Query engine ready")

# 4. Test query
print("\n4. Running test query...")
print("-" * 60)

# Parse arguments
doc_id = None
query_text = None

if len(sys.argv) > 1:
    if sys.argv[1] == "--doc" and len(sys.argv) > 3:
        doc_id = sys.argv[2]
        query_text = " ".join(sys.argv[3:])
    else:
        query_text = " ".join(sys.argv[1:])
else:
    query_text = "What documents are available and what are they about?"

# Create query engine with optional document filter
if doc_id:
    print(f"Filtering to document: {doc_id}")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="doc_id",
                value=doc_id,
                operator=FilterOperator.EQ,
            )
        ]
    )
    query_engine = index.as_query_engine(filters=filters)
else:
    query_engine = index.as_query_engine()

print(f"Query: {query_text}\n")

response = query_engine.query(query_text)
print(f"Answer: {response}\n")
print("-" * 60)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("✅ Data is in the database and queryable")
print("\nUsage:")
print("  uv run python test_query.py                    # General query")
print("  uv run python test_query.py 'your question'    # Specific question")