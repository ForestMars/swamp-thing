#!/usr/bin/env python
"""
Document ingestion script: Load documents from ~/lake into both databases.
Run with: uv run python ingest_documents.py
"""

import os
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
import hashlib

# Configuration
LAKE_DIR = Path.home() / "lake"
METADATA_DB_URI = "postgresql://rag_user:wrinklepants@localhost:5432/metadata_catalog"
VECTOR_DB_URI = "postgresql://rag_user:wrinklepants@localhost:5432/rag_db"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"

print("=" * 60)
print("DOCUMENT INGESTION SCRIPT")
print("=" * 60)

# Setup embedding model
print("\n1. Configuring embedding model...")
Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_URL
)
print(f"✅ Using embeddings: {EMBED_MODEL}")

# Load documents
print(f"\n2. Loading documents from {LAKE_DIR}...")
documents = SimpleDirectoryReader(str(LAKE_DIR)).load_data()
print(f"✅ Loaded {len(documents)} documents")

for i, doc in enumerate(documents):
    print(f"   - Document {i+1}: {doc.metadata.get('file_name', 'unknown')}")

# Connect to metadata database
print("\n3. Connecting to metadata database...")
metadata_engine = create_engine(METADATA_DB_URI)
print("✅ Connected to metadata_catalog")

# Insert metadata
print("\n4. Inserting document metadata...")
with metadata_engine.connect() as conn:
    for doc in documents:
        # Generate document ID from filename
        filename = doc.metadata.get('file_name', 'unknown')
        doc_id = hashlib.md5(filename.encode()).hexdigest()[:12]
        
        # Extract metadata (you can customize this based on your needs)
        file_path = doc.metadata.get('file_path', '')
        
        # For this example, we'll use simple metadata
        # You can enhance this to parse actual metadata from document content
        topic = "general writing"  # Could be extracted from content
        date = datetime.now().date()
        jurisdiction = "personal"
        
        # Insert or update
        insert_sql = text("""
            INSERT INTO document_metadata_catalog (id, topic, date, jurisdiction, doc_path)
            VALUES (:id, :topic, :date, :jurisdiction, :doc_path)
            ON CONFLICT (id) DO UPDATE SET
                topic = EXCLUDED.topic,
                date = EXCLUDED.date,
                jurisdiction = EXCLUDED.jurisdiction,
                doc_path = EXCLUDED.doc_path
        """)
        
        conn.execute(insert_sql, {
            "id": doc_id,
            "topic": topic,
            "date": date,
            "jurisdiction": jurisdiction,
            "doc_path": file_path
        })
        conn.commit()
        
        print(f"   ✅ Inserted metadata for: {filename} (ID: {doc_id})")

# Setup vector store
print("\n5. Setting up vector store...")
vector_store = PGVectorStore.from_params(
    database="rag_db",
    host="localhost",
    password="wrinklepants",
    user="rag_user",
    table_name="document_vectors",  # Fixed: use generic name
    embed_dim=768,
)
print("✅ Connected to vector store")

# Create index and store embeddings
print("\n6. Creating embeddings and storing in vector database...")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)
print("✅ Documents embedded and stored")

# Verify
print("\n7. Verifying ingestion...")
with metadata_engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM document_metadata_catalog"))
    count = result.scalar()
    print(f"✅ Metadata records: {count}")

print("\n" + "=" * 60)
print("INGESTION COMPLETE!")
print("=" * 60)
print("\nYou can now run the agent:")
print("  uv run python -m src.agents.main_agent")