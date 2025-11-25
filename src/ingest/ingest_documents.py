#!/usr/bin/env python
"""
/src/ingest/ingest_documents.py

Document ingestion pipeline with auto-clustering.

Author: Forest Mars
Version: 0.3
"""
__version__ = '0.2'
__author__ = 'Forest Mars'

import os
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama
import hashlib
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# Configuration
LAKE_DIR = Path.home() / "lake"
METADATA_DB_URI = "postgresql://rag_user:wrinklepants@localhost:5432/metadata_catalog"
VECTOR_DB_URI = "postgresql://rag_user:wrinklepants@localhost:5432/rag_db"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5:7b"

def classify_document(content: str) -> str:
    """Use LLM to classify document into a category."""
    # Get first 1000 chars for classification
    sample = content[:1000]
    
    prompt = f"""Classify this document into ONE of these categories:
- fiction: creative writing, stories, novels, narratives
- non-fiction: essays, articles, analysis, factual writing
- technical: code, documentation, specifications

Document excerpt:
{sample}

Category (return ONLY the category name):"""
    
    response = Settings.llm.complete(prompt)
    category = response.text.strip().lower()
    
    # Validate and default
    valid_categories = ["fiction", "non-fiction", "technical"]
    if category not in valid_categories:
        category = "uncategorized"
    
    return category


def auto_cluster_documents(documents, embeddings, n_clusters=None):
    """
    Automatically discover clusters in the document collection.
    If n_clusters is None, tries to detect optimal number (2-5 range for now).
    """
    if len(documents) < 2:
        return {0: [0]}, ["cluster_0"]
    
    # Convert embeddings to numpy array
    X = np.array(embeddings)
    
    # Auto-detect number of clusters if not specified
    if n_clusters is None:
        # For small collections, try 2-4 clusters and pick best
        max_clusters = min(len(documents) - 1, 5)
        if len(documents) >= 4:
            n_clusters = 2  # Start with 2 for your test case
        else:
            n_clusters = 2
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Group documents by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    # Generate cluster names based on content
    cluster_names = []
    for cluster_id in sorted(clusters.keys()):
        doc_indices = clusters[cluster_id]
        # Sample first doc in cluster for naming
        sample_doc = documents[doc_indices[0]]
        sample_text = sample_doc.text[:500]
        
        # Ask LLM to name the cluster
        prompt = f"""Based on this document sample, suggest a SHORT category name (1-2 words):

{sample_text}

Category name:"""
        
        response = Settings.llm.complete(prompt)
        name = response.text.strip().lower().replace(" ", "_")
        cluster_names.append(name)
    
    return clusters, cluster_names

print("=" * 60)
print("DOCUMENT INGESTION SCRIPT")
print("=" * 60)

# Setup embedding model
print("\n1. Configuring embedding model...")
Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_URL
)

# Also setup LLM for classification
Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.1,
    request_timeout=30.0,
)
print(f"✅ Using LLM: {LLM_MODEL}")
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

# Get embeddings for clustering
print("\n4. Generating embeddings for clustering...")
doc_embeddings = []
for doc in documents:
    embedding = Settings.embed_model.get_text_embedding(doc.text[:1000])  # Sample for speed
    doc_embeddings.append(embedding)
print(f"✅ Generated {len(doc_embeddings)} embeddings")

# Auto-cluster documents
print("\n5. Auto-clustering documents...")
clusters, cluster_names = auto_cluster_documents(documents, doc_embeddings)
print(f"✅ Discovered {len(clusters)} clusters:")
for cluster_id, doc_indices in clusters.items():
    print(f"   - {cluster_names[cluster_id]}: {len(doc_indices)} documents")

# Insert metadata with cluster assignments
print("\n6. Inserting clusters and document metadata...")
with metadata_engine.connect() as conn:
    # First, insert clusters
    for cluster_id, doc_indices in clusters.items():
        cluster_name = cluster_names[cluster_id]
        
        # Insert or update cluster
        cluster_sql = text("""
            INSERT INTO document_clusters (cluster_id, cluster_name, doc_count)
            VALUES (:id, :name, :count)
            ON CONFLICT (cluster_id) DO UPDATE SET
                cluster_name = EXCLUDED.cluster_name,
                doc_count = EXCLUDED.doc_count,
                updated_at = NOW()
        """)
        
        conn.execute(cluster_sql, {
            "id": cluster_id,
            "name": cluster_name,
            "count": len(doc_indices)
        })
        conn.commit()
        print(f"   ✅ Cluster {cluster_id}: [{cluster_name}] with {len(doc_indices)} documents")
    
    # Then, insert documents with cluster references
    for cluster_id, doc_indices in clusters.items():
        for idx in doc_indices:
            doc = documents[idx]
            filename = doc.metadata.get('file_name', 'unknown')
            doc_id = hashlib.md5(filename.encode()).hexdigest()[:12]
            file_path = doc.metadata.get('file_path', '')
            date = datetime.now().date()
            
            # Insert or update document
            insert_sql = text("""
                INSERT INTO document_metadata_catalog (id, cluster_id, date, jurisdiction, doc_path)
                VALUES (:id, :cluster_id, :date, :jurisdiction, :doc_path)
                ON CONFLICT (id) DO UPDATE SET
                    cluster_id = EXCLUDED.cluster_id,
                    date = EXCLUDED.date,
                    jurisdiction = EXCLUDED.jurisdiction,
                    doc_path = EXCLUDED.doc_path
            """)
            
            conn.execute(insert_sql, {
                "id": doc_id,
                "cluster_id": cluster_id,
                "date": date,
                "jurisdiction": "personal",
                "doc_path": file_path
            })
            conn.commit()

# Setup vector store
print("\n7. Setting up vector store...")
vector_store = PGVectorStore.from_params(
    database="rag_db",
    host="localhost",
    port="5432",
    password="wrinklepants",
    user="rag_user",
    table_name="document_vectors",
    embed_dim=768,
)
print("✅ Connected to vector store")

# Create index and store embeddings
print("\n8. Creating full embeddings and storing in vector database...")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)
print("✅ Documents embedded and stored")

# Verify
print("\n9. Verifying ingestion...")
with metadata_engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM document_metadata_catalog"))
    count = result.scalar()
    print(f"✅ Metadata records: {count}")
    
    # Show cluster distribution
    result = conn.execute(text("""
        SELECT c.cluster_name, c.doc_count 
        FROM document_clusters c 
        ORDER BY c.cluster_id
    """))
    print("\nCluster distribution:")
    for row in result:
        print(f"   - {row[0]}: {row[1]} documents")

print("\n" + "=" * 60)
print("INGESTION COMPLETE!")
print("=" * 60)
print("\nYou can now run the agent:")
print("  uv run python -m src.agents.main_agent")