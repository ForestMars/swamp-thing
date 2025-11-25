#!/bin/bash
# --------------------------------------------------------------------------
# DATABASE SETUP SCRIPT: RAG PGVector Prerequisites
# --------------------------------------------------------------------------

CURRENT_USER=$(whoami)
echo "Current OS User: $CURRENT_USER"

# --- CONFIGURATION ---
DB_USER="rag_user"
DB_HOST="localhost"
RAG_DB="rag_db"
METADATA_DB="metadata_catalog"

# --- 1. CREATE RAG_DB ---
echo "1. Checking/Creating database: $RAG_DB"
psql -h $DB_HOST -U $CURRENT_USER -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$RAG_DB'" | grep -q 1 || \
psql -h $DB_HOST -U $CURRENT_USER -d postgres -c "CREATE DATABASE $RAG_DB OWNER $DB_USER ENCODING 'UTF8';"

# --- 2. CREATE METADATA_DB ---
echo "2. Checking/Creating database: $METADATA_DB"
psql -h $DB_HOST -U $CURRENT_USER -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$METADATA_DB'" | grep -q 1 || \
psql -h $DB_HOST -U $CURRENT_USER -d postgres -c "CREATE DATABASE $METADATA_DB OWNER $DB_USER ENCODING 'UTF8';"

# --- 3. ENABLE PGVECTOR ---
echo "3. Enabling pgvector extension in $RAG_DB..."
psql -h $DB_HOST -U $CURRENT_USER -d $RAG_DB -c "CREATE EXTENSION IF NOT EXISTS vector;"

# --- 4. CREATE VECTOR TABLE ---
echo "4. Creating document_vectors table in $RAG_DB..."
TABLE_SQL=$(cat <<EOF
CREATE TABLE IF NOT EXISTS document_vectors (
    id VARCHAR(512) PRIMARY KEY,
    text TEXT,
    metadata_ JSONB,
    node_id VARCHAR(512),
    embedding VECTOR(768)
);
EOF
)

psql -h $DB_HOST -U $DB_USER -d $RAG_DB -c "$TABLE_SQL"

# --- 5. CREATE METADATA TABLE ---
echo "5. Creating document_metadata_catalog table in $METADATA_DB..."
psql -h $DB_HOST -U $DB_USER -d $METADATA_DB << 'EOF'
CREATE TABLE IF NOT EXISTS document_metadata_catalog (
    id TEXT PRIMARY KEY,
    cluster_id INTEGER,
    date DATE,
    jurisdiction TEXT,
    doc_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_clusters (
    cluster_id SERIAL PRIMARY KEY,
    cluster_name TEXT NOT NULL,
    doc_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_cluster ON document_metadata_catalog(cluster_id);
EOF

echo ""
echo "✅ Database $RAG_DB and $METADATA_DB fully configured"