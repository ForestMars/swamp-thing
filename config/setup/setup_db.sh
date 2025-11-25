#!/bin/bash
# --------------------------------------------------------------------------
# DATABASE SETUP SCRIPT: RAG PGVector Prerequisites
# This script ensures the PostgreSQL database and extensions exist.
#
# NOTE: Replace 'rag_user', 'password', and 'rag_db' with your actual values.
# The user executing this script must have SUPERUSER privileges.
# --------------------------------------------------------------------------

# --- CONFIGURATION (Must match values used in /config/domain_preferences.yaml) ---
DB_USER="rag_user"
DB_PASS="password"
DB_HOST="localhost"
DB_NAME="rag_db"

# --- 1. CREATE DATABASE (If it doesn't exist) ---
echo "1. Checking/Creating database: $DB_NAME"
psql -h $DB_HOST -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
psql -h $DB_HOST -U postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER ENCODING 'UTF8';"

# --- 2. CREATE PGVECTOR EXTENSION ---
# The 'vector' extension MUST be enabled within the target database.
echo "2. Enabling pgvector extension in $DB_NAME..."
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"

# --- 3. EXPLICIT TABLE DEFINITION (FOR CLARITY/VERIFICATION) ---
# NOTE: While LlamaIndex creates this implicitly, we explicitly define the structure
# for maintenance visibility.
echo "3. Creating or verifying table structure: legal_document_vectors"

TABLE_SQL=$(cat <<EOF
CREATE TABLE IF NOT EXISTS legal_document_vectors (
    id VARCHAR(512) PRIMARY KEY,
    text TEXT,
    -- metadata_ stores LlamaIndex Node metadata (JSONB is efficient for unstructured data)
    metadata_ JSONB,
    node_id VARCHAR(512),
    -- The vector column: 768 dimensions for nomic-embed-text (Ollama)
    embedding VECTOR(768)
);
EOF
)

psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "$TABLE_SQL"

echo "Database $DB_NAME is fully configured and ready for LlamaIndex ingestion."
