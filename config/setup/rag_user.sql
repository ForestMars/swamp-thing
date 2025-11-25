-- /config/db_setup/create_rag_user.sql
-- Run this script using a superuser account (e.g., psql -U postgres -f create_rag_user.sql)

-- --- CONFIGURATION ---
-- NOTE: Replace 'rag_user' and 'SuperSecretRAGPassword' with production values
\set RAG_USER 'rag_user'
\set RAG_PASS 'wrinklepants'
\set RAG_DB 'rag_db'

-- 1. CREATE THE USER/ROLE
-- 'NOCREATEDB' is the default and prevents the RAG user from making *other* databases,
-- but we grant it back specifically for the rag_db creation in step 3.
CREATE USER :RAG_USER WITH ENCRYPTED PASSWORD :'RAG_PASS' LOGIN;


-- 2. CREATE THE DATABASE (The RAG user is set as the owner)
-- The owner automatically has ALL privileges on the database.
CREATE DATABASE :RAG_DB OWNER :RAG_USER ENCODING 'UTF8';


-- 3. GRANT CONNECT AND PRIVILEGES
-- Grant explicit permissions on the database and future objects (best practice for security)
GRANT CONNECT ON DATABASE :RAG_DB TO :RAG_USER;
GRANT ALL PRIVILEGES ON DATABASE :RAG_DB TO :RAG_USER;

-- This allows the user to create the 'vector' extension and other necessary objects
ALTER USER :RAG_USER CREATEDB;


-- 4. APPLY DEFAULT PRIVILEGES (Ensures future tables/sequences are owned by the RAG user)
-- This is critical for tables created *after* the initial run.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO :RAG_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO :RAG_USER;


-- Confirmation of successful execution:
SELECT 'User ' || :'RAG_USER' || ' created with password and owner of ' || :'RAG_DB' || ' database.' AS status;
