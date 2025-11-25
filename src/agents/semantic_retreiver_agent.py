# /src/agents/semantic_retriever_agent.py

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import MetadataFilter, FilterOperator, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.anthropic import Anthropic
import os

# --- 1. CONFIGURATION LOADING ---
# NOTE: These values are loaded from environment variables sourced from global_config/domain_preferences
POSTGRES_DB_URI = os.getenv("VECTOR_DB_URI", "postgresql+psycopg2://rag_user:SuperSecretRAGPassword@localhost:5432/rag_db")
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "legal_document_vectors")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768)) # Dimension for nomic-embed-text
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-svc.internal.svc:11434")

# --- 2. Initialize Core LlamaIndex Services (Used by the Retriever) ---

# Set the embedding model (Ollama)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_URL)
# Set the LLM (Anthropic) for basic query engine functions (though main_agent.py controls synthesis)
Settings.llm = Anthropic(model="claude-3-opus-20240229") 
Settings.chunk_size = 512 # From global_config

# --- 3. Connect to the Existing Vector Store ---

try:
    vector_store = PGVectorStore.from_params(
        database="rag_db",
        host="localhost", # Should be dynamically loaded from environment in production
        password="SuperSecretRAGPassword",
        user="rag_user",
        table_name=VECTOR_TABLE,
        embed_dim=EMBEDDING_DIM,
    )
except Exception as e:
    print(f"ERROR: Could not connect to PGVectorStore: {e}")
    # Exit gracefully or raise an error if the database connection fails
    vector_store = None 

# Load the existing index from the vector store
if vector_store:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
else:
    # Create a dummy index if connection fails to allow import
    index = None 


# --- 4. Retrieval Function (The core logic for Agent 2) ---

def create_filtered_query_engine(doc_ids: list[str], query_str: str = ""):
    """
    Creates and executes a query engine that restricts the vector search
    to ONLY the documents specified by the list of doc_ids (the output of Agent 1).
    This function will be used by the Re-Ranker Agent (Task 2.3).
    """
    if not index or not doc_ids:
        # Fails gracefully if no index is loaded or no documents are provided
        return None

    # 1. DEFINE THE METADATA FILTER (The critical Agentic step)
    # The 'doc_id' field must match the key used during ingestion.
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="doc_id", 
                value=doc_ids,
                operator=FilterOperator.IN,
            )
        ]
    )

    # 2. CREATE THE RETRIEVER
    retriever = index.as_retriever(
        similarity_top_k=50, # Retrieve a wide net for the Reranker
        filters=filters # Apply the filter to constrain the search space
    )
    
    # 3. Return the retriever object. It is used by the Re-Ranker Agent (Task 2.3)
    return retriever


# NOTE: This file focuses on the RETRIEVER. The QueryEngine and final execution 
# are handled by the Re-Ranker Agent (Task 2.3) to ensure the top-k results are scored.
