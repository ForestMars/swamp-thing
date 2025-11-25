# /src/agents/reranker_agent.py

from llama_index.postprocessor.flag_embedding_rerank import FlagEmbeddingReranker
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from .semantic_retriever_agent import index, create_filtered_query_engine # Assumes Task 2.2 is implemented

# --- 1. Define the Reranker Model ---
# This uses the BGE Cross-Encoder model, which is highly effective for scoring relevance.
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-base", # High-performance open-source cross-encoder
    top_n=5, # CRITICAL: Only the top 5 chunks will be passed to the final Claude 3 Opus call
)

def create_reranked_query_engine(doc_ids: list[str], query_str: str):
    """
    Combines the semantic retriever with the reranker post-processor to execute 
    a highly precise, filtered vector search.
    """
    if not doc_ids:
        # Fails gracefully if Agent 1 (Metadata Filter) returned no matching documents
        return "No documents found matching the strict metadata criteria."

    # 1. Get the retriever with the Metadata Filter applied (Task 2.2 logic)
    # We retrieve a wider net (e.g., top 50) before re-ranking.
    retriever = index.as_retriever(
        similarity_top_k=50, 
        # Apply the filter logic from the previous step (Task 2.2) to restrict the search
        filters=create_filtered_query_engine(doc_ids).filters 
    )

    # 2. Build the Query Engine with the Reranker
    reranked_query_engine = RetrieverQueryEngine(
        retriever=retriever,
        # The reranker runs after vector search and before the final LLM call
        node_postprocessors=[reranker], 
    )
    
    # 3. Execute the final, high-precision query
    response = reranked_query_engine.query(query_str)
    return str(response)

# --- 3. Convert to Agent Tool ---
# This tool is the final RAG step, ready to be called by the orchestrating Agent (main_agent.py).
reranked_query_tool = QueryEngineTool.from_defaults(
    query_engine=create_reranked_query_engine([], ""), # Initialize with dummy data for Tool creation
    name="filtered_semantic_search",
    description=(
        "Use this tool ONLY AFTER the metadata filter has provided specific document IDs. "
        "It performs a high-precision vector search on the filtered subset of legal documents "
        "and synthesizes the final answer."
    ),
)
