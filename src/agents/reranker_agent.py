# /src/agents/reranker_agent.py (Updated for modern llama-index)

from llama_index.core.tools import QueryEngineTool
from llama_index.core.postprocessor import SimilarityPostprocessor
from .semantic_retriever_agent import index, create_filtered_query_engine

# --- 1. Reranker ---
# Note: FlagEmbeddingReranker requires a separate package installation
# For now, using SimilarityPostprocessor as a fallback
# To use FlagEmbeddingReranker, run: uv add llama-index-postprocessor-flag-embedding-reranker

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    
    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-base",
        top_n=5,
    )
except ImportError:
    print("Warning: FlagEmbeddingReranker not available. Using SimilarityPostprocessor instead.")
    print("To install: uv add llama-index-postprocessor-flag-embedding-reranker")
    
    # Fallback to similarity-based reranking
    reranker = SimilarityPostprocessor(similarity_cutoff=0.7)


def create_reranked_query_engine(doc_ids: list[str], query_str: str):
    """
    Create a query engine with metadata filtering and reranking.
    
    Args:
        doc_ids: List of document IDs from metadata filtering
        query_str: User's query string
        
    Returns:
        Query response with reranked results
    """
    if not doc_ids:
        return "No documents found matching the strict metadata criteria."

    # 1. Get the filtered query engine
    filtered_engine = create_filtered_query_engine(doc_ids)
    
    # 2. Create retriever from index with filters
    retriever = index.as_retriever(
        similarity_top_k=50,
        filters=filtered_engine.retriever.get_filters() if hasattr(filtered_engine, 'retriever') else None
    )

    # 3. Build Query Engine with reranker
    reranked_query_engine = index.as_query_engine(
        retriever=retriever,
        node_postprocessors=[reranker],
        verbose=True
    )

    # 4. Execute query
    response = reranked_query_engine.query(query_str)
    return response


# --- 3. Wrap as Tool ---
# Note: We'll create the actual query engine when the tool is called
def reranked_query_wrapper(doc_ids: list[str], query_str: str) -> str:
    """Wrapper function that can be called by the agent."""
    response = create_reranked_query_engine(doc_ids, query_str)
    return str(response)


reranked_query_tool = QueryEngineTool.from_defaults(
    query_engine=None,  # Will be created dynamically
    name="filtered_semantic_search",
    description=(
        "Use this tool ONLY AFTER the metadata filter has provided specific document IDs. "
        "It performs a high-precision vector search on the filtered subset of legal documents "
        "and synthesizes the final answer. "
        "Requires two inputs: doc_ids (list of document IDs) and query_str (user's question)."
    ),
)