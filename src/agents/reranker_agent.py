# /src/agents/reranker_agent.py (Refactored for llama-index 0.14.8)

from llama_index.postprocessor.flag_embedding_rerank import FlagEmbeddingReranker
from llama_index.core.tools import QueryEngineTool
from .semantic_retriever_agent import index, create_filtered_query_engine

# --- 1. Reranker ---
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-base",
    top_n=5,
)

def create_reranked_query_engine(doc_ids: list[str], query_str: str):
    if not doc_ids:
        return "No documents found matching the strict metadata criteria."

    # 1. Apply filtered retriever
    retriever = index.as_retriever(
        similarity_top_k=50,
        filters=create_filtered_query_engine(doc_ids).filters
    )

    # 2. Build Query Engine with reranker
    reranked_query_engine = retriever.as_query_engine(
        node_postprocessors=[reranker],
        verbose=True
    )

    # 3. Execute query
    return reranked_query_engine.query(query_str)

# --- 3. Wrap as Tool ---
reranked_query_tool = QueryEngineTool.from_defaults(
    query_engine=create_reranked_query_engine([], ""),  # dummy initialization
    name="filtered_semantic_search",
    description=(
        "Use this tool ONLY AFTER the metadata filter has provided specific document IDs. "
        "It performs a high-precision vector search on the filtered subset of legal documents "
        "and synthesizes the final answer."
    ),
)
