# /src/agents/main_agent.py

import logging
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.core.tools import QueryEngineTool, FunctionTool
import os

logging.basicConfig(level=logging.DEBUG)
Settings.debug = True

from .metadata_tool import metadata_query_tool
from .reranker_agent import reranked_query_tool

LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-opus-20240229")

llm = Anthropic(
    model=LLM_MODEL,
    temperature=0.1,
)

tools = [
    metadata_query_tool,
    reranked_query_tool
]

final_orchestrator_agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are an expert Legal RAG Agent. Your goal is to answer questions by "
        "accessing specific document subsets. ALWAYS use the 'metadata_query_tool' FIRST "
        "to filter the document space before calling the 'filtered_semantic_search' tool. "
        "The final answer must be based only on the retrieved information."
    )
)

def execute_agent_query(user_query: str):
    print(f"\n[ORCHESTRATOR] Starting query for: {user_query}")
    response = final_orchestrator_agent.query(user_query)
    print("\n[ORCHESTRATOR] Final Answer Received.")
    return str(response)

if __name__ == "__main__":
    test_query = (
        "Of the 100 most recent documents focused on litigation involving asbestos poisoning, "
        "what were the major rulings and what date were they filed?"
    )
    final_result = execute_agent_query(test_query)
    print("\n--- FINAL SYNTHESIS ---")
    print(final_result)
