# /src/agents/main_agent.py

import logging
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.core.tools import QueryEngineTool, FunctionTool
import os

# --- 0. Set Global Debugging and Verbosity ---
# CRITICAL for seeing SQL queries, vector search operations, and ReAct steps.
logging.basicConfig(level=logging.DEBUG)
Settings.debug = True 

# --- Import Tools from Previous Tasks ---
# These are the actual working tools that the agent will decide to use.
# NOTE: Ensure metadata_tool.py and reranker_agent.py are implemented and accessible.
from .metadata_tool import metadata_query_tool 
from .reranker_agent import reranked_query_tool 

# --- 1. Load Configuration ---
# Uses the LLM model from the global_config.yaml
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-opus-20240229") 

# --- 2. Initialize LLM (The Decision Brain) ---
llm = Anthropic(
    model=LLM_MODEL, 
    temperature=0.1, 
    # API key is loaded automatically from ANTHROPIC_API_KEY environment variable
) 

# --- 3. Define the Complete Toolset ---
tools = [
    metadata_query_tool, # Agent 1: Executes SQL filter (The starting point)
    reranked_query_tool  # Agent 2/3: Executes filtered vector search and synthesis (The heavy lifting)
]

# --- 4. Create the Orchestrating Agent ---
final_orchestrator_agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True, # <-- Max Verbosity for Agent Thought/Action steps
    system_prompt=(
        "You are an expert Legal RAG Agent. Your goal is to answer questions by "
        "accessing specific document subsets. ALWAYS use the 'metadata_query_tool' FIRST "
        "to filter the document space before calling the 'filtered_semantic_search' tool. "
        "The final answer must be based only on the retrieved information."
    )
)

# --- 5. Execution Function ---
def execute_agent_query(user_query: str):
    print(f"\n[ORCHESTRATOR] Starting query for: {user_query}")
    
    # The agent will internally use its tools sequentially based on the system prompt
    response = final_orchestrator_agent.query(user_query)
    
    print("\n[ORCHESTRATOR] Final Answer Received.")
    return str(response)

if __name__ == "__main__":
    # Simulate a complex, filtered query to test the full chain
    test_query = (
        "Of the 100 most recent documents focused on litigation involving asbestos poisoning, "
        "what were the major rulings and what date were they filed?"
    )
    
    # Ensure all components (DB, Ollama) are running before executing this.
    final_result = execute_agent_query(test_query)
    
    print("\n--- FINAL SYNTHESIS ---")
    print(final_result)
