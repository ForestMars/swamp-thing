# /src/agents/main_agent.py

from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.core.tools import QueryEngineTool, FunctionTool
import os

# --- Import Tools from Previous Tasks ---
# Note: You would import the actual initialized tools here,
# assuming metadata_query_tool is defined in metadata_tool.py
from .metadata_tool import metadata_query_tool 
from .reranker_agent import reranked_query_tool 

# --- 1. Load Configuration ---
# Uses the LLM model from the global_config.yaml
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-opus-20240229") 

# --- 2. Initialize LLM ---
# This LLM acts as the central brain, deciding which tool to call and when.
llm = Anthropic(
    model=LLM_MODEL, 
    temperature=0.1, 
    # API key is loaded automatically from ANTHROPIC_API_KEY environment variable
) 

# --- 3. Define the Complete Toolset ---
# The order here defines the priority/visibility to the LLM's reasoning loop.
tools = [
    metadata_query_tool, # Agent 1: Executes SQL filter (The starting point)
    reranked_query_tool  # Agent 2/3: Executes filtered vector search and synthesis (The heavy lifting)
]

# --- 4. Create the Orchestrating Agent ---
# We use the ReActAgent which forces the LLM to 'Reason' before 'Act' (calling a tool).
final_orchestrator_agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True, # Set to False in production
    # System prompt is critical to force sequential logic
    system_prompt=(
        "You are an expert Legal RAG Agent. Your goal is to answer questions by "
        "accessing specific document subsets. ALWAYS use the 'metadata_query_tool' FIRST "
        "to filter the document space before calling the 'filtered_semantic_search' tool. "
        "The final answer must be based only on the retrieved information."
    )
)

# --- 5. Execution Example (How the ragAPI.ts would call the agent) ---
def execute_agent_query(user_query: str):
    print(f"\n[ORCHESTRATOR] Starting query for: {user_query}")
    
    # The agent will internally use its tools sequentially based on the system prompt
    response = final_orchestrator_agent.query(user_query)
    
    print("\n[ORCHESTRATOR] Final Answer Received.")
    return str(response)

if __name__ == "__main__":
    # Simulate a complex, filtered query
    test_query = (
        "Of the 100 most recent documents focused on litigation involving asbestos poisoning, "
        "what were the major rulings and what date were they filed?"
    )
    final_result = execute_agent_query(test_query)
    print("\n--- FINAL SYNTHESIS ---")
    print(final_result)
