# /src/agents/main_agent.py

import logging
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import QueryEngineTool, FunctionTool
import os

# Try different agent imports based on what's available
try:
    from llama_index.core.agent import ReActAgent
    AGENT_TYPE = "ReActAgent"
except (ImportError, AttributeError):
    try:
        from llama_index.core.agent import AgentRunner, ReActAgentWorker
        AGENT_TYPE = "AgentRunner"
    except ImportError:
        from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
        AGENT_TYPE = "FunctionCalling"

logging.basicConfig(level=logging.DEBUG)
Settings.debug = True

from .metadata_tool import metadata_query_tool
from .reranker_agent import reranked_query_tool

# Use local Ollama models
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:latest")  # or "gemma2:latest"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    request_timeout=120.0,
)

tools = [
    metadata_query_tool,
    reranked_query_tool
]

system_prompt = (
    "You are an expert Legal RAG Agent. Your goal is to answer questions by "
    "accessing specific document subsets. ALWAYS use the 'metadata_query_tool' FIRST "
    "to filter the document space before calling the 'filtered_semantic_search' tool. "
    "The final answer must be based only on the retrieved information."
)

# Create agent based on what's available
if AGENT_TYPE == "ReActAgent":
    try:
        # Try the modern API
        final_orchestrator_agent = ReActAgent(
            tools=tools,
            llm=llm,
            verbose=True,
        )
        # Set system prompt if possible
        if hasattr(final_orchestrator_agent, 'system_prompt'):
            final_orchestrator_agent.system_prompt = system_prompt
    except TypeError:
        # Fallback: create with AgentRunner
        from llama_index.core.agent import AgentRunner, ReActAgentWorker
        agent_worker = ReActAgentWorker.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
        )
        final_orchestrator_agent = AgentRunner(agent_worker)

elif AGENT_TYPE == "AgentRunner":
    agent_worker = ReActAgentWorker.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
    )
    final_orchestrator_agent = AgentRunner(agent_worker)

else:  # FunctionCalling
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
        system_prompt=system_prompt,
    )
    final_orchestrator_agent = AgentRunner(agent_worker)

print(f"[INFO] Using agent type: {AGENT_TYPE}")


def execute_agent_query(user_query: str):
    """
    Execute a query through the orchestrator agent.
    
    Args:
        user_query: User's natural language question
        
    Returns:
        Agent's response as a string
    """
    print(f"\n[ORCHESTRATOR] Starting query for: {user_query}")
    
    try:
        # Try chat method first (modern API)
        if hasattr(final_orchestrator_agent, 'chat'):
            response = final_orchestrator_agent.chat(user_query)
        # Fallback to query method
        elif hasattr(final_orchestrator_agent, 'query'):
            response = final_orchestrator_agent.query(user_query)
        else:
            raise AttributeError("Agent has neither 'chat' nor 'query' method")
        
        print("\n[ORCHESTRATOR] Final Answer Received.")
        return str(response)
    
    except Exception as e:
        print(f"\n[ERROR] Agent execution failed: {e}")
        raise


if __name__ == "__main__":
    test_query = (
        "Of the 100 most recent documents focused on litigation involving asbestos poisoning, "
        "what were the major rulings and what date were they filed?"
    )
    
    try:
        final_result = execute_agent_query(test_query)
        print("\n--- FINAL SYNTHESIS ---")
        print(final_result)
    except Exception as e:
        print(f"\n--- EXECUTION FAILED ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()