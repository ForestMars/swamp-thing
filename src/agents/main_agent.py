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
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")  # Match actual model name
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
        # Try creating ReActAgent with AgentRunner pattern
        from llama_index.core.agent import AgentRunner, ReActAgentWorker
        
        agent_worker = ReActAgentWorker.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
        )
        final_orchestrator_agent = AgentRunner(agent_worker)
        print("[INFO] Created ReActAgent using AgentRunner pattern")
        
    except Exception as e:
        print(f"[WARNING] Could not create ReActAgent: {e}")
        # Fallback to direct ReActAgent instantiation
        try:
            final_orchestrator_agent = ReActAgent(
                tools=tools,
                llm=llm,
                verbose=True,
            )
        except Exception as e2:
            print(f"[ERROR] Both ReActAgent methods failed: {e2}")
            raise

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


async def execute_agent_query_async(user_query: str, timeout: float = 60.0):
    """
    Execute a query through the orchestrator agent (async version).
    
    Args:
        user_query: User's natural language question
        timeout: Maximum time to wait for response (seconds)
        
    Returns:
        Agent's response as a string
    """
    import asyncio
    
    print(f"\n[ORCHESTRATOR] Starting query for: {user_query}")
    print(f"[ORCHESTRATOR] Timeout set to {timeout} seconds")
    
    try:
        print("[DEBUG] Running agent as workflow...")
        
        # Run with timeout
        async def run_with_timeout():
            # The agent needs 'user_msg' parameter, not 'input'
            handler = final_orchestrator_agent.run(user_msg=user_query)
            
            # Collect all events
            result = None
            async for event in handler.stream_events():
                event_name = type(event).__name__
                print(f"[DEBUG] Event: {event_name}")
                
                # Look for the final result
                if hasattr(event, 'response'):
                    result = event.response
                    print(f"[DEBUG] Found response: {str(result)[:100]}...")
                elif hasattr(event, 'output'):
                    result = event.output
                    print(f"[DEBUG] Found output: {str(result)[:100]}...")
                elif hasattr(event, 'msg'):
                    result = event.msg
                    print(f"[DEBUG] Found msg: {str(result)[:100]}...")
            
            if result is None:
                # If streaming didn't work, try getting the final result directly
                result = await handler
            
            return result
        
        result = await asyncio.wait_for(run_with_timeout(), timeout=timeout)
        
        print("\n[ORCHESTRATOR] Final Answer Received.")
        return str(result)
    
    except asyncio.TimeoutError:
        print(f"\n[ERROR] Agent execution timed out after {timeout} seconds")
        raise TimeoutError(f"Agent did not complete within {timeout} seconds")
    
    except Exception as e:
        print(f"\n[ERROR] Agent execution failed: {e}")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise


def execute_agent_query(user_query: str, timeout: float = 60.0):
    """
    Synchronous wrapper for the async agent query.
    
    Args:
        user_query: User's natural language question
        timeout: Maximum time to wait for response (seconds)
        
    Returns:
        Agent's response as a string
    """
    import asyncio
    return asyncio.run(execute_agent_query_async(user_query, timeout))


if __name__ == "__main__":
    test_query = (
        "What did the inspector say to Vic?"
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