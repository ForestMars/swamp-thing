# /src/agents/main_agent.py

from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from .metadata_tool import metadata_query_tool
# (Import your Semantic Retrieval Tool when it's built in Task 2.2)

# Initialize LLM with the provider and API Key (from env var)
llm = Anthropic(model="claude-3-opus-20240229") 

# Define the set of tools available to the Agent
tools = [metadata_query_tool] # Add the retrieval tool here later

# Create the Agent
agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,
)

# Example Usage:
# result = agent.query("Of the 100 most recent documents focused on litigation involving asbestos poisoning, what were the major rulings?")

# The Agent will choose to use metadata_query_tool first.
