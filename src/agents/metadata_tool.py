# /src/agents/metadata_tool.py (Updated for llama-index 0.14.8)

from llama_index.core.tools import QueryEngineTool
from llama_index.indices.query.query_engine import SQLStructStoreQueryEngine
from sqlalchemy import create_engine
import os
import json  # Used to parse the endpoint list if stored as a JSON environment variable

# --- CONFIGURATION LOADING ---
DB_URI = os.getenv("METADATA_DB_URI")
TABLE_NAME = "document_metadata_catalog"

# Load the list of endpoints from the environment variable (CRITICAL STEP)
# Assumes the environment variable DATA_ENDPOINTS_JSON is a JSON array string
DATA_ENDPOINTS_JSON = os.getenv("DATA_ENDPOINTS_JSON", '["s3://default-bucket/"]')
try:
    ENDPOINT_LIST = json.loads(DATA_ENDPOINTS_JSON)
except json.JSONDecodeError:
    # Fallback if the environment variable is not valid JSON
    ENDPOINT_LIST = DATA_ENDPOINTS_JSON.split(',')

# Format the list into a string for the LLM prompt
ENDPOINT_LIST_STR = "\n - ".join(ENDPOINT_LIST)

# --- 1. Create SQLAlchemy Engine ---
engine = create_engine(DB_URI)

# --- 2. Define the Query Engine that executes SQL ---
sql_query_engine = SQLStructStoreQueryEngine(
    sql_database=engine,
    tables=[TABLE_NAME],
    verbose=True
)

# --- 3. Define the Tool the LLM Agent can use ---
metadata_query_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="metadata_query_tool",
    description=(
        "Use this tool to find document IDs based on structured criteria like "
        "topic, filing date, or jurisdiction. "
        "The current active document repositories are:\n - "
        f"{ENDPOINT_LIST_STR}"
        "\nInput must be a question that can be translated to SQL, focusing on ID, date, and topic columns."
    ),
)

