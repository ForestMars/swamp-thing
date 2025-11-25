# /src/agents/metadata_tool.py (working for llama-index 0.14.8)

from llama_index.core.tools import QueryEngineTool
from llama_index.indices.struct_store.sql import SQLStructStoreIndex
from sqlalchemy import create_engine
import os
import json

# --- CONFIGURATION ---
DB_URI = os.getenv("METADATA_DB_URI")
TABLE_NAME = "document_metadata_catalog"

DATA_ENDPOINTS_JSON = os.getenv("DATA_ENDPOINTS_JSON", '["s3://default-bucket/"]')
try:
    ENDPOINT_LIST = json.loads(DATA_ENDPOINTS_JSON)
except json.JSONDecodeError:
    ENDPOINT_LIST = DATA_ENDPOINTS_JSON.split(',')

ENDPOINT_LIST_STR = "\n - ".join(ENDPOINT_LIST)

# --- 1. Create SQLAlchemy Engine ---
engine = create_engine(DB_URI)

# --- 2. Build SQLStructStoreIndex ---
sql_index = SQLStructStoreIndex(
    sql_database=engine,
    table_names=[TABLE_NAME]
)

# --- 3. Convert to Query Engine ---
sql_query_engine = sql_index.as_query_engine(
    verbose=True
)

# --- 4. Wrap as Agent Tool ---
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
