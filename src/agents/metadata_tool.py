# /src/agents/metadata_tool.py

from llama_index.core.tools import FunctionTool
from sqlalchemy import create_engine, text
import os
import json

# --- CONFIG ---
DB_URI = os.getenv("METADATA_DB_URI")
TABLE_NAME = "document_metadata_catalog"
DATA_ENDPOINTS_JSON = os.getenv("DATA_ENDPOINTS_JSON", '["s3://default-bucket/"]')
try:
    ENDPOINT_LIST = json.loads(DATA_ENDPOINTS_JSON)
except json.JSONDecodeError:
    ENDPOINT_LIST = DATA_ENDPOINTS_JSON.split(',')

ENDPOINT_LIST_STR = "\n - ".join(ENDPOINT_LIST)

engine = create_engine(DB_URI)

def metadata_query(user_query: str) -> str:
    """
    Execute a SQL query based on user query (natural language),
    then return a JSON string or list of document IDs.
    """
    # Here, you need to map the user_query to actual SQL.
    # This could be as simple or complex as you like.
    # For example, interpret a structured pattern or do some LLM -> SQL translation.
    # But for now, I'll assume you do a naive WHERE match on a “topic” column.

    # **Very naive example** — you should sanitize inputs / parameterize
    sql = text(f"SELECT id FROM {TABLE_NAME} WHERE topic LIKE :topic")
    params = {"topic": f"%{user_query}%"}
    result = engine.execute(sql, params)
    ids = [row["id"] for row in result]
    return json.dumps(ids)

metadata_query_tool = FunctionTool.from_defaults(
    fn=metadata_query,
    name="metadata_query_tool",
    description=(
        "Query document metadata in the SQL database to filter doc IDs. "
        "Input is a natural language question about topic, date, jurisdiction, etc. "
        "Returns a JSON-encoded list of document IDs."
    ),
)
