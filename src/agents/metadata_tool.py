from llama_index.core.tools import FunctionTool
from sqlalchemy import create_engine, text
import os
import json
import yaml
from pathlib import Path

# --- LOAD CONFIG FROM YAML ---
def load_domain_config(config_path: str = None) -> dict:
    """Load domain configuration from YAML file."""
    if config_path is None:
        # Try to find config in common locations
        possible_paths = [
            Path("config/domain_config.yaml"),
            Path("domain_config.yaml"),
            Path("config.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "Could not find domain_config.yaml. "
                "Please specify CONFIG_PATH environment variable or place config in: "
                f"{[str(p) for p in possible_paths]}"
            )
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config_path = os.getenv("CONFIG_PATH")
config = load_domain_config(config_path)

# Extract values from config
DB_URI = config.get("metadata_db_uri")
if not DB_URI:
    raise ValueError(
        "metadata_db_uri is not set in domain_config.yaml. "
        "Please add it to your configuration file."
    )

TABLE_NAME = "document_metadata_catalog"
ENDPOINT_LIST = config.get("data_endpoints", [])
DOCUMENT_LIMIT = config.get("document_limit", 100)
RETRIEVAL_K = config.get("retrieval_k", 5)

if not ENDPOINT_LIST:
    raise ValueError(
        "data_endpoints is not set in domain_config.yaml. "
        "Please add at least one data endpoint."
    )

ENDPOINT_LIST_STR = "\n - ".join(ENDPOINT_LIST)

# Create engine after validation
engine = create_engine(DB_URI)


def metadata_query(user_query: str) -> str:
    """
    Execute a SQL query based on user query (natural language),
    then return a JSON string or list of document IDs.
    
    Args:
        user_query: Natural language query about documents
        
    Returns:
        JSON-encoded list of document IDs matching the query
    """
    # WARNING: This is a naive example. For production:
    # 1. Use proper LLM -> SQL translation
    # 2. Add query validation and sanitization
    # 3. Handle multiple search fields beyond just 'topic'
    # 4. Add error handling for database connections
    
    try:
        # Apply document limit from config
        sql = text(
            f"SELECT id FROM {TABLE_NAME} "
            f"WHERE topic LIKE :topic "
            f"LIMIT :limit"
        )
        params = {
            "topic": f"%{user_query}%",
            "limit": DOCUMENT_LIMIT
        }
        
        with engine.connect() as conn:
            result = conn.execute(sql, params)
            ids = [row[0] for row in result]
        
        return json.dumps({
            "ids": ids,
            "count": len(ids),
            "limit": DOCUMENT_LIMIT
        })
    
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "ids": [],
            "count": 0
        })


metadata_query_tool = FunctionTool.from_defaults(
    fn=metadata_query,
    name="metadata_query_tool",
    description=(
        f"Query document metadata in the SQL database to filter doc IDs. "
        f"Input is a natural language question about topic, date, jurisdiction, etc. "
        f"Returns a JSON-encoded list of document IDs (max {DOCUMENT_LIMIT} results). "
        f"Data sources: {ENDPOINT_LIST_STR}"
    ),
)