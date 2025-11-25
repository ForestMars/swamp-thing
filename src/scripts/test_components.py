#!/usr/bin/env python
"""
Test individual components before running the full agent.
Run with: uv run python test_components.py
"""

import sys
import os

# Test 1: Check if Ollama is running
print("=" * 60)
print("TEST 1: Checking Ollama connection...")
print("=" * 60)
try:
    import requests
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json()
        print(f"✅ Ollama is running at {ollama_url}")
        print(f"Available models: {[m['name'] for m in models.get('models', [])]}")
    else:
        print(f"❌ Ollama responded with status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Cannot connect to Ollama: {e}")
    print("Please start Ollama with: ollama serve")
    sys.exit(1)

# Test 2: Check database connection
print("\n" + "=" * 60)
print("TEST 2: Checking database configuration...")
print("=" * 60)
try:
    from src.agents.metadata_tool import DB_URI, ENDPOINT_LIST
    print(f"✅ Metadata DB URI: {DB_URI}")
    print(f"✅ Data endpoints: {ENDPOINT_LIST}")
    
    # Try to connect
    from sqlalchemy import create_engine, text
    engine = create_engine(DB_URI)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
except Exception as e:
    print(f"⚠️  Database connection failed: {e}")
    print("This is expected if you haven't set up the database yet.")
    print("The agent will fail when trying to query metadata.")

# Test 3: Check vector store connection
print("\n" + "=" * 60)
print("TEST 3: Checking vector store configuration...")
print("=" * 60)
try:
    from src.agents.semantic_retriever_agent import vector_store, index
    if vector_store is None:
        print("⚠️  Vector store not connected")
        print("This is expected if you haven't set up PostgreSQL with pgvector.")
    else:
        print("✅ Vector store configured")
    
    if index is None:
        print("⚠️  Index not created")
    else:
        print("✅ Index created")
except Exception as e:
    print(f"⚠️  Vector store check failed: {e}")

# Test 4: Test LLM directly
print("\n" + "=" * 60)
print("TEST 4: Testing LLM connection...")
print("=" * 60)
try:
    from llama_index.llms.ollama import Ollama
    
    llm = Ollama(
        model="qwen2.5:latest",
        base_url=ollama_url,
        temperature=0.1,
        request_timeout=30.0,
    )
    
    print("Sending test prompt to LLM...")
    response = llm.complete("Say 'Hello, I am working!' in exactly 5 words.")
    print(f"✅ LLM Response: {response.text}")
    
except Exception as e:
    print(f"❌ LLM test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check tools
print("\n" + "=" * 60)
print("TEST 5: Checking tools...")
print("=" * 60)
try:
    from src.agents.metadata_tool import metadata_query_tool
    print(f"✅ Metadata tool: {metadata_query_tool.metadata.name}")
    
    from src.agents.reranker_agent import reranked_query_tool
    print(f"✅ Reranker tool: {reranked_query_tool.metadata.name}")
    
except Exception as e:
    print(f"❌ Tool check failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If all tests passed (✅), you can try running the agent.")
print("If you see warnings (⚠️), the agent may fail when accessing those components.")
print("\nTo run the agent: uv run python -m src.agents.main_agent")