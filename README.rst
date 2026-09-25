
Swamp Thing
===========

Overview
--------

This repository is a prototype for a document-focused retrieval-augmented generation (RAG) system built around Postgres, LlamaIndex, and local Ollama models. The design is intentionally multi-stage: document metadata is filtered before semantic search, then candidate chunks are reranked, and finally an LLM synthesizes the answer.

The project is best thought of as an experimental research and orchestration layer for local document search rather than a finished production application.

Project goals
-------------

* Index and store large document collections locally or on a networked filesystem
* Filter by metadata before doing semantic retrieval
* Reduce retrieval noise with a reranking step
* Use a local model stack for embeddings and generation
* Keep the retrieval flow modular and agent-driven

High-level architecture
----------------------

1. Ingestion
   The ingestion layer reads source files, embeds them, clusters them, and stores metadata in a Postgres catalog.

2. Metadata filtering
   The metadata tool queries a document catalog to reduce the candidate set to relevant document IDs before vector search.

3. Semantic retrieval
   The vector store is queried only within the filtered document subset to improve precision.

4. Reranking
   Retrieved chunks are reranked to preserve only the most relevant material for synthesis.

5. Final answer generation
   A local LLM agent answers the question based only on the retrieved evidence.

Repository layout
-----------------

* ``config/`` - domain and global configuration files
* ``src/agents/`` - SQL metadata tool, retriever, reranker, and orchestration agent
* ``src/ingest/`` - ingestion pipeline for loading and embedding documents
* ``src/manage/`` - cluster management utilities
* ``xyx/examples/`` - sample usages and experiments
* ``st`` - project entrypoint or helper script

Core modules
------------

``src/ingest/ingest_documents.py``
    Builds the initial document index. It loads local files, creates embeddings, clusters documents, writes metadata to PostgreSQL, and stores vectors in a PGVectorStore.

``src/agents/metadata_tool.py``
    Exposes a tool that translates a natural-language document question into a metadata query against the SQL catalog and returns matching document IDs.

``src/agents/semantic_retriever_agent.py``
    Connects to the vector database and creates filtered retrieval logic using metadata constraints.

``src/agents/reranker_agent.py``
    Applies reranking to the retrieved nodes before the final answer is generated.

``src/agents/main_agent.py``
    Builds the top-level agent, configures local Ollama models, and orchestrates the retrieval workflow.

Configuration
-------------

Configuration is split into a global layer and a domain-specific layer:

* ``config/global_config.yaml`` - shared defaults for models, chunking, and retrieval behavior
* ``config/domain_config.yaml`` - domain-specific metadata and document endpoint settings
* ``config/domain_endpoints.yaml`` - endpoint definitions for the document sources

These files are intended to separate infrastructure defaults from client/domain-specific rules.

Environment and runtime assumptions
-----------------------------------

This project expects the following services to be available locally:

* PostgreSQL instances for metadata and vector storage
* Local Ollama service for embeddings and generation
* A document lake or filesystem directory containing the source files

The code currently contains several hardcoded local values for database credentials and hostnames, so it is best treated as a local prototype rather than a fully configurable deployment.

Setup
-----

This project uses ``uv`` and the Python package metadata in ``pyproject.toml``.

Typical setup flow:

1. Install dependencies with ``uv sync``
2. Ensure PostgreSQL is running and the expected databases/tables exist
3. Start Ollama and pull the required models
4. Configure ``config/*.yaml`` for your environment
5. Run the ingestion script
6. Use the agent to query the indexed corpus

Example commands
----------------

.. code-block:: bash

   uv sync
   uv run python src/ingest/ingest_documents.py
   uv run python src/agents/main_agent.py

Current project status
----------------------

This repository is in an active prototype state. The architecture is coherent and the retrieval flow is sensible, but the code still has rough edges, including:

* hardcoded configuration values
* partially finished management scripts
* version drift in LlamaIndex integration patterns
* inconsistent naming between metadata IDs and vector metadata keys

That makes the repo a good foundation for a local RAG system, but not yet a polished application.

Next steps
----------

The immediate next tasks are to:

1. clean up configuration and environment management
2. standardize the data model between metadata and vector retrieval
3. validate the full ingestion-to-query pipeline with a known dataset
4. document the operational workflow for local setup and troubleshooting

This README is intentionally a practical snapshot of the repo as it exists now, not a marketing document.
