# Kubernetes RAG Assistant

An enterprise-grade, secure, context-aware RAG application for querying official Kubernetes documentation.

## Architecture Overview

This project implements an end-to-end RAG (Retrieval-Augmented Generation) pipeline:
- **Backend**: FastAPI with async Python.
- **Database**: PostgreSQL with `pgvector` extension.
  - Utilizes async `SQLAlchemy` and `asyncpg`.
  - Uses `pgvector` for storing embeddings and cosine similarity dense search.
  - Uses PostgreSQL full-text search (`tsvector` generated column) for sparse search.
- **Hybrid Search**: Combines pgvector dense search and full-text search using Reciprocal Rank Fusion (RRF). Safe `plainto_tsquery` is used to prevent injection.
- **Re-ranking**: Local cross-encoder (`BAAI/bge-reranker-base`) refines the retrieved chunks.
- **Generation & Rewriting**: Uses OpenAI `gpt-4o-mini` to contextualize queries based on chat history and generate the final answer with citations.
- **Authentication**: JWT authentication with bcrypt password hashing.
- **Frontend**: Streamlit chat interface with session management and chunk citation display.
- **Evaluation**: RAGAS metrics for Faithfulness, Answer Relevance, and Context Precision/Recall.

## Setup Instructions

conda activate kubernetes

### 1. Environment Variables

Create or update the `.env` file in the root directory:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ragdb
JWT_SECRET=supersecretkey
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Run with Docker Compose

Start the PostgreSQL database (which has pgvector pre-installed), the FastAPI backend, and the Streamlit frontend.

```bash
docker-compose up -d --build
```
This maps:
- FastAPI Backend to `http://localhost:8000`
- Streamlit Frontend to `http://localhost:8501`
- Postgres DB to `localhost:5432`

## How to Run Ingestion

The ingestion script runs independently of the main API. It clones a subset of the Kubernetes documentation (`content/en/docs/concepts/`), chunks it, generates OpenAI embeddings, and stores it in the database.

Ensure the database container is running, then execute:

```bash
# Assuming you have a local python environment matching the backend requirements:
cd backend
pip install -r requirements.txt
python ingestion/ingest.py
```
*Alternatively, you can run this inside the backend docker container.*

## How to Start the App (Local Dev)

If you prefer to run services outside of Docker for development:

1. **Start DB**: `docker-compose up db -d`
2. **Start Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8000
   ```
3. **Start Frontend**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```

## How to Run Evaluations

The evaluation uses the RAGAS framework.

```bash
cd evaluation
pip install -r requirements.txt
python run_ragas_eval.py
```
*Note: Ensure the ingestion process is complete and `OPENAI_API_KEY` is set in the `.env` file before running evaluations.*

The results will be saved in `results.csv` and `results.md`.



```
**Hybrid RAG System for Kubernetes Documentation**

Built a Hybrid RAG system for Kubernetes documentation, implementing document ingestion, chunking, and indexing of Markdown files using pgvector-based semantic search and PostgreSQL full-text search with Reciprocal Rank Fusion (RRF).

Enhanced retrieval quality by integrating a history-aware query rewriter using gpt-4o-mini and a BAAI/bge-reranker-large re-ranking layer, enabling more accurate context-aware multi-turn question answering.

Developed a FastAPI backend with JWT authentication and PostgreSQL-backed chat history storage, and evaluated the system on 100+ test queries using RAGAS, achieving 0.89 Faithfulness, 0.86 Answer Relevance, and 0.91 Context Recall.


Docs → Ingestion/Indexing → Hybrid Retrieval → Query Rewriting/Reranking → Backend → Evaluation


Tech: Python, FastAPI, PostgreSQL, pgvector, LangChain, OpenAI, Docker, Streamlit, RAGAS
```