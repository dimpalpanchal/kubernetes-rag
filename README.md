# Kubernetes RAG Assistant

An enterprise-grade, secure, retrieval-augmented generation (RAG) assistant for querying official Kubernetes documentation.

Key features
- Hybrid retrieval (pgvector + PostgreSQL full-text search) with Reciprocal Rank Fusion (RRF)
- Local re-ranking with a cross-encoder and history-aware query rewriting using OpenAI
- FastAPI backend with JWT auth and PostgreSQL storage (pgvector + tsvector)
- Streamlit frontend chat UI with chunk-level citations
- Evaluation via RAGAS metrics

Quick Start (Docker)
--------------------
1. Create a `.env` in the repository root with the values below:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ragdb
JWT_SECRET=supersecretkey
OPENAI_API_KEY=your_openai_api_key_here
```

2. Build and start services:

```bash
docker-compose up -d --build
```

Services
- Backend: http://localhost:8000
- Frontend: http://localhost:8501
- Postgres: localhost:5432

Local Development (without Docker)
---------------------------------
Recommended: use the provided conda environment:

```bash
conda activate kubernetes
```

Start the DB (if using docker for DB only):

```bash
docker-compose up db -d
```

Start the backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Start the frontend:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Ingestion
---------
Ingestion clones/reads Kubernetes markdown, chunks it, embeds text, and stores vectors and sparse indices in Postgres. To run ingestion locally:

```bash
cd backend
pip install -r requirements.txt
python ingestion/ingest.py
```

Ensure the database is reachable and `OPENAI_API_KEY` is set.

Evaluation
----------
Run the RAGAS evaluation suite:

```bash
cd evaluation
pip install -r requirements.txt
python run_ragas_eval.py
```

Outputs are written to `results.csv` and `results.md` in the `evaluation` folder.

Configuration & notes
- Database: PostgreSQL with `pgvector` extension and a `tsvector` generated column for full-text search.
- Re-ranking: a local cross-encoder model refines retrieval results.
- Query rewriting: `gpt-4o-mini` is used for history-aware rewrites; replace with your preferred model if needed.

Contributing
------------
- Fork, create a feature branch, and open a PR.
- Run tests and linters where applicable.

Questions or help
- Open an issue or contact the maintainer.

License
-------
See `LICENSE` (if present) or consult the project owner for licensing.

Tech stack: Python, FastAPI, PostgreSQL, pgvector, LangChain, OpenAI, Docker, Streamlit, RAGAS