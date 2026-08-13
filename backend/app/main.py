from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from app.core.database import engine, Base, create_database_if_not_exists
from app.api import auth, chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database if not exists
    await create_database_if_not_exists()
    # Create tables on startup
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Clean up on shutdown
    await engine.dispose()

app = FastAPI(title="Kubernetes RAG Assistant", lifespan=lifespan)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
