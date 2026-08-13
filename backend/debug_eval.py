import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from app.services.hybrid_search import HybridSearchService
from app.services.reranker import RerankerService
from app.services.generation import GenerationService
from app.core.database import AsyncSessionLocal, create_database_if_not_exists

async def main():
    await create_database_if_not_exists()
    search = HybridSearchService()
    reranker = RerankerService()
    generation = GenerationService()
    
    questions = [
        "What is a Pod in Kubernetes?",
        "How do Deployments manage Pods?",
        "What is the purpose of a Service?"
    ]
    
    async with AsyncSessionLocal() as db:
        for q in questions:
            print(f"\n==========================================")
            print(f"QUESTION: {q}")
            top_chunks = await search.search(db, q, top_k=20)
            reranked_chunks = await asyncio.to_thread(reranker.rerank, q, top_chunks, top_k=4)
            
            print(f"\n--- RETRIEVED CHUNKS ({len(reranked_chunks)}) ---")
            for i, chunk in enumerate(reranked_chunks):
                src = (chunk.metadata_ or {}).get("source", "Unknown")
                title = (chunk.metadata_ or {}).get("chunk_title", "")
                print(f"[{i+1}] Source: {src} | Title: {title}")
                print(f"    Content snippet: {chunk.content[:150]}...\n")
                
            answer = await generation.generate(q, reranked_chunks, "")
            print(f"--- GENERATED ANSWER ---")
            print(answer)

if __name__ == "__main__":
    asyncio.run(main())
