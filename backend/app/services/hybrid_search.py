from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from langchain_openai import OpenAIEmbeddings
from app.models.document import DocumentChunk
from app.core.config import settings

class HybridSearchService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

    async def search(self, db: AsyncSession, query: str, top_k: int = 20) -> List[DocumentChunk]:
        # Generate dense embedding for query
        query_embedding = await self.embeddings.aembed_query(query)
        
        # Sparse Search Query using plainto_tsquery
        ts_query = func.plainto_tsquery("english", query)

        # Dense search
        stmt_dense = (
            select(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        res_dense = await db.execute(stmt_dense)
        dense_docs = res_dense.scalars().all()

        # Sparse search
        stmt_sparse = (
            select(DocumentChunk)
            .where(DocumentChunk.ts_vector.op("@@")(ts_query))
            .order_by(func.ts_rank(DocumentChunk.ts_vector, ts_query).desc())
            .limit(top_k)
        )
        res_sparse = await db.execute(stmt_sparse)
        sparse_docs = res_sparse.scalars().all()

        # Simple Reciprocal Rank Fusion (RRF)
        k = 60
        scores = {}
        docs_dict = {}

        for rank, doc in enumerate(dense_docs):
            if doc.id not in scores:
                scores[doc.id] = 0
            scores[doc.id] += 1 / (k + rank + 1)
            docs_dict[doc.id] = doc

        for rank, doc in enumerate(sparse_docs):
            if doc.id not in scores:
                scores[doc.id] = 0
            scores[doc.id] += 1 / (k + rank + 1)
            docs_dict[doc.id] = doc

        # Sort by RRF score
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        top_docs = [docs_dict[doc_id] for doc_id in sorted_doc_ids[:top_k]]

        return top_docs
