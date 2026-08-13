import re
import json
import difflib
import httpx
import asyncio
from typing import List, Tuple
from app.core.config import settings
from app.models.document import DocumentChunk

class RerankerService:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.groq_api_key = settings.GROQ_API_KEY
        self.model_name = model_name
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import CrossEncoder
            self._local_model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        return self._local_model

    def _normalize_text(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", text.lower().strip())
        return clean

    def deduplicate_chunks(self, scored_docs: List[Tuple[DocumentChunk, float]], similarity_threshold: float = 0.80) -> List[DocumentChunk]:
        unique_docs: List[DocumentChunk] = []
        accepted_norms: List[str] = []

        for doc, score in scored_docs:
            norm_content = self._normalize_text(doc.content)
            
            is_duplicate = False
            for accepted in accepted_norms:
                if norm_content in accepted or accepted in norm_content:
                    is_duplicate = True
                    break
                ratio = difflib.SequenceMatcher(None, norm_content, accepted).ratio()
                if ratio >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                accepted_norms.append(norm_content)

        return unique_docs

    async def _rerank_with_groq(self, query: str, docs: List[DocumentChunk]) -> List[Tuple[DocumentChunk, float]]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        chunks_formatted = ""
        for idx, doc in enumerate(docs):
            snippet = doc.content[:400].replace("\n", " ")
            chunks_formatted += f"[Chunk {idx}]: {snippet}\n\n"
            
        system_prompt = (
            "You are an expert document relevance reranker for a RAG system.\n"
            "Evaluate how relevant each candidate chunk is to the user query.\n"
            "Assign a relevance score from 0.0 (completely irrelevant) to 10.0 (highly relevant).\n"
            "Respond ONLY with a JSON object containing a key 'rankings' which is an array of objects:\n"
            '{"rankings": [{"index": 0, "score": 9.5}, {"index": 1, "score": 4.0}]}'
        )
        
        user_prompt = f"User Query: {query}\n\nCandidate Chunks:\n{chunks_formatted}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            res_data = resp.json()
            content_str = res_data["choices"][0]["message"]["content"]
            parsed = json.loads(content_str)
            rankings = parsed.get("rankings", [])
            
            scores_dict = {}
            for item in rankings:
                idx = item.get("index")
                score = float(item.get("score", 0.0))
                if isinstance(idx, int) and 0 <= idx < len(docs):
                    scores_dict[idx] = score
                    
            scored_docs = [(docs[i], scores_dict.get(i, 0.0)) for i in range(len(docs))]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs

    def _rerank_local(self, query: str, docs: List[DocumentChunk]) -> List[Tuple[DocumentChunk, float]]:
        model = self._get_local_model()
        pairs = [[query, doc.content] for doc in docs]
        scores = model.predict(pairs)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    async def rerank_async(self, query: str, docs: List[DocumentChunk], top_k: int = 4) -> List[DocumentChunk]:
        if not docs:
            return []
            
        if self.groq_api_key:
            try:
                scored_docs = await self._rerank_with_groq(query, docs)
            except Exception as e:
                print(f"Groq reranker notice: {e}, falling back to local model.")
                scored_docs = await asyncio.to_thread(self._rerank_local, query, docs)
        else:
            scored_docs = await asyncio.to_thread(self._rerank_local, query, docs)
            
        deduplicated = self.deduplicate_chunks(scored_docs)
        return deduplicated[:top_k]

    def rerank(self, query: str, docs: List[DocumentChunk], top_k: int = 4) -> List[DocumentChunk]:
        if not docs:
            return []
        if self.groq_api_key:
            try:
                # Synchronous fallback or event loop execution
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task or fallback
                        return self.deduplicate_chunks(self._rerank_local(query, docs))[:top_k]
                    return loop.run_until_complete(self.rerank_async(query, docs, top_k))
                except Exception:
                    return self.deduplicate_chunks(self._rerank_local(query, docs))[:top_k]
            except Exception:
                pass
        return self.deduplicate_chunks(self._rerank_local(query, docs))[:top_k]
