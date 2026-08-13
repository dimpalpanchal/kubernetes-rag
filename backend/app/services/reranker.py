from typing import List, Tuple
from sentence_transformers import CrossEncoder
from app.models.document import DocumentChunk

class RerankerService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, docs: List[DocumentChunk], top_k: int = 4) -> List[DocumentChunk]:
        if not docs:
            return []
            
        pairs = [[query, doc.content] for doc in docs]
        scores = self.model.predict(pairs)
        
        # Pair docs with scores and sort
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
