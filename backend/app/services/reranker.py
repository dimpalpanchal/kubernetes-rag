import re
import difflib
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from app.models.document import DocumentChunk

class RerankerService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name, max_length=512)

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

    def rerank(self, query: str, docs: List[DocumentChunk], top_k: int = 4) -> List[DocumentChunk]:
        if not docs:
            return []
            
        pairs = [[query, doc.content] for doc in docs]
        scores = self.model.predict(pairs)
        
        # Pair docs with scores and sort
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out duplicates before picking top_k
        deduplicated = self.deduplicate_chunks(scored_docs)
        
        return deduplicated[:top_k]

