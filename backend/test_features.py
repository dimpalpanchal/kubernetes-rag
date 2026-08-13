import asyncio
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.services.intent_classifier import IntentClassifierService
from app.services.reranker import RerankerService
from app.models.document import DocumentChunk

async def test_intent_classifier():
    classifier = IntentClassifierService()
    
    convo_tests = [
        "hello",
        "hi",
        "hey there",
        "thanks!",
        "thank you so much",
        "who are you?",
        "what can you do?",
        "good morning"
    ]
    
    query_tests = [
        "What is a Kubernetes Pod?",
        "How do I set up horizontal pod autoscaler?",
        "Explain ingress controllers",
        "How to view logs in kubectl?"
    ]
    
    print("--- Testing Intent Classifier ---")
    for msg in convo_tests:
        intent = await classifier.classify(msg)
        print(f"Message: '{msg}' -> Intent: {intent}")
        assert intent == "CONVERSATIONAL", f"Expected CONVERSATIONAL for '{msg}', got '{intent}'"
        
    for msg in query_tests:
        intent = await classifier.classify(msg)
        print(f"Message: '{msg}' -> Intent: {intent}")
        assert intent == "KUBERNETES_QUERY", f"Expected KUBERNETES_QUERY for '{msg}', got '{intent}'"
        
    print("Intent Classifier tests passed!\n")

def test_reranker_deduplication():
    reranker = RerankerService()
    
    doc1 = DocumentChunk(id=1, content="Kubernetes, also known as K8s, is an open source system for managing containerized applications across multiple hosts.", metadata_={"source": "doc1.md"})
    doc2 = DocumentChunk(id=2, content="Kubernetes, also known as K8s, is an open source system for managing containerized applications across multiple hosts.", metadata_={"source": "doc2.md"})
    doc3 = DocumentChunk(id=3, content="Kubernetes builds upon a decade and a half of experience at Google running production workloads at scale.", metadata_={"source": "doc3.md"})
    
    scored_docs = [
        (doc1, 0.95),
        (doc2, 0.94),
        (doc3, 0.85)
    ]
    
    print("--- Testing Chunk Deduplication ---")
    deduped = reranker.deduplicate_chunks(scored_docs)
    print(f"Original chunks count: {len(scored_docs)}, Deduplicated chunks count: {len(deduped)}")
    for d in deduped:
        print(f"Accepted chunk ID {d.id} from {d.metadata_['source']}")
        
    assert len(deduped) == 2, f"Expected 2 unique chunks, got {len(deduped)}"
    assert deduped[0].id == 1
    assert deduped[1].id == 3
    print("Chunk Deduplication tests passed!\n")

async def main():
    await test_intent_classifier()
    test_reranker_deduplication()
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
