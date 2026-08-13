import os
import sys
import json
import asyncio
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.hybrid_search import HybridSearchService
from app.services.reranker import RerankerService
from app.services.generation import GenerationService
from app.core.database import AsyncSessionLocal, engine, create_database_if_not_exists

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

async def main():
    await create_database_if_not_exists()
    search = HybridSearchService()
    reranker = RerankerService()
    generation = GenerationService()
    
    benchmark_file = Path(__file__).parent / "benchmark.jsonl"
    
    questions = []
    ground_truths = []
    
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["question"])
            ground_truths.append(data["ground_truth"])
            
    answers = []
    contexts_list = []
    
    print("Generating responses for benchmark questions...")
    
    async with AsyncSessionLocal() as db:
        for q in questions:
            print(f"Q: {q}")
            
            top_chunks = await search.search(db, q, top_k=20)
            reranked_chunks = await asyncio.to_thread(reranker.rerank, q, top_chunks, top_k=4)
            
            contexts = [chunk.content for chunk in reranked_chunks]
            contexts_list.append(contexts)
            
            answer = await generation.generate(q, reranked_chunks, "")
            answers.append(answer)
            
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Running RAGAS evaluation...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()
    
    output_file = Path(__file__).parent / "results.csv"
    df.to_csv(output_file, index=False)
    
    print("\nEvaluation Results:")
    print(result)
    
    with open(Path(__file__).parent / "results.md", "w", encoding="utf-8") as f:
        f.write("# RAGAS Evaluation Results\n\n")
        f.write("```text\n")
        f.write(str(result))
        f.write("\n```\n")

if __name__ == "__main__":
    asyncio.run(main())
