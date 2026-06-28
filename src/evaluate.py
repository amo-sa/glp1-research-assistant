"""
Step 5b: Score pipeline outputs using Ragas.

Reads data/processed/pipeline_outputs.json (produced by run_pipeline_for_eval.py)
and scores each result on four metrics:
  - Faithfulness:       are all claims in the answer supported by retrieved chunks?
  - Answer Relevancy:   does the answer actually address the question asked?
  - Context Precision:  of the chunks retrieved, what proportion were relevant?
  - Context Recall:     of the chunks needed, what proportion were retrieved?

Intentionally imports nothing from the RAG pipeline (no langgraph, no chromadb,
no sentence-transformers) -- only Ragas and its LangChain dependencies.
This keeps the two dependency sets completely separate so they never conflict.

Results are written to data/processed/eval_results.json.
Exit code 0 = all metrics passed thresholds (CI gate passes).
Exit code 1 = one or more metrics below threshold (CI gate fails).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

JETSTREAM_API_KEY = os.getenv("JETSTREAM_API_KEY")
LLM_BASE_URL = "https://llm.jetstream-cloud.org/api"
LLM_MODEL = "llama-4-scout"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

PIPELINE_OUTPUTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "pipeline_outputs.json"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "eval_results.json"

THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.60,
}


def load_pipeline_outputs(sample_limit=None) -> list[dict]:
    if not PIPELINE_OUTPUTS_PATH.exists():
        print(f"ERROR: {PIPELINE_OUTPUTS_PATH} not found.")
        print("Run: python src/run_pipeline_for_eval.py first.")
        sys.exit(1)

    with open(PIPELINE_OUTPUTS_PATH) as f:
        data = json.load(f)

    outputs = data["outputs"]
    valid = [o for o in outputs if o.get("final_answer")]
    failed = len(outputs) - len(valid)
    if failed:
        print(f"  Skipping {failed} failed pipeline outputs (no final_answer)")

    if sample_limit:
        valid = valid[:sample_limit]
        print(f"  Sample mode: scoring {sample_limit} of {len(valid)} valid outputs")

    return valid


def build_ragas_samples(outputs: list[dict]):
    from ragas import SingleTurnSample

    samples = []
    for output in outputs:
        if not output["retrieved_contexts"]:
            print(f"  Skipping (no retrieved contexts): {output['question'][:60]}...")
            continue
        samples.append(SingleTurnSample(
            user_input=output["question"],
            response=output["final_answer"],
            retrieved_contexts=output["retrieved_contexts"],
            reference=output["reference_answer"],
        ))
    return samples


def setup_judge_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=JETSTREAM_API_KEY,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def setup_judge_embeddings():
    """Wrap local embedding model for answer relevancy metric."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return LangchainEmbeddingsWrapper(embeddings)


def run_scoring(samples, judge_llm, judge_embeddings) -> dict:
    from ragas import EvaluationDataset, evaluate
    import numpy as np

    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    dataset = EvaluationDataset(samples=samples)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    for metric in metrics:
        metric.llm = judge_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = judge_embeddings

    print(f"Scoring {len(samples)} samples across {len(metrics)} metrics...")
    print("(This makes multiple LLM calls per sample -- expect several minutes)")
    results = evaluate(dataset=dataset, metrics=metrics)


    def extract_score(value):
        """Handle both a single float and a list of per-sample scores."""
        if isinstance(value, list):
            valid = [v for v in value if v is not None]
            return float(np.mean(valid)) if valid else 0.0
        return float(value)

    return {
        "faithfulness": extract_score(results["faithfulness"]),
        "answer_relevancy": extract_score(results["answer_relevancy"]),
        "context_precision": extract_score(results["context_precision"]),
        "context_recall": extract_score(results["context_recall"]),
    }


def save_results(scores: dict, outputs: list[dict]):
    all_passed = all(scores[m] >= THRESHOLDS[m] for m in THRESHOLDS)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "thresholds": THRESHOLDS,
        "passed": all_passed,
        "per_question": [
            {
                "question": o["question"],
                "category": o["category"],
                "generated_answer": o["final_answer"],
                "reference_answer": o["reference_answer"],
                "retrieved_pmids": o["retrieved_pmids"],
                "iterations": o["iterations"],
            }
            for o in outputs
        ],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full results saved to {RESULTS_PATH}")


def print_summary(scores: dict) -> bool:
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    all_passed = True
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        import math
        if math.isnan(score):
            status = "ERROR (nan)"
            all_passed = False
        elif score >= threshold:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False
        print(f"  {metric:<25} {score:.3f}  (threshold: {threshold})  {status}")
    print("="*60)
    print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME METRICS BELOW THRESHOLD'}")
    print("="*60)
    return all_passed


def main():
    sample_limit = None
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        sample_limit = int(sys.argv[idx + 1])

    print("Loading pipeline outputs...")
    outputs = load_pipeline_outputs(sample_limit)
    if not outputs:
        print("ERROR: No valid pipeline outputs to score.")
        sys.exit(1)

    print(f"Building Ragas samples from {len(outputs)} outputs...")
    samples = build_ragas_samples(outputs)
    if not samples:
        print("ERROR: No scoreable samples after filtering.")
        sys.exit(1)

    print("Setting up judge LLM and embeddings...")
    judge_llm = setup_judge_llm()
    judge_embeddings = setup_judge_embeddings()

    scores = run_scoring(samples, judge_llm, judge_embeddings)
    save_results(scores, outputs)
    all_passed = print_summary(scores)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()