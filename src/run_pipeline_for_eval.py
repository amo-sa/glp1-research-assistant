"""
Step 5a: Run the multi-agent RAG pipeline on all golden set questions
and save outputs to data/processed/pipeline_outputs.json.

This script is intentionally separate from evaluate.py so that:
  1. The pipeline (langgraph, chromadb, sentence-transformers) and the
     eval harness (ragas, langchain) have completely independent dependency
     sets and never conflict with each other at import time.
  2. You can re-run just the Ragas scoring step (evaluate.py) without
     re-running all the expensive pipeline LLM calls -- useful when
     iterating on eval metrics or thresholds.
  3. CI can run these as two separate steps with clear separation of concerns.

Run this first, then run evaluate.py.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to path so we can import research_graph
sys.path.insert(0, str(Path(__file__).parent))

from research_graph import run_research

GOLDEN_TEST_SET_PATH = Path(__file__).parent.parent / "data" / "golden_test_set.json"
OUTPUTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "pipeline_outputs.json"


def main():
    # Load golden test set
    with open(GOLDEN_TEST_SET_PATH) as f:
        golden_items = json.load(f)

    # Check for --sample flag for quick development runs
    sample_limit = None
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        sample_limit = int(sys.argv[idx + 1])
        golden_items = golden_items[:sample_limit]
        print(f"Sample mode: running {sample_limit} of {len(golden_items)} questions")

    print(f"Running pipeline on {len(golden_items)} questions...")
    outputs = []

    for i, item in enumerate(golden_items):
        print(f"\n[{i+1}/{len(golden_items)}] {item['question'][:70]}...")
        try:
            result = run_research(item["question"])
            outputs.append({
                "question": item["question"],
                "category": item["category"],
                "reference_answer": item["reference_answer"],
                "relevant_pmids": item["relevant_pmids"],
                "final_answer": result["final_answer"],
                "retrieved_contexts": [c["text"] for c in result["chunks"]],
                "retrieved_pmids": [c["metadata"]["pmid"] for c in result["chunks"]],
                "iterations": result["iterations"],
            })
            print(f"  done ({result['iterations']} iteration(s))")
        except Exception as e:
            print(f"  ERROR: {e}")
            # Save a failed entry so evaluate.py knows this question errored
            outputs.append({
                "question": item["question"],
                "category": item["category"],
                "reference_answer": item["reference_answer"],
                "relevant_pmids": item["relevant_pmids"],
                "final_answer": None,
                "retrieved_contexts": [],
                "retrieved_pmids": [],
                "iterations": 0,
                "error": str(e),
            })

    # Save to disk
    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(outputs),
        "successful": sum(1 for o in outputs if o.get("final_answer")),
        "outputs": outputs,
    }
    with open(OUTPUTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {payload['successful']}/{payload['total']} successful outputs to {OUTPUTS_PATH}")
    print("Now run: python src/evaluate.py")


if __name__ == "__main__":
    main()