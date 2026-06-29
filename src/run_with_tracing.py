"""
Step 6: Run the research pipeline with Phoenix tracing enabled.

Usage:
  1. Start Phoenix in a separate terminal:
       python -m phoenix.server.main serve
  2. Run this script:
       python src/run_with_tracing.py
  3. Open http://localhost:6006 to view traces

Each question produces one trace in Phoenix, showing the full span hierarchy:
search_agent -> summarize_agent -> critique_agent -> supervisor -> (repeat if revised)
"""

import sys
from pathlib import Path

# Setup tracing BEFORE importing the pipeline -- the auto-instrumentation
# must be registered before LangGraph is used, not after
sys.path.insert(0, str(Path(__file__).parent))
from tracing import setup_tracing
setup_tracing()

from research_graph import run_research # noqa: E402


def main():
    questions = [
        "What does the research say about the effect of semaglutide on body weight?",
        "How do the GI side effects of tirzepatide compare to semaglutide?",
        "What is the recommended dosage of metformin for type 2 diabetes?",
    ]

    for question in questions:
        print(f"\nRunning: {question[:70]}...")
        run_research(question)

    print("\nDone. Open http://localhost:6006 to view traces in Phoenix.")


if __name__ == "__main__":
    main()