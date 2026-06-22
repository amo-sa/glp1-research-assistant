"""
Step 4: Multi-agent research pipeline using LangGraph.

Architecture: rules-based supervisor routing to three specialist agents.
  - search_agent:     retrieves relevant chunks from Chroma for the question
  - summarize_agent:  drafts an answer grounded in the retrieved chunks
  - critique_agent:   checks the draft for unsupported claims / hallucinations
  - supervisor:       decides what to run next based on current state;
                      loops back if critique flags issues, finalizes if approved

State flows through a shared ResearchState TypedDict. Each node reads the
full state and returns only the fields it updated -- LangGraph merges those
back into the state automatically (last-write-wins per field).

Why this over the manual RAG pipeline (Step 3)?
  - Compound questions get decomposed: the supervisor can route to search
    multiple times with different sub-queries if needed
  - Summarization and critique are separated: the critique agent has fresh
    context and isn't primed to defend the summary it just wrote
  - The pipeline can loop: if critique finds unsupported claims, the graph
    can route back rather than just returning a bad answer
  - Each agent's output is inspectable independently in the final state
"""

import os
from pathlib import Path
from typing import Literal

import chromadb
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict

load_dotenv()

JETSTREAM_API_KEY = os.getenv("JETSTREAM_API_KEY")
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "glp1_abstracts"
LLM_BASE_URL = "https://llm.jetstream-cloud.org/api"
LLM_MODEL = "llama-4-scout"
TOP_K = 5
MAX_ITERATIONS = 3  # safety ceiling: prevents infinite supervisor loops


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """Shared state passed through every node in the graph.

    Each node receives the full state and returns a dict of only the fields
    it changed. LangGraph merges those changes back automatically.
    """
    question: str        # original user question, set once, never modified
    chunks: list[dict]   # retrieved context chunks, set by search_agent
    draft_answer: str    # LLM-generated answer, set by summarize_agent
    critique: str        # critique feedback, set by critique_agent
    final_answer: str    # approved answer, set by supervisor on completion
    iterations: int      # how many supervisor loops have run; guards vs. infinite loop


# ---------------------------------------------------------------------------
# Shared clients (loaded once, reused across all node calls)
# ---------------------------------------------------------------------------

_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_collection(COLLECTION_NAME)
_llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=JETSTREAM_API_KEY)


def _call_llm(messages: list[dict], temperature: float = 0.1) -> str:
    """Shared LLM call helper used by all agents."""
    response = _llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Nodes (plain Python functions: state -> dict of updates)
# ---------------------------------------------------------------------------

def search_agent(state: ResearchState) -> dict:
    """Retrieve the top-k most relevant chunks for the question.

    Embeds the question and queries Chroma. Returns updated 'chunks' field.
    In a more advanced version, the supervisor could call this multiple times
    with different sub-queries for compound questions.
    """
    print(f"  [search_agent] embedding question and querying Chroma (top {TOP_K})...")

    question_vector = _embedding_model.encode(
        state["question"], normalize_embeddings=True
    ).tolist()

    results = _collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    for c in chunks:
        meta = c["metadata"]
        print(f"    [{c['distance']:.4f}] PMID {meta['pmid']} | {meta['section']}")

    # Only return the field this node owns -- LangGraph merges the rest
    return {"chunks": chunks}


def summarize_agent(state: ResearchState) -> dict:
    """Draft an answer grounded strictly in the retrieved chunks.

    On first call: standard strict RAG prompt.
    On revision loop: explicitly includes the prior draft and the critique
    agent's specific complaints so the agent is correcting what was actually
    wrong, not just re-rolling the same prompt and hoping for better output.
    """
    print("  [summarize_agent] drafting answer from retrieved chunks...")

    context_blocks = []
    for chunk in state["chunks"]:
        meta = chunk["metadata"]
        header = f"[PMID {meta['pmid']} | {meta['section']} | {meta['journal']}]"
        context_blocks.append(f"{header}\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    critique = state.get("critique", "")
    prior_draft = state.get("draft_answer", "")

    if critique.startswith("REVISE") and prior_draft:
        # Revision loop -- give the agent the prior draft and the specific
        # issues so it can fix exactly what was wrong rather than starting
        # from scratch blindly
        revision_block = (
            f"\n\n---\nYour previous answer was flagged for revision.\n\n"
            f"Previous answer:\n{prior_draft}\n\n"
            f"Issues identified by the fact-checker:\n{critique}\n\n"
            f"Write a corrected answer that specifically addresses these issues. "
            f"Be more careful about what the excerpts actually support."
        )
    else:
        revision_block = ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical research assistant. Answer the question using "
                "ONLY the PubMed abstract excerpts provided. Do not use any outside "
                "knowledge. If the excerpts are insufficient, say so explicitly. "
                "Cite every claim with its PMID and section label."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PubMed abstract excerpts:\n\n{context_str}\n\n"
                f"---\nQuestion: {state['question']}"
                f"{revision_block}"
            ),
        },
    ]

    draft = _call_llm(messages)
    print(f"  [summarize_agent] draft complete ({len(draft)} chars)")
    return {"draft_answer": draft}


def critique_agent(state: ResearchState) -> dict:
    """Check the draft answer for unsupported claims or hallucinations.

    The critique agent has a fresh context -- it wasn't the one that wrote
    the summary, so it isn't primed to defend it. It checks each claim in
    the draft against the retrieved chunks and flags anything unsupported.

    Returns a critique string starting with either APPROVED or REVISE,
    which the supervisor's routing function reads to decide next steps.
    """
    print("  [critique_agent] checking draft for unsupported claims...")

    context_blocks = []
    for chunk in state["chunks"]:
        meta = chunk["metadata"]
        header = f"[PMID {meta['pmid']} | {meta['section']}]"
        context_blocks.append(f"{header}\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a rigorous scientific fact-checker. You will be given a "
                "set of PubMed abstract excerpts (the only permitted sources) and "
                "a draft answer written by a summarization agent.\n\n"
                "Your job: check every factual claim in the draft answer against "
                "the provided excerpts. Identify any claim that is not directly "
                "supported by the excerpts.\n\n"
                "Respond in exactly one of these two formats:\n"
                "APPROVED: <one sentence explaining why the draft is well-supported>\n"
                "REVISE: <specific list of unsupported claims that need to be fixed>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source excerpts:\n\n{context_str}\n\n"
                f"---\nDraft answer to check:\n\n{state['draft_answer']}"
            ),
        },
    ]

    critique = _call_llm(messages)
    print(f"  [critique_agent] verdict: {critique[:80]}...")
    return {"critique": critique}


def supervisor(state: ResearchState) -> dict:
    """Decide the next step and finalize when approved.

    This is a rules-based supervisor (no LLM call needed for routing):
      - No chunks yet -> route to search_agent (handled via graph edges)
      - No draft yet -> route to summarize_agent (handled via graph edges)
      - No critique yet -> route to critique_agent (handled via graph edges)
      - Critique starts with APPROVED -> finalize
      - Critique starts with REVISE -> if under iteration limit, loop back
        to summarize; if over limit, finalize anyway with a warning

    Note: the actual routing decisions (which node fires next) are handled
    by the conditional edge function below -- this node only handles
    finalization logic (writing final_answer) when the loop is done.
    The supervisor node itself only runs when critique exists.
    """
    iterations = state.get("iterations", 0) + 1
    print(f"  [supervisor] iteration {iterations}, critique verdict: "
          f"{state.get('critique', '')[:60]}...")

    critique = state.get("critique", "")
    at_limit = iterations >= MAX_ITERATIONS

    if critique.startswith("APPROVED") or at_limit:
        if at_limit and not critique.startswith("APPROVED"):
            print(f"  [supervisor] iteration limit reached, finalizing anyway")
            final = (
                f"{state['draft_answer']}\n\n"
                f"[Note: critique flagged issues but iteration limit reached: "
                f"{critique}]"
            )
        else:
            print("  [supervisor] critique approved, finalizing answer")
            final = state["draft_answer"]
        return {"final_answer": final, "iterations": iterations}

    # REVISE case: don't set final_answer, just update iterations so the
    # conditional edge knows to loop back to summarize_agent
    print("  [supervisor] critique requested revision, looping back...")
    return {"iterations": iterations}


# ---------------------------------------------------------------------------
# Routing (conditional edges)
# ---------------------------------------------------------------------------

def route_after_supervisor(state: ResearchState) -> Literal["summarize_agent", "__end__"]:
    """Called by LangGraph after supervisor runs to decide the next node.

    If final_answer is set, the supervisor approved -- go to END.
    If not, loop back to summarize_agent for another attempt.

    The return value must be a node name or the special "__end__" string
    (imported as END from langgraph.graph).
    """
    if state.get("final_answer"):
        return "__end__"
    return "summarize_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Assemble and compile the research graph.

    Node registration order doesn't matter -- only the edges determine
    execution order. Compiling the graph validates that all referenced
    node names actually exist and that the graph has a valid structure
    (e.g. no node is unreachable, START and END are properly connected).
    """
    graph = StateGraph(ResearchState)

    # Register nodes -- each is just a function reference, not a call
    graph.add_node("search_agent", search_agent)
    graph.add_node("summarize_agent", summarize_agent)
    graph.add_node("critique_agent", critique_agent)
    graph.add_node("supervisor", supervisor)

    # Fixed edges: the happy path when everything proceeds in order
    graph.add_edge(START, "search_agent")         # always start with retrieval
    graph.add_edge("search_agent", "summarize_agent")
    graph.add_edge("summarize_agent", "critique_agent")
    graph.add_edge("critique_agent", "supervisor")

    # Conditional edge after supervisor: approve -> END, revise -> loop back
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,  # this function inspects state and returns node name
        {
            "summarize_agent": "summarize_agent",  # REVISE: loop back
            "__end__": END,                         # APPROVED: done
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_research(question: str) -> ResearchState:
    """Run the full multi-agent research pipeline for a question.

    Returns the final state so callers can inspect every intermediate
    result (chunks, draft, critique, final answer), not just the output.
    """
    graph = build_graph()

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    # Initial state: only question is set; all other fields start empty/zero
    initial_state: ResearchState = {
        "question": question,
        "chunks": [],
        "draft_answer": "",
        "critique": "",
        "final_answer": "",
        "iterations": 0,
    }

    final_state = graph.invoke(initial_state)

    print(f"\nFINAL ANSWER:")
    print(final_state["final_answer"])
    print(f"\nCompleted in {final_state['iterations']} iteration(s)")
    print('='*60)

    return final_state


def main():
    test_questions = [
        # "Is there evidence that GLP-1 drugs affect outcomes beyond weight loss, such as cardiovascular or metabolic benefits?",
        # "Which drug is more effective for long-term weight loss — semaglutide or tirzepatide?",
        "What does the research say about the effect of semaglutide on body weight?",
        "How do the GI side effects of tirzepatide compare to semaglutide?",
        "What is the recommended dosage of metformin for type 2 diabetes?",
    ]

    # result = run_research("Which drug is more effective for long-term weight loss — semaglutide or tirzepatide?")


    for question in test_questions:
        result = run_research(question)
        print()


if __name__ == "__main__":
    main()