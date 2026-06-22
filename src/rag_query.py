"""
Step 3: Manual RAG query pipeline -- no framework, every step explicit.

Pipeline:
  1. Embed the user's question (same model used to embed the corpus)
  2. Query Chroma for the top-k most similar chunks
  3. Format a strict prompt: system instruction + retrieved chunks + question
  4. Call Llama 4 Scout via the Jetstream OpenAI-compatible API
  5. Print the answer with citations

The point of building this by hand (before LangGraph in Step 4) is to see
every mechanical piece of RAG with nothing abstracted away. When we rebuild
this as a multi-agent system, you'll know exactly what each node is hiding.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

JETSTREAM_API_KEY = os.getenv("JETSTREAM_API_KEY")
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "glp1_abstracts"

LLM_BASE_URL = "https://llm.jetstream-cloud.org/api"
LLM_MODEL = "llama-4-scout"

# How many chunks to retrieve per query.
# 5 is the starting point -- too few risks missing relevant context,
# too many risks diluting the prompt with marginally relevant chunks.
# We'll tune this empirically in Step 5 (eval harness).
TOP_K = 5

SYSTEM_PROMPT = """You are a medical research assistant. Answer the user's \
question using ONLY the PubMed abstract excerpts provided below.

Do not use any knowledge outside of these excerpts. If the provided excerpts \
do not contain enough information to answer the question, respond with exactly:
"The available abstracts do not contain sufficient information to answer this question."

For every claim you make, cite the source using the PMID and section label \
provided at the start of each excerpt (e.g. "According to PMID 12345678 \
[RESULTS]..."). Do not make claims without a citation."""


_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_question(question: str) -> list[float]:
    """Embed the user's question using the same model used on the corpus.

    Critical: if you embed the corpus with model A and the query with model B,
    the vectors live in different spaces and similarity scores are meaningless.
    Same model, always.
    """
    vector = _embedding_model.encode(question, normalize_embeddings=True)
    return vector.tolist()


def retrieve_chunks(question_vector: list[float], k: int) -> list[dict]:
    """Query Chroma for the top-k chunks most similar to the question vector.

    Returns a list of dicts, each with 'text' and 'metadata' keys.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=k,
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
    return chunks


def format_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """Build the messages list to send to the LLM.

    The context block prefixes each chunk with its citation info so the model
    can reference PMIDs inline in its answer without post-hoc matching.
    Returns the OpenAI messages format: [{"role": ..., "content": ...}, ...]
    """
    context_blocks = []
    for chunk in chunks:
        meta = chunk["metadata"]
        header = f"[PMID {meta['pmid']} | {meta['section']} | {meta['journal']}]"
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    user_message = f"""PubMed abstract excerpts:

{context_str}

---
Question: {question}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def call_llm(messages: list[dict]) -> str:
    """Send the formatted prompt to Llama 4 Scout and return the response text."""
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=JETSTREAM_API_KEY,
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,  # low temperature: we want faithful, consistent answers
                          # not creative ones. higher temp = more varied but less
                          # grounded responses, bad for faithfulness.
    )
    return response.choices[0].message.content


def rag_query(question: str, verbose: bool = True) -> dict:
    """Full RAG pipeline: question -> retrieved chunks -> LLM answer.

    Returns a dict with 'question', 'chunks', and 'answer' so callers
    (including the eval harness in Step 5) can inspect every stage,
    not just the final answer.
    """
    if verbose:
        print(f"\nQuestion: {question}")
        print(f"Retrieving top {TOP_K} chunks...")

    question_vector = embed_question(question)
    chunks = retrieve_chunks(question_vector, TOP_K)

    if verbose:
        print(f"\nRetrieved chunks:")
        for c in chunks:
            meta = c["metadata"]
            print(f"  [{c['distance']:.4f}] PMID {meta['pmid']} | "
                  f"{meta['section']} | {meta['journal'][:40]}")

    messages = format_prompt(question, chunks)

    if verbose:
        print(f"\nCalling {LLM_MODEL}...")

    answer = call_llm(messages)

    if verbose:
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(answer)
        print('='*60)

    return {
        "question": question,
        "chunks": chunks,
        "answer": answer,
        "messages": messages,
    }


def main():
    # A few test questions deliberately chosen to exercise different retrieval
    # and faithfulness scenarios:
    # Q1: factual, likely well-covered in the corpus
    # Q2: comparative, requires pulling from multiple chunks
    # Q3: deliberately out of scope -- tests the "I don't know" escape hatch
    test_questions = [
        "What does the research say about the effect of semaglutide on body weight?",
        "How do the GI side effects of tirzepatide compare to semaglutide?",
        "What is the recommended dosage of metformin for type 2 diabetes?",  # out of scope
    ]

    for question in test_questions:
        result = rag_query(question)
        print()


if __name__ == "__main__":
    main()