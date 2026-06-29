# GLP-1 Research Assistant — Multi-Agent RAG System with Automated Evaluation

A production-inspired AI engineering project demonstrating end-to-end RAG pipeline
design, multi-agent orchestration, automated evaluation, and LLM observability.
Built to prepare for AI Engineer interviews and showcase depth of understanding
beyond surface-level library usage.

## What this project does

Answers clinical questions about GLP-1 receptor agonist drugs (semaglutide,
tirzepatide, liraglutide) by retrieving and synthesizing evidence from 250 real
PubMed abstracts — grounded strictly in peer-reviewed literature, with citations.

**Example question:** "Which drug produces greater weight loss, tirzepatide or semaglutide?"

**System answer:** "Tirzepatide produces greater weight loss than semaglutide based
on pooled head-to-head analyses (pooled MD = -5.19 kg, 95% CI: -7.96 to -2.42,
p=0.0002) [PMID 42211533 | RESULTS]. In a network meta-analysis, tirzepatide 15mg
resulted in the greatest percent weight reduction (MD -17.97%) compared to
semaglutide 7.2mg (MD -14.66%) [PMID 42207966 | RESULTS]."

## Architecture

```
PubMed API (250 abstracts)
        ↓
Structure-aware chunking (per IMRaD section)
        ↓
all-MiniLM-L6-v2 embeddings → Chroma vector store
        ↓
LangGraph multi-agent pipeline:
  Supervisor → Search agent → Summarize agent → Critique agent
       ↑_______________revision loop___________________|
        ↓
Ragas evaluation harness (faithfulness, relevancy, precision, recall)
        ↓
GitHub Actions CI/CD — eval-regression gate on every PR
```

## Key technical decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` (local) | Zero cost, no rate limits, acceptable quality for narrow domain |
| Vector DB | Chroma (local) | Open-source, file-based persistence, no infra required |
| LLM | Llama 4 Scout via Jetstream | Free, OpenAI-compatible API, no quota limits |
| Eval framework | Ragas | LLM-as-a-judge metrics, measures faithfulness not just BLEU/ROUGE |
| Tracing | Arize Phoenix (local) | Open-source, full OpenTelemetry support, no account required |
| Orchestration | LangGraph | Explicit state graph with conditional edges, supports revision loops |

## What was built step by step

| Step | What | Key concept demonstrated |
|---|---|---|
| 1 | PubMed ingestion via E-utilities API | Rate limiting, idempotent caching, query-hashed cache invalidation |
| 2 | Chunking + embeddings → Chroma | Structure-aware chunking, cosine similarity, vector DB design |
| 3 | Manual RAG (no framework) | Every mechanical step explicit — embed, retrieve, prompt, generate |
| 4 | LangGraph multi-agent rebuild | Supervisor pattern, state graphs, critique-revision loop |
| 5 | Ragas evaluation harness | Golden test set construction, faithfulness/precision/recall metrics |
| 6 | Arize Phoenix tracing | OpenTelemetry auto-instrumentation, span-level LLM observability |
| 7 | Attention from scratch (NumPy) | Real pretrained weights, verified against model output (diff: 0.000000) |
| 8 | Docker + GitHub Actions CI/CD | Containerization, eval-regression gate blocking PRs on quality drop |

## Evaluation results (baseline)

Scored against a 15-question hand-curated golden test set across four categories:
factual, comparative, partially-answerable, and out-of-scope.

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | 0.739 | Claims supported by retrieved chunks |
| Answer Relevancy | 0.615 | Answers address the question asked |
| Context Precision | 0.506 | Retrieved chunks were relevant |
| Context Recall | 0.411 | Needed evidence was retrieved |

**Key finding:** context precision and recall identified retrieval as the bottleneck.
The embedding model preferentially retrieves BACKGROUND/INTRODUCTION sections over
RESULTS sections — a query-document semantic gap that would require query rewriting
to address. Implemented retrieval deduplication (max 2 chunks per source from a
15-candidate pool) which improved answer relevancy by +0.188 over baseline.

## From-scratch component

`src/scratch/attention.py` implements scaled dot-product attention in NumPy using
real pretrained weights extracted from `all-MiniLM-L6-v2`. Verified against the
model's actual internal activations — max absolute difference: **0.000000**.

Demonstrates understanding of:
- Q/K/V projection matrices and why they differ from raw token embeddings
- Why scaling by `sqrt(d_k)` prevents softmax saturation
- How attention weights capture linguistic relationships (visible in the heatmap)

## Project setup

### Prerequisites
- Python 3.11
- Docker Desktop
- A free [Jetstream](https://jetstream-cloud.org) account (for LLM access)
- A free [NCBI API key](https://www.ncbi.nlm.nih.gov/account/settings/) (for PubMed)

### Local setup

```bash
git clone https://github.com/YOUR_USERNAME/glp1-research-assistant
cd glp1-research-assistant
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env
# Fill in JETSTREAM_API_KEY, NCBI_API_KEY, NCBI_EMAIL in .env
```

### Run ingestion and embedding (one-time setup)

```bash
python src/ingest_pubmed.py        # fetches 250 PubMed abstracts
python src/build_vector_store.py   # chunks, embeds, stores in Chroma
```

### Run the research assistant

```bash
python src/research_graph.py
```

### Run with tracing (requires Phoenix server)

```bash
# Terminal 1
python -m phoenix.server.main serve

# Terminal 2
python src/run_with_tracing.py
# Open http://localhost:6006 to view traces
```

### Run evaluation

```bash
python src/run_pipeline_for_eval.py   # run pipeline on golden test set
python src/evaluate.py                # score with Ragas
```

### Docker

```bash
docker compose up -d           # start pipeline + Phoenix
docker compose exec pipeline python src/research_graph.py
```

## CI/CD

GitHub Actions runs on every PR:
- **Lint:** ruff checks for syntax errors and unused imports
- **Eval gate:** runs the full pipeline against the golden test set and fails
  the build if any metric drops below threshold — blocking the PR merge

## Learning notes

Each step has a detailed learning note in `docs/learning_notes/` covering:
- What was built and why
- What broke and how it was fixed
- Likely interview questions and answers

## Tech stack

Python · LangGraph · Chroma · sentence-transformers · Ragas · Arize Phoenix ·
OpenTelemetry · Docker · GitHub Actions · NumPy · PyTorch · Llama 4 Scout