# Step 1 — PubMed ingestion

## What was built
- `src/ingest_pubmed.py`: pulls PubMed abstracts about GLP-1 drugs (semaglutide,
  tirzepatide, liraglutide) and weight loss/obesity, using NCBI's E-utilities API.
- Two-step API pattern: `esearch` (query -> list of PMIDs) then `efetch`
  (PMIDs -> full XML records), fetched in batches of 50.
- Raw XML responses cached to `data/raw/{query_hash}/batch_NNN.xml`. The cache
  directory is namespaced by a SHA-256 hash (first 10 chars) of the search
  query string, so changing the query automatically gets a fresh cache
  directory instead of silently serving stale batches from a different query.
- Rate limiting via a fixed `time.sleep()` between batches (0.11s with an API
  key for the 10 req/sec tier, 0.34s without for the 3 req/sec tier).
- Result: 250 PMIDs pulled, 5 batches of 50, all cached locally.

## What broke / had to fix
- First pass cached batches as a flat `data/raw/batch_000.xml` etc. -- no
  connection between the cached file and the query that produced it. If the
  search query changed later, the script would happily serve old cached
  batches under the new query, with no error or warning. Fixed by hashing
  the query string and nesting batches under `data/raw/{hash}/`.
- Initial local environment setup hiccups (not code bugs, just environment):
  - Windows cmd.exe doesn't support bash brace expansion (`mkdir {a,b,c}`) --
    switched to Git Bash, which gave proper bash syntax.
  - venv activation path differs on Windows: `.venv/Scripts/activate`, not
    `.venv/bin/activate` like Mac/Linux.
  - `.env` values should never be quoted -- `python-dotenv` includes literal
    quote characters in the value if you wrap it in quotes, which would have
    sent a malformed API key to NCBI.

## What the raw data actually looks like
- One XML file = up to 50 `<PubmedArticle>` records, not one record per file.
- Abstracts are often pre-segmented by PubMed itself into labeled sections
  (`<AbstractText Label="INTRODUCTION">`, `"METHODOLOGY"`, `"RESULTS"`, etc.)
  for structured/clinical abstracts -- this gives us a natural, semantically
  meaningful chunking boundary for Step 2, rather than an arbitrary
  fixed-token-count split.
- XML entities (e.g. `&#x2009;`, `&lt;`) are present in the raw text and are
  auto-decoded by `xml.etree.ElementTree` when reading `.text` -- no manual
  unescaping needed.

## Likely interview questions tied to this step
- "Walk me through how you'd design an ingestion pipeline for a third-party
  API with rate limits." -> two-phase search/fetch, batching, fixed-delay
  throttling vs. exponential backoff, why caching raw responses matters.
- "Why cache raw API responses to disk instead of processing them inline?"
  -> decouples fetch cost from parsing iteration; lets you fix parsing bugs
  without re-hitting the network; standard "bronze layer" pattern.
- "How do you make an ingestion script idempotent / safe to re-run?"
  -> cache-existence check before fetching; discuss the stale-cache bug we
  hit and how hashing the query input fixed it.
- "What's the difference between a generator and a list here, and why does
  it matter?" -> `chunked()` yields lazily instead of materializing all
  chunks upfront; at this scale (250 items) it's stylistic, but the pattern
  matters at scale.
- "Why not just dump the whole 250-PMID list in a single API call?" -> NCBI's
  documented best practice caps batch size for the XML endpoint; smaller
  batches also bound the blast radius of a failed request (you don't lose
  246 already-fetched records because record 247 errored).
- "How would you handle a partially-failed run?" -> honest answer right now:
  not fully handled yet -- if `efetch_batch` raises mid-way through a batch
  loop, earlier batches stay cached (good), but there's no retry/resume
  logic for the failed batch itself. Worth flagging as a known gap.

## Known gaps / things not yet handled
- No retry logic on transient network failures (only `raise_for_status()`,
  which fails hard rather than retrying). Acceptable for a portfolio project
  at this scale, but worth naming explicitly if asked "is this
  production-ready."
- `esearch` itself isn't cached -- only the `efetch` batches are. Re-running
  the script always re-does the search step. Low cost (one request) so left
  as-is intentionally.