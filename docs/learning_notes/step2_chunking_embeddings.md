# Step 2 — Parsing, chunking, embeddings, vector store

## What was built
- `src/build_vector_store.py`: parses cached PubMed XML into clean article
  records, chunks each article by section, embeds every chunk, and stores
  the result in a persistent local Chroma collection.
- Parsing: `xml.etree.ElementTree` walks each `<PubmedArticle>`, pulling
  PMID, title, journal, and abstract sections. Articles with no PMID/title
  or no abstract text are skipped rather than crashing the batch.
- Chunking: structure-aware, not fixed-size. PubMed abstracts that are
  pre-segmented into labeled sections (INTRODUCTION, METHODOLOGY, RESULTS,
  CONCLUSION) are chunked along those boundaries -- one chunk per section.
  Unstructured abstracts (no `Label` attribute on `<AbstractText>`) fall
  back to a single chunk labeled "ABSTRACT", via `.get("Label", "ABSTRACT")`'s
  default value -- no separate branch of code needed, it's a natural
  consequence of the same loop handling however many `<AbstractText>`
  elements actually exist per article.
- Each chunk carries metadata (pmid, title, journal, section) so any
  retrieved chunk can be traced back to its source article for citation.
- Embedding: `all-MiniLM-L6-v2` via sentence-transformers, run locally
  (no API, no cost, no rate limit). Embeddings normalized to unit length
  (`normalize_embeddings=True`) so cosine similarity and dot product are
  equivalent.
- Storage: Chroma `PersistentClient`, collection explicitly configured for
  cosine similarity (`metadata={"hnsw:space": "cosine"}`) rather than
  relying on the library default. `upsert` (not `add`) used so the script
  is safely re-runnable.
- Result: 250 ingested articles -> 686 chunks -> 686 chunks stored and
  verified in Chroma (count, sample metadata, and sample text all checked
  directly against the persisted collection, not just trusted from the
  script's own print statements).

## What broke / had to fix
- Initial script imported `sentence_transformers` at module level, which
  meant the parsing/chunking logic couldn't even be tested without that
  (heavier, torch-dependent) library installed. Fixed by moving the import
  inside `embed_and_store()` -- a lazy import -- so parsing and chunking
  are testable in isolation. This is a real separation-of-concerns issue:
  parsing XML and embedding text are different responsibilities with
  different dependencies, and they shouldn't be coupled by import order.
- Validated parsing/chunking logic against a hand-built synthetic XML file
  covering three edge cases before trusting it on real data: a fully
  structured abstract (4 labeled sections), an unstructured single-block
  abstract (tests the fallback), and a title-only record with no abstract
  at all (tests that we skip gracefully instead of crashing). All three
  behaved as expected before running against the real 250-article corpus.

## Key concepts
- Embeddings are dense vectors where direction (not length) encodes
  semantic meaning. Cosine similarity measures the angle between two
  vectors, ignoring magnitude -- the right metric for comparing text of
  different lengths. Dot product is equivalent to cosine similarity when
  vectors are pre-normalized to unit length (our case). Jaccard similarity
  (word-set overlap) and Levenshtein distance (character edit distance)
  are lexical/surface-level metrics, not semantic -- different family
  entirely, and neither would catch "Ozempic" ~ "semaglutide".
- The embedding vector is only ever used for the search/matching step --
  the LLM never sees it. Chroma stores the raw text alongside the vector;
  retrieval returns the original text payload (the `document` field), not
  the vector itself. The vector is the index key, not the content.
- Chroma "collection" ~= a table, but schemaless on the metadata side
  (no enforced columns) and natively indexes a vector field for
  similarity search -- closer to a NoSQL document store with an ANN index
  bolted on than a strict relational table.
- `collection.upsert()` requires columnar/struct-of-arrays shape
  (`ids=[...], documents=[...], metadatas=[...]`), not array-of-structs
  (`[{...}, {...}]`) -- hence converting `chunks` (a list of dicts) into
  three parallel lists before calling it.

## Likely interview questions tied to this step
- "How did you decide on a chunking strategy?" -> structure-aware
  (section-based) chunking using the source document's own boundaries,
  vs. naive fixed-token-count splitting; explain why this avoids cutting
  semantic units in half, and the unstructured-abstract fallback.
- "Why cosine similarity over Euclidean distance for text embeddings?" ->
  text length shouldn't affect similarity; cosine is magnitude-invariant.
- "How does the LLM actually get the retrieved text?" -> walk through:
  vector search finds nearest chunk IDs -> Chroma returns the associated
  raw text/metadata stored alongside the vector -> that plain text gets
  inserted into the prompt. The embedding itself never reaches the LLM.
- "Why a local embedding model instead of a hosted API?" -> zero cost,
  zero external dependency/rate-limit risk, acceptable quality tradeoff
  for a narrow single-topic corpus at this scale; would reconsider hosted
  APIs at much larger scale or for higher-stakes retrieval quality needs.
- "How would you test this kind of parsing/chunking code?" -> built a
  small synthetic XML fixture covering known edge cases (structured,
  unstructured, malformed) rather than only eyeballing real data, so
  correctness could be verified deterministically before trusting it on
  the full corpus.
- "What's the difference between `add` and `upsert` in Chroma, and why
  does it matter?" -> `upsert` makes the embedding script idempotent /
  safely re-runnable, consistent with the caching approach from Step 1.

## Known gaps / things not yet handled
- Retrieval *quality* is unverified -- we've confirmed the pipeline runs
  correctly end-to-end and that the right text/metadata round-trips
  through storage, but we have not yet confirmed that semantically similar
  queries actually retrieve the most relevant chunks. That's explicitly
  deferred to the eval harness step (Ragas/DeepEval + golden test set),
  not assumed to be fine just because the pipeline didn't error.
- No chunk-size upper bound enforced. A RESULTS section with unusually
  long text (many reported statistics) is stored as one chunk regardless
  of length. Worth revisiting if embedding quality looks weak on long
  sections during evaluation.
- `k` (how many chunks to retrieve per query) hasn't been chosen or tuned
  yet -- that decision belongs to Step 3 (manual RAG) and should ideally
  be justified empirically later via the eval harness, not picked
  arbitrarily.