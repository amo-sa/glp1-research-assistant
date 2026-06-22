# Step 3 — Manual RAG pipeline (no framework)

## What was built
- `src/rag_query.py`: full RAG pipeline built by hand, no LangChain or
  LangGraph. Every mechanical step explicit and visible:
  1. Embed the user's question (same model as corpus: all-MiniLM-L6-v2)
  2. Query Chroma for top-k most similar chunks (k=5)
  3. Format a strict prompt: system instruction + retrieved chunks with
     citation headers + question
  4. Call Llama 4 Scout via Jetstream's OpenAI-compatible API
  5. Return a dict containing question, chunks, answer, and messages --
     not just the answer string, so the eval harness (Step 5) can inspect
     every layer of the pipeline
- LLM: Llama 4 Scout via Jetstream (free, no quota, OpenAI-compatible API,
  accessible from anywhere via Open WebUI proxy + Bearer token)
- temperature=0.1: low temperature for faithful, consistent answers rather
  than creative/varied ones. Temperature affects probability distribution
  over tokens (divides logits before softmax), not a binary random/not-random
  switch -- higher temp = flatter distribution = more token variety.

## What the three test questions showed
- Q1 (semaglutide + body weight): retrieval distances tight (0.14-0.23),
  answer cited three separate PMIDs inline, stayed grounded in retrieved
  text. Clean pass.
- Q2 (GI side effects comparison): model gave a calibrated *partial* answer
  -- used what one chunk supported (tirzepatide has higher comparative risk
  for severe GI events vs semaglutide per PMID 42201797), then explicitly
  flagged that a comprehensive comparison wasn't available and appended the
  "insufficient information" phrase. This is the hardest behavior to get
  right and it worked correctly.
  - Also surfaced a real retrieval behavior: two chunks from the same paper
    (PMID 42211533, CONCLUSION and RESULTS) appeared in the top 5, using
    2 of 5 context slots on one source. This is a known RAG "redundancy
    problem" -- worth addressing later with a max-chunks-per-source limit
    if eval metrics show it's hurting answer quality.
- Q3 (metformin dosage -- deliberately out of scope): retrieval distances
  were 0.42-0.54, a completely different scale from the in-scope questions
  (0.14-0.33). The model correctly fired the escape hatch: "The available
  abstracts do not contain sufficient information to answer this question."
  No hallucination, no parametric memory leakage. The strict system prompt
  and explicit "say I don't know" instruction worked as designed.
  This is a concrete, demonstrable example of the faithfulness design
  working -- retrieval distance as a signal of "nothing relevant in corpus"
  + model correctly declining rather than confabulating.

## What broke / had to fix
- Embedding model was reloaded on every call to embed_question() -- a new
  SentenceTransformer() instance per question. Fixed by loading once at
  module level as _embedding_model (underscore prefix = internal/private
  by convention) and reusing across calls. Matters especially for Step 5
  where the eval harness will call rag_query() in a loop over many questions.

## Key design decisions
- Strict system prompt: "answer ONLY using the provided excerpts" + explicit
  "say so if insufficient" escape hatch. The escape hatch is necessary
  because without it, instruction-tuned models tend to produce a confident
  answer rather than admit uncertainty -- their training optimizes for
  being helpful, which can conflict with staying faithful to retrieved
  context. Giving explicit permission to decline dramatically reduces
  confabulation on out-of-scope questions.
- Context-first prompt structure: retrieved chunks before the question,
  separated by a --- divider. Helps avoid the "lost in the middle" problem
  where models pay less attention to content in the middle of a long prompt.
- Citation headers prefixed to each chunk: [PMID | section | journal]
  so the model can reference sources inline without post-hoc matching.
- rag_query() returns a full dict (question, chunks, answer, messages),
  not just the answer string -- forward-looking design so the eval harness
  in Step 5 can inspect every layer without changing this function.
- Chroma and OpenAI clients also initialized once rather than per-call
  for efficiency.

## API / tool design notes
- OpenAI-compatible API format: messages as a list of
  {"role": ..., "content": ...} dicts. Roles: "system" (standing
  instructions, model trained to follow these), "user" (human turn),
  "assistant" (model's prior turns, used in multi-turn conversations).
- response.choices[0].message.content: choices is a list because the API
  supports n>1 to generate multiple candidate completions in one call.
  We use the default n=1, so choices always has one item.
- collection.query() returns batch-first results (outer list = one entry
  per query sent) because the API supports multiple queries at once.
  [0] peels off the single-query wrapper to get the actual top-k results.

## Likely interview questions tied to this step
- "Walk me through exactly how RAG works mechanically." -> embed question
  with same model as corpus -> vector similarity search -> retrieve top-k
  chunks + their raw text payloads -> inject plain text into prompt ->
  LLM generates grounded answer. Emphasize: the LLM never sees the vector,
  only the text payload stored alongside it.
- "How did you handle the case where the answer isn't in the corpus?" ->
  two-layer defense: (1) retrieval distances expose when nothing relevant
  exists (high distance = no good match), (2) strict system prompt with
  explicit "say so if insufficient" instruction. Demonstrate with the
  metformin example: distances 0.42-0.54 vs. 0.14-0.23 for in-scope
  questions, and model correctly declined.
- "Why low temperature for RAG?" -> optimizing for faithfulness to
  retrieved context, not response diversity. Temperature flattens/sharpens
  the token probability distribution -- high temp increases variance in
  word choice, doesn't improve grounding.
- "Why Llama 4 Scout over DeepSeek R1 or gpt-oss-120b for this task?" ->
  both R1 and gpt-oss-120b are reasoning models optimized for problems
  that benefit from extended chain-of-thought. RAG generation is mostly
  "read these chunks and summarize faithfully" -- that doesn't need deep
  reasoning, it needs instruction-following and faithfulness. Reasoning
  models add latency without improving the bottleneck (retrieval quality).
- "What's the redundancy problem in retrieval and how would you fix it?" ->
  multiple chunks from the same source can appear in top-k, crowding out
  diverse sources. Fix: max-chunks-per-source limit, or MMR (Maximal
  Marginal Relevance) which penalizes retrieved chunks similar to
  already-selected ones.

## Known gaps / things not yet handled
- k=5 is an untested starting point. Needs empirical tuning via eval
  harness in Step 5 -- try k=3, 5, 10 and compare faithfulness/precision.
- Redundancy problem identified (Q2 pulled two chunks from same paper)
  but not yet addressed. Worth fixing if eval metrics show it hurts quality.
- Retrieval distance threshold not enforced -- if all top-k chunks have
  high distances (like the metformin question), we still pass them all to
  the LLM and rely on the system prompt to handle it. A cleaner approach
  would be to detect "no relevant chunks found" before even calling the
  LLM and return the refusal directly. Deferred to after eval harness
  gives us real numbers.
- No conversation history / multi-turn support yet. Each call to rag_query()
  is stateless -- the LLM has no memory of prior questions. Multi-turn
  is out of scope for now but worth flagging.