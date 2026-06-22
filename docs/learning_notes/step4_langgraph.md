# Step 4 — LangGraph multi-agent pipeline

## What was built
- `src/research_graph.py`: multi-agent research pipeline using LangGraph.
- Four nodes: search_agent, summarize_agent, critique_agent, supervisor.
- Shared ResearchState TypedDict: question, chunks, draft_answer, critique,
  final_answer, iterations. Every node reads full state, returns only the
  fields it changed -- LangGraph merges updates automatically.
- Rules-based supervisor (no LLM call for routing): reads critique field,
  routes to END if starts with "APPROVED", loops back to summarize_agent
  if "REVISE", forces finalization if MAX_ITERATIONS (3) reached.
- Conditional edge: add_conditional_edges("supervisor", route_fn, mapping)
  -- route_fn inspects state and returns a string key; mapping resolves
  that key to the actual next node. Routing logic separated from node logic
  by design (nodes update state, edges control flow).
- Graph compiled from StateGraph(ResearchState) -- compilation validates
  that all referenced node names exist and graph structure is sound.

## What the three test questions showed
- Q1 (semaglutide + body weight): approved in 1 iteration. Answer slightly
  more precise than Step 3's manual RAG version (explicitly distinguished
  oral vs subcutaneous routes of administration).
- Q2 (GI side effects comparison): the money result. First draft made an
  overconfident claim about tirzepatide's comparative GI risk. Critique
  agent returned REVISE. Supervisor looped back. Second draft was more
  calibrated: explicitly stated a direct comparison couldn't be made,
  distinguished what each PMID did and didn't actually say. Approved in
  iteration 2. This is a concrete, demonstrable example of the critique
  loop catching a real (if subtle) faithfulness issue and producing a
  measurably better answer on the second attempt.
- Q3 (metformin -- out of scope): correct refusal in 1 iteration. Critique
  agent approved the refusal quickly (nothing to fact-check when the
  summarize agent correctly declines to answer). Minor formatting difference
  from Step 3 ("[Insufficient information]" appended in brackets) -- shows
  low-temperature outputs aren't perfectly deterministic even at temp=0.1,
  though the substance was identical.

## Key LangGraph concepts
- StateGraph: the graph object. Nodes and edges registered against it,
  then compiled. Compilation validates structure.
- TypedDict state: shared whiteboard every node reads from and writes to.
  Nodes return partial dicts (only their changed fields); LangGraph merges
  with last-write-wins semantics per field. For fields that need to
  accumulate (e.g. a message history), use Annotated[list, operator.add]
  reducer -- we didn't need this since all our fields are overwrite-style.
- Nodes: plain Python functions (state: ResearchState) -> dict. No special
  interface required beyond that signature.
- Edges: add_edge(A, B) for fixed transitions, add_conditional_edges(A,
  routing_fn, mapping) for branching. Routing functions return string keys
  that the mapping resolves to node names. Separation of state updates
  (nodes) from routing (edges) is intentional design.
- START / END: special sentinel nodes imported from langgraph.graph.
  START is the entry point; END is the terminal node. graph.add_edge(START,
  "first_node") sets the entry point.
- graph.compile(): validates and freezes the graph. Returns a runnable
  that accepts an initial state dict via .invoke().
- graph.invoke(initial_state): runs the graph to completion, returns the
  final state dict. Synchronous; for async use graph.ainvoke().

## Why this over manual RAG (Step 3)
- Compound questions: supervisor could route to search_agent multiple times
  with different sub-queries before summarizing (not implemented here but
  the architecture supports it trivially -- just add a loop edge back to
  search_agent from supervisor).
- Summarization and critique are separated: critique agent has fresh context,
  isn't statistically primed to defend what it just wrote (demonstrated
  concretely with Q2).
- Pipeline can loop: Q2 actually triggered this -- first draft revised,
  second draft approved. Manual RAG had no mechanism to catch and correct
  a bad first answer.
- Every intermediate result is inspectable in final state (chunks, draft,
  critique, final answer) -- not just the output string.

## Supervisor pattern vs. alternatives
- Supervisor (what we built): reactive, one-step-at-a-time routing. Looks
  at current state after each agent, decides next step. Doesn't know the
  full path upfront -- discovers it as execution proceeds.
- Plan-and-execute: planner agent generates an explicit ordered plan first,
  executor works through it mechanically. Good when task structure is
  predictable and you want the plan inspectable/loggable upfront.
- Hierarchical multi-agent: supervisors can themselves be nodes in a
  higher-level supervisor's graph -- for very complex multi-team workflows.
- Rules-based vs. LLM-driven supervisor: our supervisor uses if/else logic
  (reads critique prefix string, no LLM call). An LLM-driven supervisor
  would call the LLM to decide routing -- more flexible for unpredictable
  cases, but adds latency and a potential failure mode (LLM picks wrong
  node). Rules-based is easier to debug, test, and reason about; prefer
  it unless the routing logic is genuinely too complex for explicit rules.

## Known gaps / things not yet handled
- Single retrieval pass per question: the architecture supports looping
  back to search_agent but the supervisor doesn't currently do this. A
  natural extension for compound questions: detect multiple sub-questions
  and run targeted retrievals for each before summarizing.
- Critique agent doesn't distinguish refusal from substantive answer:
  if summarize_agent correctly refuses (out-of-scope question), critique
  approves immediately -- it doesn't check for false negatives (cases
  where the model should have answered but refused). A more sophisticated
  critique would handle both directions.
- No streaming: graph.invoke() blocks until completion. For a real UI,
  graph.stream() yields intermediate state updates so the user sees
  progress (search complete, draft ready, critique running) rather than
  waiting for the full pipeline.
- Shared clients (_embedding_model, _chroma_client, _llm_client) are
  module-level globals -- fine for a script, but in a server/API context
  these should be dependency-injected rather than globally shared to
  support concurrent requests safely.
- State history not accumulated across iterations: draft_answer and critique
  are overwritten each loop rather than appended. Full fix requires
  Annotated[list, operator.add] accumulator on both fields and restructuring
  the revision prompt to show full iteration history. Currently the summarize
  agent only sees the immediately prior failure, not all prior failures.

## What broke / had to fix
- Initial summarize_agent ignored state["critique"] entirely on revision
  loops -- it received the same prompt regardless of whether it was a first
  attempt or a second attempt after critique feedback. The second attempt
  was re-rolling the same prompt at temperature 0.1 and hoping for a
  better output, not actually incorporating the feedback. Fixed by checking
  if critique.startswith("REVISE") and injecting both the prior draft and
  the specific critique complaints into the revision prompt. After the fix,
  the second draft on Q2 explicitly addressed the critique's complaint
  ("the claim doesn't specify what tirzepatide's risk is being compared
  against") rather than just generally being more careful. Concrete
  demonstration that the fix produced targeted correction, not random
  variance.
- State only keeps the most recent draft and critique, not the full history
  of all iterations. Both draft_answer and critique are plain strings with
  last-write-wins semantics -- iteration 1's draft and critique are gone by
  the time iteration 3 runs. The summarize agent only sees the immediately
  prior attempt, not the full trajectory. This could cause a non-converging
  loop where fixing issue A reintroduces issue B (whose critique was
  already overwritten). Proper fix: use LangGraph's Annotated[list,
  operator.add] accumulator pattern on draft_answers and critiques fields
  so the full history is visible. Noted as a known gap rather than fixed
  now -- meaningful refactor touching multiple nodes, and it's a stronger
  interview answer to name it precisely than to have silently fixed it.

## Likely interview questions tied to this step
- "What's the difference between a chain and a graph in LangChain/LangGraph?"
  -> chain = fixed linear sequence, no branching or looping; graph = explicit
  nodes + edges, supports conditional branching, looping, arbitrary topology.
  LangGraph is specifically for stateful, cyclic graphs; LangChain chains
  are acyclic by design.
- "Why separate the routing function from the supervisor node?" -> nodes
  update state, edges control flow -- separation lets you change routing
  logic without touching node logic, and lets LangGraph validate graph
  structure at compile time.
- "Why a rules-based supervisor instead of an LLM-driven one?" -> for our
  routing logic (read a string prefix, decide approve/revise), explicit
  rules are simpler, faster, more testable, and less likely to fail than
  an LLM call. LLM-driven routing adds value when routing decisions require
  genuine reasoning the rules can't capture.
- "Walk me through what happened on Question 2." -> search retrieved 5
  chunks, summarize drafted an overconfident claim, critique flagged it
  with REVISE, supervisor looped back to summarize, second draft was more
  calibrated and explicitly acknowledged the evidence gap, critique
  approved. Concrete demonstration that the critique loop improves
  faithfulness in a real case, not just in theory.
- "How does state flow between nodes in LangGraph?" -> shared TypedDict,
  each node receives full current state and returns a partial dict of
  updated fields, LangGraph merges with last-write-wins per field.
  Accumulator pattern (Annotated + reducer) for fields that need to grow.

## Additional test questions

Q4 (cardiovascular/metabolic benefits beyond weight loss):
  - Ran to iteration limit (3 iterations) -- the only question to do so.
  - Oscillation pattern: iteration 1 overclaimed (said there IS evidence
    of cardiovascular benefits independent of weight loss, which the excerpts
    don't support that specifically), iteration 2 underclaimed (said excerpts
    don't provide direct evidence for things they do mention), iteration 3
    landed correctly calibrated.
  - Root cause of oscillation: three interacting factors:
    (1) genuine evidence ambiguity -- excerpts mention cardiovascular risk
    reduction but don't address whether it's independent of weight loss,
    making the specific question unanswerable from available evidence;
    (2) LLM's helpfulness bias toward confident answers on first attempt,
    then overcorrection toward caution after specific critique;
    (3) state history limitation -- iteration 3 only saw iteration 2's
    failure, not iteration 1's, so it was navigating blind to earlier mistakes.
  - The convergence is the critique loop working as designed. The fact that
    it needed 3 iterations signals the summarize agent's base prompt isn't
    well-calibrated for epistemic uncertainty -- better prompt engineering
    could reduce iteration count, and the eval harness is what lets you
    measure whether prompt changes actually help.
  - Retrieval redundancy problem appeared again: two chunks from PMID
    42215399 (CONCLUSION and BACKGROUND) in top 5, using 2 of 5 context
    slots on one paper.

Q5 (semaglutide vs tirzepatide long-term effectiveness):
  - Approved in 1 iteration. Top retrieval distance was 0.1232 -- closest
    seen across all test questions, indicating the corpus had a chunk
    almost directly about this comparison.
  - Answer cited specific quantitative statistics (pooled MD = -5.19 kg,
    95% CI: -7.96 to -2.42, p=0.0002) -- these are either verbatim from
    the source chunk or confabulated. Can't tell by eyeballing the output.
    This is the highest hallucination risk case: specific numbers sound
    authoritative even when wrong, and manual inspection doesn't scale.
    Exactly the motivation for the eval harness in Step 5.
  - Key insight: RAG system performance is highly corpus-dependent. A
    question the corpus can answer directly (direct head-to-head data exists)
    will always outperform one requiring inference across partial sources,
    regardless of system architecture.