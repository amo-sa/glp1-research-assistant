# Step 6 — Tracing and observability with Arize Phoenix

## What was built
- `src/tracing.py`: OpenTelemetry setup module. Call setup_tracing() once
  at the start of any script to register Phoenix as the trace backend and
  auto-instrument LangGraph. Safe to call multiple times -- checks if a
  provider is already registered before setting up.
- `src/run_with_tracing.py`: pipeline runner with tracing enabled. Imports
  tracing.py and calls setup_tracing() BEFORE importing research_graph --
  import order matters because LangChainInstrumentor must patch LangGraph's
  internals before they're used, not after.
- Phoenix running locally at http://localhost:6006, storing traces on disk.
  Start with: python -m phoenix.server.main serve

## What Phoenix shows
- Three traces visible, one per question, each with full span hierarchy:
  LangGraph (root) -> search_agent -> summarize_agent -> critique_agent
  -> supervisor -> route_after_supervisor
- Every LangGraph node is a separate named span with its own latency,
  input state, and output state recorded
- Input/Output panels on each span show the exact state passed in and
  returned -- the full prompt including all retrieved chunks, not the
  truncated print statement version
- Total Cost: $0 -- confirms Jetstream endpoint, not a paid API
- Revision loops appear as duplicate spans: a question triggering two
  summarize/critique iterations shows two summarize_agent spans and two
  critique_agent spans in the hierarchy, making the loop visually obvious

## Key concepts

### Why observability matters for LLM systems specifically
Traditional structured logging catches mechanical failures (exceptions,
HTTP errors, slow queries). LLM failures are semantic -- the function
returned a string, but that string might be a hallucination or an answer
that ignored retrieved context. You can't detect semantic failures from
status codes. Span-level tracing captures the exact LLM inputs/outputs
so you can inspect what actually happened, not just whether it errored.

### OpenTelemetry
Vendor-neutral CNCF observability standard. Our instrumentation uses
OpenTelemetry's TracerProvider and OTLP exporter, not Phoenix-specific
APIs. This means the same spans could be sent to Jaeger, Honeycomb, or
any other OTLP-compatible backend by changing one URL. Standard protocols
prevent vendor lock-in -- same argument as using the OpenAI-compatible
API interface rather than a proprietary SDK.

### Auto-instrumentation vs. manual instrumentation
LangChainInstrumentor().instrument() hooks into LangGraph's internals
automatically -- we didn't modify research_graph.py at all. This is
auto-instrumentation: the library handles span creation for every node
execution and LLM call without explicit tracing code in the application.
Manual instrumentation (adding trace.get_tracer().start_as_current_span()
around specific functions) is more granular but requires changing
application code. Auto-instrumentation is the right default for framework
code; manual instrumentation adds value for custom business logic that
the framework doesn't know about (e.g. the Chroma query in search_agent,
which doesn't appear as its own span in the current setup).

### BatchSpanProcessor
Buffers spans and sends them in batches rather than one HTTP request per
span. A pipeline execution that generates 6-10 spans (search + 2x
summarize + 2x critique + supervisor + routing) would otherwise make 6-10
HTTP calls to Phoenix. Batching reduces overhead, especially for rapid
span generation during a revision loop.

### Traces vs. spans
A trace is one complete pipeline execution (one run_research() call).
A span is one unit of work within that trace (one agent node, one LLM
call). Spans nest inside each other forming a tree -- the LangGraph root
span contains all agent spans, which would contain LLM call spans if we
had manual instrumentation at that level. Trace ID connects all spans
from the same execution so you can reconstruct the full picture.

## Phoenix UI features worth knowing
- Tracing: per-trace and per-span inspection (what we used)
- Evaluators: LLM-as-a-judge scoring on every incoming trace automatically
  -- in production, run faithfulness evaluation on all queries continuously
  rather than just against a static golden test set
- Datasets & Experiments: store trace sets and run A/B experiments
  comparing prompt versions -- could compare summarize_agent system prompt
  v1 vs v2 against the same questions with real trace evidence
- Dashboards: aggregate latency, revision loop frequency, error rate
  charts across many traces -- useful once you have 50+ traces
- Prompts: version and track system prompts alongside the traces they
  produced -- ties prompt changes to measurable outcome changes

## Why Phoenix over LangSmith
LangSmith is LangChain's hosted tracing product -- excellent but requires
an account and has usage limits on the free tier. Phoenix is fully
open-source, runs locally, stores traces on disk, requires zero external
accounts or API keys. Same reasoning as Chroma over Pinecone: zero cost,
zero external dependency, completely self-contained. Both use
OpenTelemetry under the hood so the instrumentation code is portable.

## Known gaps / things not yet handled
- Chroma queries in search_agent are not traced as separate spans -- they
  appear inside the search_agent span but without their own timing or
  input/output recording. Manual instrumentation would be needed to add
  this granularity.
- Phoenix project name is hardcoded as "glp1-research-assistant" in
  tracing.py -- should be configurable via environment variable for
  multi-environment setups (dev vs. prod).
- Traces are not integrated with the eval harness -- eval_results.json
  and Phoenix traces are separate artifacts. A production system would
  link eval scores back to specific trace IDs so you can click from a
  low faithfulness score directly to the trace that produced it.
- run_pipeline_for_eval.py doesn't call setup_tracing() -- eval pipeline
  runs are not traced. Worth adding so eval runs are also observable.

## Likely interview questions tied to this step
- "Why do you need observability for an LLM system beyond standard logging?"
  -> LLM failures are semantic (wrong answer, hallucination) not mechanical
  (exception, HTTP error). Standard logs tell you the function ran; traces
  tell you what the function actually did -- the exact prompt, the exact
  response, the latency of each step.
- "What is OpenTelemetry and why did you use it?" -> vendor-neutral CNCF
  observability standard. Spans exportable to any OTLP-compatible backend
  (Phoenix, Jaeger, Honeycomb) by changing one URL. Prevents vendor lock-in.
- "What's the difference between a trace and a span?" -> trace = one
  complete pipeline execution, span = one unit of work within it. Spans
  nest to form a tree representing the full execution hierarchy.
- "What would you add to this observability setup in production?" ->
  continuous LLM-as-a-judge evaluation on every incoming trace (not just
  golden test set), alerting when faithfulness drops below threshold,
  linking eval scores to specific trace IDs, dashboards showing revision
  loop frequency and per-agent latency trends over time.
- "Why auto-instrumentation instead of manual?" -> auto-instrumentation
  covers all framework-level calls (LangGraph nodes, LLM calls) without
  modifying application code. Manual instrumentation adds value for
  custom logic the framework doesn't know about. Use both: auto for the
  framework, manual for your specific business logic that needs tracing.