"""
Step 6: Observability via Arize Phoenix.

This module sets up OpenTelemetry tracing with Phoenix as the backend.
Call setup_tracing() once at the start of any script that should emit traces.

How it works:
  1. Phoenix runs as a local server (started separately with `phoenix serve`)
  2. This module registers an OTLP exporter pointing at that server
  3. openinference-instrumentation-langchain auto-instruments LangGraph --
     every node execution and LLM call automatically becomes a span without
     any changes to research_graph.py
  4. Spans are sent to Phoenix over HTTP and stored on disk
  5. Phoenix UI (http://localhost:6006) shows traces, span details,
     LLM inputs/outputs, and latency breakdowns

Why OpenTelemetry:
  OpenTelemetry is a vendor-neutral observability standard (CNCF project).
  Using it means our instrumentation isn't locked to Phoenix -- we could
  point the same spans at Jaeger, Honeycomb, or any other OTLP-compatible
  backend by changing one URL. This is the same reason you'd use a standard
  logging interface rather than a vendor-specific SDK.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

# Phoenix's default OTLP endpoint when running locally
PHOENIX_OTLP_ENDPOINT = "http://localhost:6006/v1/traces"

# Project name shown in the Phoenix UI -- groups traces by project
PHOENIX_PROJECT_NAME = "glp1-research-assistant"


def setup_tracing():
    """Register Phoenix as the OpenTelemetry trace backend and
    auto-instrument LangChain/LangGraph.

    Call this once at the start of any script before running the pipeline.
    Safe to call multiple times -- checks if a provider is already registered.
    """
    # Avoid double-registration if called multiple times in the same process
    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        return

    # Build the exporter -- sends spans to Phoenix over HTTP
    exporter = OTLPSpanExporter(endpoint=PHOENIX_OTLP_ENDPOINT)

    # BatchSpanProcessor buffers spans and sends them in batches rather than
    # one HTTP request per span -- more efficient, especially for a pipeline
    # that generates many spans quickly (search + summarize + critique + supervisor)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Register as the global OpenTelemetry provider so all instrumentation
    # in this process uses it
    trace.set_tracer_provider(provider)

    # Auto-instrument LangChain/LangGraph -- hooks into the framework internals
    # to create spans for every node execution and LLM call automatically.
    # This is why we don't need to modify research_graph.py at all.
    LangChainInstrumentor().instrument()

    print(f"Phoenix tracing enabled -> {PHOENIX_OTLP_ENDPOINT}")
    print(f"View traces at: http://localhost:6006 (project: {PHOENIX_PROJECT_NAME})")