# Step 8 — Docker and GitHub Actions CI/CD

## What was built
- `Dockerfile`: containerizes the full pipeline. Base image python:3.11-slim,
  installs system dependencies, copies requirements.txt separately from code
  (exploits Docker layer caching so pip install only reruns when requirements
  change, not on every code change), installs all Python packages, copies
  project code, creates a Vertex AI stub to fix the Ragas 0.3.9 import bug.
- `docker-compose.yml`: orchestrates two services -- pipeline (our code) and
  phoenix (tracing UI). Services communicate by name on a shared Docker network.
  Data volumes mount local ./data and ./chroma_db into the container so
  pipeline outputs persist after the container stops.
- `.dockerignore`: excludes .env (secrets), .venv (reinstalled fresh inside
  container), data/raw and chroma_db (large, mounted as volumes instead),
  .git (not needed inside container).
- `requirements.txt`: minimal direct dependencies with pinned versions.
  Added langchain-core==1.4.8 explicitly to resolve transitive conflict
  between langgraph (requires >=1.4.7) and langchain-openai (was requiring
  <1.0.0 at 0.3.35 -- resolved by upgrading to 1.3.3).
- `.github/workflows/ci.yml`: two jobs:
    - lint: ruff checks on every push to main and every PR
    - eval: full pipeline + Ragas scoring on PRs only, fails build if any
      metric drops below threshold (the eval-regression gate)
- `README.md`: full project documentation including architecture diagram,
  step-by-step build summary, eval results, setup instructions.

## How Docker image building works
Each Dockerfile instruction creates a layer. Layers are cached -- if a
layer's inputs haven't changed since the last build, Docker reuses the
cached layer. Critical optimization: COPY requirements.txt before COPY . .
means the slow pip install layer is cached on code-only changes and only
reruns when requirements.txt actually changes.

Layer order in our Dockerfile:
  1. FROM python:3.11-slim       -- base OS, always cached after first pull
  2. WORKDIR /app               -- cached
  3. RUN apt-get install        -- cached (build-essential, git)
  4. COPY requirements.txt      -- invalidated if requirements.txt changes
  5. RUN pip install            -- slow (~8 min), cached after first build
  6. COPY . .                   -- invalidated on any code change
  7. RUN mkdir + stub           -- runs whenever layer 6 runs

## Docker dependency conflict lesson
Minimal requirements.txt (direct deps only) caused pip resolution failure
in a clean Docker environment -- langchain-openai 0.3.35 required
langchain-core<1.0.0 but langgraph 1.2.6 required langchain-core>=1.4.7.
These are incompatible; no single version satisfies both.

On the local machine this worked because packages were installed in a
specific sequence that happened to land at a compatible state. A fresh
Docker environment resolves everything simultaneously and hits the wall.

Fix: explicitly pin langchain-core==1.4.8 and upgrade langchain-openai to
1.3.3 (compatible with langchain-core 1.4.x). Also added the Vertex AI stub
as a RUN command in the Dockerfile since the local stub lived in .venv which
is excluded by .dockerignore.

Production alternative: pip-compile (from pip-tools) generates a fully
resolved lockfile from a minimal input file -- all transitive dependencies
pinned, reproducible everywhere, no manual conflict hunting.

## How the eval-regression gate works
The entire mechanism is a Unix exit code:
- evaluate.py exits with code 0 if all metrics pass thresholds
- evaluate.py exits with code 1 if any metric fails
- GitHub Actions treats non-zero exit code as step failure
- A failed step fails the job
- A failed job blocks PR merge (if branch protection rules are set)

No special GitHub integration needed -- it's just exit codes, the same
mechanism that makes any CI system work with any tool.

## Why eval only on PRs, not on every push
The eval job makes 15+ LLM API calls. Running on every push to main would:
- Burn through Jetstream API quota unnecessarily
- Slow down the feedback loop for simple commits (docs, README changes)
- Add no value since main is already the "known good" state

PRs are the right gate -- that's when code that might introduce regressions
is trying to land. Direct pushes to main are for maintenance that doesn't
affect LLM behavior.

## Threshold calibration
Initial thresholds (0.70/0.70/0.60/0.60) were aspirational -- set before
running eval. Actual baseline scores (faithfulness 0.739, relevancy 0.615,
precision 0.506, recall 0.411) mean 3/4 metrics would fail immediately.

The right approach: set thresholds to match current baseline with a small
buffer for judge LLM variance, then tighten as the system improves:
  faithfulness:      0.70  (achieving ~0.72-0.74)
  answer_relevancy:  0.55  (achieving ~0.61)
  context_precision: 0.45  (achieving ~0.48-0.51)
  context_recall:    0.38  (achieving ~0.40-0.43)

This way the gate catches regressions from the current baseline rather than
blocking all PRs because aspirational targets weren't met on day one.

## GitHub Secrets
API keys are never committed to the repo. They're stored as GitHub repository
secrets (Settings → Secrets and variables → Actions) and injected into the
CI environment via ${{ secrets.SECRET_NAME }} syntax. The CI runner never
logs secret values. This is the standard pattern for any CI/CD credential
management -- same principle as .env locally, but managed by the platform.

## Docker vs. GitHub Actions runner for CI
The CI workflow runs directly on the GitHub Actions ubuntu-latest runner,
not inside our Docker container. This is intentional:
- GitHub Actions already provides a clean, ephemeral Linux environment
- Running Docker-inside-Docker adds complexity without benefit for CI
- The runner environment is fully reproducible via requirements.txt
Docker is for local development reproducibility and potential deployment;
GitHub Actions handles CI reproducibility through its own runner isolation.

## Likely interview questions tied to this step
- "Why containerize an ML/AI pipeline?" -> reproducibility across machines
  and environments, eliminates "works on my machine" problems, dependency
  isolation, easier deployment. Demonstrate understanding of layer caching
  as the key build optimization.
- "How does your eval-regression gate work?" -> evaluate.py exits with
  code 1 if metrics fail thresholds, GitHub Actions treats non-zero exit
  as failure, which blocks PR merge. The mechanism is exit codes -- no
  special LLM/AI integration needed, just standard CI conventions.
- "Why run eval only on PRs?" -> cost/quota management, PRs are the right
  gate point (that's when regressions would be introduced), direct pushes
  to main are already known-good state.
- "How do you handle secrets in CI?" -> GitHub repository secrets injected
  as environment variables, never committed to repo, never logged by runner.
  Same principle as .env locally but managed by the platform.
- "What's the difference between Docker and docker-compose?" -> Docker
  builds and runs single containers; docker-compose orchestrates multiple
  containers as a unit (shared network, volume management, dependency
  ordering via depends_on, single up/down command for the whole stack).
- "Why separate the pipeline runner from the eval scorer into two scripts?"
  -> dependency isolation (langgraph vs ragas have conflicting langchain-core
  requirements), allows re-running scoring without re-running expensive
  pipeline LLM calls, clearer separation of concerns, each script has one job.

## Known gaps / things not yet handled
- Chroma vector store rebuild in CI: the workflow caches chroma_db between
  runs but a fresh repo has no cache. First CI run would need to rebuild
  from scratch, which requires the raw PubMed XML (not committed). A
  production setup would commit a small sample corpus or fetch it as part
  of CI setup.
- Branch protection rules not configured: the eval gate fails the job but
  doesn't actually block merges unless branch protection is enabled in
  GitHub repo settings (Settings → Branches → Require status checks to
  pass before merging). Worth enabling for a real team project.
- Eval thresholds still set at aspirational values (0.70/0.70/0.60/0.60)
  rather than calibrated to current baseline. Should be updated before
  running the first real PR through the gate.
- No Slack/email notification on eval failure -- in production you'd want
  the team alerted when the gate fails, not just a red CI badge.