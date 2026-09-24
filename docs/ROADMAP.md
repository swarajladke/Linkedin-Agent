# Pilot Phase 1: Roadmap, Architectural Constraints & Progress Tracker

**Project:** Pilot — Goal-Directed Autonomous Career Agent  
**Phase:** Phase 1 (World Model & Ingestion Engine) **Status:** **7 of 7 Steps Completed (100%)** | **Phase 1 Complete**  
**Repository:** [`swarajladke/Linkedin-Agent`](https://github.com/swarajladke/Linkedin-Agent.git) (Branch: `main`)

---

## 1. Core Architectural Tenets & Hard Constraints

Pilot is designed around a closed-loop decision framework:
$$\text{Observe} \longrightarrow \text{Diagnose} \longrightarrow \text{Choose Action} \longrightarrow \text{Predict Outcome} \longrightarrow \text{Act} \longrightarrow \text{Record Actual Outcome} \longrightarrow \text{Adapt Strategy}$$

### Non-Negotiable Engineering Rules
1. **Auditable Decision Loop over LLM Generation**: The system prioritizes calibration, prediction error tracking, and verifiable world model state over cover letter generation.
2. **Deterministic & Debuggable**: Hand-rolled loops only. No LangChain, LangGraph, or opaque graph databases. Python 3.11+ and PostgreSQL 16 with `pgvector`.
3. **No LinkedIn Scraping or Logged-in Automation**: Inputs derive purely from candidate-provided resumes, public GitHub APIs, and permitted public job postings.
4. **Pre-Execution Falsifiable Predictions**: Every autonomous action MUST record its reasoning and a falsifiable probabilistic prediction in the database *before* execution.
5. **Strict Grounding Guarantee**: Every factual claim made about a candidate must link directly to an atomic row in `evidence_claims` with an exact `source_excerpt` (verbatim text span) and `source_url`. Ungrounded or unverified claims are dropped.
6. **Single Source of Truth**: No duplicated columns across joins. Strategy versions are joined through `strategies.id` with parent lineage; outcome Brier scores are computed automatically in-database via trigger.

---

## 2. Progress Dashboard

| # | Step Name | Scope & Key Deliverables | Status | Commit SHA | CI Status |
| :-: | :--- | :--- | :-: | :---: | :---: |
| **1** | **Scaffold + Docker + Config** | Pyproject, Ruff, Pytest, Docker Compose (pgvector 16), Pydantic Settings (`SecretStr`) | **COMPLETED** | `c84ffa1` | Verified |
| **2** | **World Model + Migration 0001** | 15 SQLAlchemy 2.0 models, Alembic sync migration, polymorphic discriminators, Brier trigger, constraints | **COMPLETED** | `da9d9d2` | [Green (Run 35972591152)](https://github.com/swarajladke/Linkedin-Agent/actions/runs/35972591152) |
| **3** | **Resume Reader + GitHub Client** | Verbatim text/PDF span reader with exact locators, rate-limited public GitHub REST client, local disk cache | **COMPLETED** | `a430481` | [Green (Run 35974080710)](https://github.com/swarajladke/Linkedin-Agent/actions/runs/35974080710) |
| **4** | **Grounded Extractor + Dedup** | Structured LLM extraction, character-indexed verbatim grounding validation, claim dropping, SHA-256 hash dedup, idempotent upsert | **COMPLETED** | `42ddd73` | [Green (Run 35981759883)](https://github.com/swarajladke/Linkedin-Agent/actions/runs/35981759883) |
| **5** | **Goal Compiler** | Free-text objective parser into `target_spec`, numeric metrics, and back-solved sub-goal milestone timelines | **COMPLETED** | `52cd4b6` | [Green (Run 35984792388)](https://github.com/swarajladke/Linkedin-Agent/actions/runs/35984792388) |
| **6** | **CLI (`pilot`)** | Typer + Rich terminal commands: `pilot init`, `pilot ingest`, `pilot goal set`, `pilot show` | **COMPLETED** | `ca80fd7` | [Green (Run 35993869378)](https://github.com/swarajladke/Linkedin-Agent/actions/runs/35993869378) |
| **7** | **Final Tests** | End-to-end integration tests: grounding guarantee validation, timeline back-solving, upsert idempotency | **COMPLETED** | `TBD` | CI Testing |

---

## 3. Detailed Step Breakdown

### Step 1: Scaffold + Docker Compose + Config *(Completed)*
- **Objective:** Establish the foundational infrastructure, build configuration, linting rules, and database containers.
- **Implemented Artifacts:**
  - `pyproject.toml`: Configured for Python `>=3.11`, SQLAlchemy 2.0 sync, Alembic, Psycopg 3, pgvector, Pydantic v2, Typer, Rich, PyPDF, HTTPX, and OpenAI SDK.
  - `docker-compose.yml`: Official `pgvector/pgvector:pg16` image with health checks.
  - `src/pilot/config.py`: Pydantic `BaseSettings` object with `SecretStr` for database credentials and LLM keys.
  - `src/pilot/constants.py`: Single source of truth for embedding dimensionality (`EMBEDDING_DIM = 1536`).
  - `.gitignore` & `.env.example`.

---

### Step 2: World Models + Migration 0001 *(Completed)*
- **Objective:** Model the entire relational world state, agent decision records, and learning loop.
- **Implemented Artifacts:**
  - `src/pilot/db/models.py`: 15 SQLAlchemy 2.0 mapped models:
    1. `users`: Registered job seekers.
    2. `goals`: Top-level active objectives with target specs, constraints, criteria, and sub-goals.
    3. `strategies`: Tactical versions with self-referential `parent_version_id` lineage.
    4. `companies`: Target employers.
    5. `people`: Industry contacts, hiring managers, and recruiters.
    6. `roles`: Specific job openings with status and requirements.
    7. `applications`: Funnel pipeline stages from `discovered` to `offer`.
    8. `conversations`: Communication threads and channels.
    9. `actions`: Pre-action reasoning, action type, target type/id discriminator, predicted outcome, probability in $[0, 1]$, and dynamic `strategy_version` association proxy.
    10. `action_outcomes`: Resolution record with database-computed Brier score.
    11. `evidence_claims`: Verifiable claims with polymorphic `(entity_type, entity_id)`, `source_excerpt`, `content_hash`, confidence in $[0, 1]$, and pgvector `embedding`.
    12. `predictions`: Funnel-level probabilistic forecasts only (no `action_id`).
    13. `calibration`: Strategy-level probability decile calibration records.
    14. `strategy_notes`: Market hypotheses and pivot reflections.
    15. `escalations`: Human intervention records.
  - `migrations/versions/0001_initial_world_model.py`:
    - Enabled extensions: `vector` and `pgcrypto`.
    - Triggers: `update_updated_at_column()` and `compute_action_outcome_brier_score()`.
    - Native Postgres ENUMs: `goal_status`, `application_stage`, `role_status`, `strategy_note_status`.
    - Symmetrical `downgrade()`.
  - `.github/workflows/ci.yml`: Spins up `pgvector:pg16` service container, executes `alembic upgrade head`, validates zero-diff with `alembic check`, and runs pytest suite.

---

### Step 3: Resume Reader + GitHub Source Client *(Completed)*
- **Objective:** Ingest unstructured sources (resumes and GitHub profiles) into verbatim text spans and structured metadata without LLM processing.
- **Implemented Artifacts:**
  - `src/pilot/ingestion/reader.py`:
    - Parses PDF (using `pypdf`), TXT, and Markdown.
    - Emits atomic `SourceSpan` objects with precise locators (`page=1;chars=100-240` or `line=14-16`) and `file://` URLs.
    - Segments at paragraph and bullet boundaries, maintaining verbatim fidelity against `raw_text`.
    - Typed errors: `ImageOnlyPDFError` (density check preventing scanned PDFs without OCR), `EncryptedPDFError`, and `UnsupportedFileFormatError`.
  - `src/pilot/ingestion/github.py`:
    - HTTP client (`httpx`) extracting profile, repositories, language breakdowns, decoded READMEs, and commit counts.
    - Excludes forks and archived repositories by default.
    - Handles rate limits via `X-RateLimit-Remaining` inspection and exponential backoff retries on `5xx`.
    - Every object carries originating `html_url`.
  - `src/pilot/ingestion/cache.py`: Filesystem caching (`.cache/`) preventing redundant network or file parsing calls.
  - `tests/test_reader.py` & `tests/test_github.py`: Mocked tests (10/10 passing in CI).

---

### Step 4: Grounded Extractor + Deduplication *(Completed)*
- **Objective:** Transform raw `ResumeDocument` spans and `GitHubUserData` into validated `EvidenceClaim` records using structured LLM extraction.
- **Implemented Artifacts:**
  - `src/pilot/extraction/schemas.py`: `ExtractedClaim` with minimum excerpt length of 8, categorization (`kind`), and `ExtractionBatch`.
  - `src/pilot/extraction/llm.py`: `StructuredLLMClient` protocol and `OpenAIStructuredClient` with lazy SDK import and bounded JSON/Pydantic validation retries.
  - `src/pilot/extraction/grounding.py`: `GroundingValidator` with character-level index mapping from normalized text to original verbatim substrings and refined locators (`f"{span.locator};excerpt_chars={start}-{end}"`).
  - `src/pilot/extraction/spans.py`: `github_to_spans` flattening profile, repositories, commit counts, and chunked READMEs into uniform `SourceSpan`s.
  - `src/pilot/extraction/extractor.py`: `GroundedExtractor` enforcing confidence floor, strict grounding gate, provenance rewriting, and in-run deduplication.
  - `src/pilot/extraction/repository.py`: `upsert_evidence_claims` performing idempotent PostgreSQL upserts on `(entity_id, content_hash)` conflict.
  - Test suite: `tests/test_grounding.py`, `tests/test_extractor.py`, `tests/test_github_spans.py`, `tests/test_evidence_dedup.py`.

---

### Step 5: Goal Compiler *(Completed)*
- **Objective:** Translate human natural language career objectives and constraints into a machine-executable, time-sequenced world model goal.
- **Implemented Artifacts:**
  - `src/pilot/goals/schemas.py`: `CompiledGoalDraft` and `FunnelAssumptions` models.
  - `src/pilot/goals/errors.py`: `InfeasibleGoalError` for invalid or mathematically unachievable goals.
  - `src/pilot/goals/compiler.py`: `GoalCompiler` implementing pure compilation, numeric sanitization of `success_criteria`, deterministic Python back-solving of funnel volumes (`interviews_needed`, `applications_needed`, `roles_to_source`), and strictly ordered sub-goal timelines within `(now, deadline)`.
  - `src/pilot/goals/repository.py`: `persist_compiled_goal` inserting `Goal`, `Strategy(version=1)`, and baseline `StrategyNote(hypothesis=...)`, with automatic transition of existing active goals to `GoalStatus.PAUSED`.
  - `tests/test_goal_compiler.py`: 6 tests verifying classification shape, numeric sanitization, mathematical volume correctness, deadline ordering, infeasibility errors, and DB persistence.

---

### Step 6: CLI Interface *(Completed)*
- **Objective:** Provide developer and candidate terminal control over the agent state.
- **Implemented Artifacts:**
  - `src/pilot/cli/main.py`: Interactive commands built on Typer + Rich:
    1. `pilot init --email <email> --name <name> [--github <username>]`: Verifies database connectivity, programmatically executes Alembic migrations up to `head`, and idempotently registers or updates the candidate user.
    2. `pilot ingest --resume <path> [--github <username>] [--force-refresh]`: Ingests resume spans, fetches GitHub repositories/READMEs, runs `GroundedExtractor`, persists claims with hash deduplication, displays grounding pass-rates, and prints dropped claim audits.
    3. `pilot goal set "<objective>" --deadline <YYYY-MM-DD> [-c key=value]`: Runs `GoalCompiler` with fast-fail checks, renders categorized target specs and numeric criteria, and visualizes back-solved sub-goal milestone schedules.
    4. `pilot show [--claims] [--goal] [--limit N]`: Inspects active goals, strategy versions, and top verified evidence claims by source with shortened locators.
  - `src/pilot/cli/__init__.py`: Exports `app`.
  - `tests/test_cli.py`: 7 tests covering CLI help trees, idempotent user initialization, grounded extraction tables and deduplication idempotency, image-only PDF error handling, past deadline rejection, and empty database hints.

---

### Step 7: Final Phase 1 Test Suite *(Completed)*
- **Objective:** Verify cross-module guarantees, strict grounding invariants under adversarial attacks, mathematical timeline and funnel monotonicity, and ingestion idempotency.
- **Implemented Artifacts:**
  - `tests/integration/conftest.py`: Shared integration fixtures (`clean_db`, `session`, `seeded_user`, `scripted_llm`), reusable `assert_grounding_invariant(persisted_claims, source_spans)` validation helper, and end-to-end `phase1_pipeline(...)` harness.
  - `tests/integration/test_e2e_phase1.py`: Full lifecycle pipeline test asserting the terminal world model state: valid `evidence_claims` provenance and 64-char hashes, exactly one active `Goal`, one `Strategy(version=1, parent=None)`, one baseline `StrategyNote`, and sub-goal timelines within `(created_at, deadline)`.
  - `tests/integration/test_grounding_invariant.py`: `AdversarialLLM` test suite verifying 9 adversarial attack vectors (paraphrasing, silent number mutation, cross-span splices, fabricated clauses, short tokens, empty/whitespace, and unicode homoglyphs) with an exact 1.0 hallucination drop rate.
  - `tests/integration/test_goal_timeline_invariant.py`: 16-variant mathematical matrix over horizons (14, 45, 90, 180 days), offer targets (1, 3), and funnel priors (optimistic, pessimistic) proving strictly ordered sub-goal deadlines, monotonic funnel volumes, self-consistency of success criteria, infeasible rate-limit rejection, and deterministic compilation.
  - `tests/integration/test_idempotency.py`: Tests proving that repeated ingest executions preserve row count and hash sets, advance `verified_at` while leaving `created_at` immutable, preserve distinct provenance for identical claims across sources, and deduplicate CLI `pilot init` invocations.
  - `.github/workflows/ci.yml`: Split test pipeline into independent "Run Unit Tests" (`pytest -m "not integration"`) and "Run Integration Tests" (`pytest -m "integration"`).

---

## 4. Phase 1 Repository Tree

```text
Linkedin-Agent/
├── .github/
│   └── workflows/
│       └── ci.yml                          # Dual-stage CI (Ruff, Alembic check, Unit tests, Integration tests)
├── alembic/
│   └── env.py
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 0001_initial_world_model.py     # 15 tables, 2 triggers, pgvector, pgcrypto
├── src/
│   └── pilot/
│       ├── cli/
│       │   ├── __init__.py                 # Typer app export
│       │   └── main.py                     # pilot init, ingest, goal set, show
│       ├── config.py                       # Pydantic Settings with SecretStr
│       ├── constants.py                    # EMBEDDING_DIM = 1536
│       ├── db/
│       │   ├── models.py                   # 15 mapped SQLAlchemy 2.0 models
│       │   └── session.py                  # Sync engines and sessionmaker
│       ├── extraction/
│       │   ├── extractor.py                # GroundedExtractor pipeline
│       │   ├── grounding.py                # GroundingValidator (verbatim character mapping)
│       │   ├── llm.py                      # StructuredLLMClient & OpenAIStructuredClient
│       │   ├── repository.py               # Idempotent upsert_evidence_claims
│       │   ├── schemas.py                  # ExtractedClaim, ExtractionBatch
│       │   └── spans.py                    # github_to_spans flattener
│       ├── goals/
│       │   ├── compiler.py                 # GoalCompiler & back-solver
│       │   ├── errors.py                   # InfeasibleGoalError
│       │   ├── repository.py               # persist_compiled_goal (Goal, Strategy, Note)
│       │   └── schemas.py                  # CompiledGoalDraft, FunnelAssumptions
│       ├── ingestion/
│       │   ├── cache.py                    # File-based caching
│       │   ├── github.py                   # Public GitHub API client
│       │   └── reader.py                   # PDF and text SourceSpan reader
│       └── schemas/
│           ├── evidence.py                 # EvidenceClaimCreate & content hashing
│           ├── goal.py                     # TargetSpec, SubGoal, GoalCreate, GoalRead
│           └── strategy.py                 # StrategyCreate, StrategyNoteCreate
├── tests/
│   ├── conftest.py                         # Root db_engine and transactional db_session
│   ├── fixtures/
│   │   ├── image_only_resume.pdf           # Scanned PDF density fixture
│   │   ├── sample_resume.pdf               # Rendered PDF resume
│   │   └── sample_resume.txt               # Plain text resume fixture
│   ├── integration/
│   │   ├── conftest.py                     # clean_db, seeded_user, ScriptedLLM, phase1_pipeline
│   │   ├── test_e2e_phase1.py              # Full lifecycle terminal state
│   │   ├── test_goal_timeline_invariant.py # Mathematical timeline & funnel monotonicity
│   │   ├── test_grounding_invariant.py     # Adversarial grounding invariant (1.0 drop rate)
│   │   └── test_idempotency.py             # 3-run ingest idempotency & refresh verification
│   ├── test_cli.py                         # CLI commands unit tests
│   ├── test_evidence_dedup.py              # Hash uniqueness unit tests
│   ├── test_extractor.py                   # Extractor unit tests
│   ├── test_github.py                      # GitHub client unit tests
│   ├── test_github_spans.py                # GitHub to spans unit tests
│   ├── test_goal_compiler.py               # Goal compiler unit tests
│   ├── test_grounding.py                   # Grounding validator unit tests
│   ├── test_migration.py                   # Migration 0001 schema tests
│   └── test_reader.py                      # Resume reader unit tests
├── docker-compose.yml                      # PostgreSQL 16 + pgvector container
├── pyproject.toml                          # Build config, dependencies, ruff, pytest markers
└── README.md                               # Project overview and CLI guide
```

---

## 5. Phase 1 Exit Criteria & System Guarantees

Phase 1 establishes the rock-solid substrate upon which Phase 2 (Job Intelligence) and Phase 3 (Autonomous Planner) operate. All three core invariants are verified under automated testing:

1. **Strict Grounding Invariant**:
   $$\forall c \in \text{evidence\_claims}, \quad c.\text{source\_excerpt} \text{ exists verbatim in } S_{\text{source}}$$
   Every factual claim persisted to `evidence_claims` is guaranteed to contain a verbatim excerpt from candidate source documents. Hallucinated, spliced, paraphrased, mutated, or homoglyphic excerpts have an exact 1.0 drop rate.
2. **Mathematical Timeline & Funnel Monotonicity Invariant**:
   $$\text{now} < t_{\text{sg1}} < t_{\text{sg2}} < t_{\text{sg3}} < t_{\text{sg4}} < t_{\text{goal}}$$
   $$\text{roles\_to\_source} \ge \text{applications\_needed} \ge \text{responses\_needed} \ge \text{interviews\_needed} \ge \text{min\_offers}$$
   Sub-goal deadlines are strictly increasing in dependency order and bounded within $(t_{\text{now}}, t_{\text{goal}})$. Back-solved funnel volumes are monotonic and mathematically self-consistent with conversion priors. Rate-limiting constraints that make funnels mathematically impossible raise `InfeasibleGoalError` prior to persistence.
3. **Ingest Idempotency & Provenance Refresh Invariant**:
   Re-running ingestion arbitrarily many times yields zero duplicate rows in `evidence_claims`, leaves original `created_at` immutable, and updates `verified_at` to mark active re-verification. Identical claim texts from distinct sources (`resume` vs `github`) preserve distinct provenance URLs and content hashes.
