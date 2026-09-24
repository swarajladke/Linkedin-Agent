# Pilot Phase 1: Roadmap, Architectural Constraints & Progress Tracker

**Project:** Pilot — Goal-Directed Autonomous Career Agent  
**Phase:** Phase 1 (World Model & Ingestion Engine)  
**Status:** **3 of 7 Steps Completed (43%)** | **4 Steps Remaining (57%)**  
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
| :-: | :--- | :--- | :-: | :---: | :-: |
| **1** | **Scaffold + Docker + Config** | Pyproject, Ruff, Pytest, Docker Compose (pgvector 16), Pydantic Settings (`SecretStr`) | **COMPLETED** | `c84ffa1` | Verified |
| **2** | **World Model + Migration 0001** | 15 SQLAlchemy 2.0 models, Alembic sync migration, polymorphic discriminators, Brier trigger, constraints | **COMPLETED** | `da9d9d2` | [Green (Run 35972591152)](https://github.com/swarajladke/Linkedin-Agent/actions/runs/35972591152) |
| **3** | **Resume Reader + GitHub Client** | Verbatim text/PDF span reader with exact locators, rate-limited public GitHub REST client, local disk cache | **COMPLETED** | `a430481` | [Green (Run 35974080710)](https://github.com/swarajladke/Linkedin-Agent/actions/runs/35974080710) |
| **4** | **Grounded Extractor + Dedup** | Structured LLM extraction, provenance validation against spans, claim dropping, SHA-256 hash deduplication | **REMAINING** | *Pending* | *Pending* |
| **5** | **Goal Compiler** | Free-text objective parser into `target_spec`, numeric metrics, and back-solved sub-goal milestone timelines | **REMAINING** | *Pending* | *Pending* |
| **6** | **CLI (`pilot`)** | Typer + Rich terminal commands: `pilot init`, `pilot ingest`, `pilot goal set`, `pilot show` | **REMAINING** | *Pending* | *Pending* |
| **7** | **Final Tests** | End-to-end integration tests: grounding guarantee validation, timeline back-solving, upsert idempotency | **REMAINING** | *Pending* | *Pending* |

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

### Step 4: Grounded Extractor + Deduplication *(Next Up)*
- **Objective:** Transform raw `ResumeDocument` spans and `GitHubUserData` into validated `EvidenceClaim` records using structured LLM extraction.
- **Specifications:**
  - **LLM Wrapper**: Structured output extraction using Pydantic schemas with automatic validation, retry on schema mismatch, and fallback/raise.
  - **Strict Grounding Enforcement**: Every extracted claim must reference an exact verbatim `source_excerpt` found within the ingested `SourceSpan` list. Any claim fabricated or hallucinated without an exact matching source excerpt and valid `source_url` is automatically dropped.
  - **Deduplication & Idempotency**:
    - Generates SHA-256 `content_hash = sha256(claim.strip().lower() + "|" + source_url.strip())`.
    - Upserts into PostgreSQL `evidence_claims` table on `(entity_id, content_hash)` conflict, updating `confidence` and `verified_at`.

---

### Step 5: Goal Compiler *(Remaining)*
- **Objective:** Translate human natural language career objectives and constraints into a machine-executable, time-sequenced world model goal.
- **Specifications:**
  - **Input:** Raw objective string (e.g., *"Land an Applied AI Engineer role, remote or Bangalore, by Dec 1"*) and constraints dict.
  - **Output Schema:**
    - `target_spec`: Breakdown into `must_have`, `nice_to_have`, and `unstated_but_real` (implicit industry table-stakes).
    - `success_criteria`: Quantitative targets (e.g., minimum offers, salary threshold, role seniority level).
    - `sub_goals`: Sequenced milestones back-solved from the deadline (e.g., portfolio readiness by Week 2, target pipeline of 30 roles by Week 4, interview conversions by Week 7).
  - Persists directly into the `goals` table and initializes version 1 in `strategies`.

---

### Step 6: CLI Interface *(Remaining)*
- **Objective:** Provide developer and candidate terminal control over the agent state.
- **Commands:**
  1. `pilot init`: Initialize database connections, execute migrations, and register/verify the candidate user.
  2. `pilot ingest --resume <path> [--github <username>]`: Ingest resume and GitHub profile, execute grounded extraction, and populate `evidence_claims`.
  3. `pilot goal set "<objective>"`: Run the goal compiler and persist the active goal and baseline strategy.
  4. `pilot show`: Render rich terminal dashboards summarizing candidate evidence claims, active goal specs, back-solved milestones, and current strategies.

---

### Step 7: Final Phase 1 Test Suite *(Remaining)*
- **Objective:** Verify end-to-end reliability, grounding guarantees, and timeline logic.
- **Test Scenarios:**
  1. **Grounding Rule Verification**: Provide LLM outputs with synthetic hallucinated claims; assert that extractor drops 100% of claims lacking exact `source_excerpt` spans.
  2. **Goal Compiler Output Shape**: Verify target spec classification and that sub-goal deadlines are mathematically ordered and strictly precede the final goal deadline.
  3. **Idempotency Verification**: Running `pilot ingest` repeatedly on the same resume/GitHub source must not create duplicate claims in `evidence_claims`.
