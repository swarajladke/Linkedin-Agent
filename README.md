# Pilot: Goal-Directed Autonomous Career Agent

Pilot is an autonomous career agent designed around a closed-loop decision framework:
**observe → diagnose → choose action → predict outcome → act → record actual outcome → adapt strategy**.

The core objective of Pilot is **auditable decision-making and learning from prediction errors**, rather than generic text generation or resume keyword stuffing.

---

## Hard Architectural Constraints

1. **Python 3.11+ & PostgreSQL with pgvector**: Explicit schema with relational integrity and typed JSONB documents. No graph databases.
2. **Hand-rolled loop**: No LangChain / LangGraph. All decision loops and agent cycles are transparent and debuggable.
3. **No LinkedIn browser automation or logged-in scraping**: Inputs come exclusively from permitted career endpoints, public APIs, user-uploaded resumes, and public GitHub profiles/repositories.
4. **Mandatory falsifiable prediction**: Every single action persists its explicit reasoning, predicted outcome, and predicted probability before execution.
5. **Strict Grounding Guarantee**: Every factual claim made about an entity must link directly to an atomic row in `evidence_claims` with an exact `source_excerpt` and `source_url`. Ungrounded claims are rejected.

---

## Phase 1 Deliverables (Complete)

- [x] **1. Scaffold + Docker Compose + Config**: Pyproject, Ruff, Pytest, Docker Compose (PostgreSQL 16 + pgvector), Pydantic Settings.
- [x] **2. Models + Migration 0001**: SQLAlchemy 2.0 models and Alembic migration for all core tables.
- [x] **3. Resume Reader + GitHub Client**: Deterministic raw text and document extraction without LLM.
- [x] **4. Grounded Extractor + Deduplication**: Structured LLM extraction with validation, content-hash deduplication, and strict provenance enforcement.
- [x] **5. Goal Compiler**: Objective & constraints parser into target specs, numeric success criteria, and back-solved sub-goal timelines.
- [x] **6. CLI**: `pilot init`, `pilot ingest`, `pilot goal set`, `pilot show`.
- [x] **7. Tests**: End-to-end integration and invariant test suite (grounding guarantee, mathematical timelines, idempotency).

---

## Phase 2 Deliverables: Job Intelligence (Complete)

- [x] **1. Migration 0002 & Schema**: Add sourcing columns to `roles` (`source`, `external_id`, `raw_posting`, timestamps), unique `(source, external_id)`; create `role_assessments` table with check constraint `fit_score ∈ [0, 1]`, unique `(role_id, goal_id, assessor_version)`.
- [x] **2. Job Board Adapters**: Public, unauthenticated `JobSource` adapters for Greenhouse, Ashby, and Lever with deterministic HTML stripping, disk caching, and rate-limit backoff.
- [x] **3. Sourcing Repository**: Idempotent `upsert_roles` with company deduplication, `first_seen_at` immutability, `last_seen_at` progression, and absence-based `RoleStatus.CLOSED` transition.
- [x] **4. Grounded Role Assessor**: Auditable fit scoring pure function from structured sub-signals, strict candidate claim ID validation, blocking gap penalty, and deterministic recommended action derivation.
- [x] **5. Intelligence Repository**: `upsert_assessments` with versioning semantics (in-place update under same version, new row on version bump for replay).
- [x] **6. CLI Extensions**: `pilot source`, `pilot assess`, `pilot explain <role_id>`, and top opportunities table in `pilot show`.
- [x] **7. Invariant Tests**: Pure function score, fabricated claim rejection, blocking gap constraints, 3-run sourcing idempotency, closed role transition, versioning replay, and aggregate claim grounding invariant.

---

## Phase 3 Deliverables: Planner (Complete)

- [x] **1. Migration 0003 & Schema**: Add `cycles` table with unique constraint `(goal_id, cycle_number)` and foreign key `actions.cycle_id`.
- [x] **2. Observe & Diagnose**: Pure observation reader and deterministic bottom-up funnel pace classification with structured LLM root-cause hypotheses.
- [x] **3. Action Selection & Pre-execution Predictions**: Action generation, deterministic expected value scoring, constraint-bounded selection, and pre-execution prediction persistence.
- [x] **4. Orchestration & Replay Harness**: Single-transaction cycle orchestration with rollback on error, and counterfactual replay harness comparing alternate selection policies.
- [x] **5. CLI & Invariant Tests**: `pilot cycle run/list/show`, `pilot outcome`, `pilot replay`, atomic execution, and orphan-action rollback guarantees.

---

## Phase 4 Deliverables: Critic (Complete)

- [x] **1. Migration 0004 & Models**: `writing_samples` (unique `user_id, content_hash`) and `critic_reviews` (unique `action_id, attempt`). Zero-diff Alembic migrations with symmetrical downgrade.
- [x] **2. Statistical Voice Profiling**: Extraction of quantitative stylometrics (sentence length, variance, contraction rate, passive voice rate, pronoun frequency, type-token ratio, and candidate-mined banned clichés) using `ResumeReader`.
- [x] **3. Mandatory Verification Gate**: Three-check pipeline (`grounding_check` with framing exemption, deterministic `voice_check`, and regex/parser `factual_check`).
- [x] **4. Regenerate or Drop**: Automatic redrafting feedback loop with hard cap of 2 attempts before dropping action without escalation.
- [x] **5. CLI Extensions**: `pilot voice add <path>`, `pilot voice show`, `pilot review <action_id>`, and `pilot cycle run` pass / regenerate / drop reporting.
- [x] **6. System Invariant**: "Nothing Unchecked Ships" guarantee — every human escalation payload mathematically traces to a passing `critic_reviews` row.

---

## Quickstart

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Virtual environment manager (`uv`, `poetry`, or standard `venv`)

### 1. Environment Setup
Copy the example environment file and populate necessary keys:
```bash
cp .env.example .env
```

### 2. Launch Database Service
Start the PostgreSQL 16 container with `pgvector`:
```bash
docker compose up -d
```

### 3. Install Package in Editable Mode
```bash
pip install -e ".[dev]"
```

---

## CLI Usage

### 1. Initialize Database & Candidate
Connects to PostgreSQL, runs Alembic migrations up to `head`, and upserts candidate user:
```bash
pilot init --email candidate@example.com --name "Candidate Name" --github username
```

### 2. Ingest Evidence (Resume & GitHub)
Parses resume, fetches public GitHub code/repositories, and runs grounded structured claim extraction:
```bash
pilot ingest --resume path/to/resume.pdf --github username
```

### 3. Ingest Candidate Writing Samples & View Voice Profile
Ingests articles, essays, or past cover letters to build candidate statistical voice baseline:
```bash
pilot voice add path/to/sample.md
pilot voice show
```

### 4. Set Career Goal & Back-Solve Funnel
Decomposes objective into target specs, numeric criteria, and back-solved phased milestones:
```bash
pilot goal set "Land an Applied AI Engineer role, remote or Bangalore" --deadline 2026-12-01 -c max_applications_per_day=5
```

### 5. Inspect World Model & Progress
Renders active goal specifications, back-solved milestones, verified evidence claims, and top opportunities:
```bash
pilot show
pilot show --goal
pilot show --claims --limit 15
```

### 6. Source Job Postings (Permitted Job Boards)
Fetches job postings from configured boards (`config/boards.yaml`), parses HTML content, and idempotently upserts roles:
```bash
pilot source --all
pilot source --board canonical --board ramp
```

### 7. Assess Open Roles Against Career Goal
Evaluates unassessed open roles against candidate claims using grounded scoring and blocking gap detection:
```bash
pilot assess --limit 20 --min-fit 0.5
```

### 8. Explain Role Assessment & Grounded Evidence Citations
Inspects an assessment with full provenance, displaying verbatim text excerpts and locators for every supporting evidence claim:
```bash
pilot explain <role_id>
```

### 9. Run Decision Cycle Through Critic Verification Gate
Executes the closed decision loop, evaluating draft packages through grounding, voice, and factual checks:
```bash
pilot cycle run
pilot cycle run --dry-run
pilot cycle list
pilot cycle show 1
```

### 10. Inspect Critic Verification Attempts & Reviews
Displays all evaluation attempts, check passes/failures, and offending text for an executed or dropped action:
```bash
pilot review <action_id>
```

### 11. Record Outcomes & Counterfactual Replay
Records actual real-world response/interview outcomes with automatic Brier score computation, and replays cycles under counterfactual policies:
```bash
pilot outcome <action_id> --success
pilot replay 1 --min-fit 0.85 --max-actions 3
```

