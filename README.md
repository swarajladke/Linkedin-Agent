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

## Phase 5 Deliverables: Learning & Calibration (Complete)

- [x] **1. Migration 0005 & Schema**: `strategies.policy` JSONB, `strategy_notes.evidence` JSONB, `strategy_notes.cycle_id` FK, `actions.horizon_days`. Symmetrical downgrade, zero-diff `alembic check`.
- [x] **2. Outcome Resolution & Timeout**: Idempotent mapping of application stage transitions and expired prediction horizons into binary outcomes with in-database Brier scores. Critic drops isolated.
- [x] **3. Deterministic Calibration & Bayesian Shrinkage**: Pure `compute_calibration` with decile reliability, Murphy decomposition (reliability, resolution, uncertainty), ECE, and skill score against base rate. Bayesian Beta shrinkage plugging into the pre-execution prediction seam.
- [x] **4. Deterministic Failure Detection**: Statistical triggers (`credible_interval_breached`, `consecutive_starved`, `calibration_drift`, `critic_drop_spike`, `refuted_hypothesis`) with minimum sample size floor.
- [x] **5. Bounded Reflection & Replay-Gated Adoption**: LLM reflection constrained to allow-list parameters and bounded deltas; counterfactual `replay_cycle` validation enforcing material selection differences before version bump; cooldown protection; non-destructive forward rollback.
- [x] **6. Wire into Decision Cycle**: Closed-loop orchestration in exact sequence: `resolve_outcomes → update calibration → evaluate hypotheses → observe → diagnose → generate → score → select → predict (calibrated) → execute (critic gate) → detect → (reflect → evaluate_and_adopt) → commit`.
- [x] **7. CLI & ASCII Diagram**: `pilot calibration [--strategy vN]`, `pilot strategy history`, `pilot strategy diff <vA> <vB>`, `pilot strategy rollback <vN>`, and `pilot learn --dry-run`.
- [x] **8. System Invariants**: Mean Brier score over the last third of cycles strictly drops compared to the first third (empirically honest probabilities); no pivot without evidence.

---

## System Guarantees Across All Phases

| Guarantee | Phase Introduced | Scope & Invariant Definition | Enforcement Mechanism |
| :--- | :--- | :--- | :--- |
| **Grounding** | Phase 1 | Every claim asserted about a candidate links to verified evidence: $\forall c \in \text{claims}, \text{excerpt}(c) \subseteq \text{doc}(c.\text{source})$. | Character-indexed `GroundingValidator` & extraction gate |
| **Timeline** | Phase 1 | Sub-goal milestone deadlines are strictly ordered within goal horizon: $\text{created\_at} < d_1 < d_2 < \dots < d_K \leq \text{deadline}$. | Deterministic `GoalCompiler` back-solving |
| **Idempotency** | Phases 1–5 | Repeated execution of sourcing, ingestion, outcome resolution, and calibration produces zero duplicate rows: $f(f(x)) = f(x)$. | Unique constraints & PostgreSQL UPSERT ON CONFLICT |
| **Prediction-Before-Action** | Phase 3 | Every action persists explicit reasoning and calibrated probability before execution: $a.\text{created\_at} \leq a.\text{executed\_at} \land a.p \in [0, 1]$. | Pre-execution `record_prediction()` flush seam |
| **Nothing-Unchecked-Ships** | Phase 4 | No generated package reaches human review or dispatch without passing verification: $\forall e \in \text{escalations}, \exists r \in \text{reviews} \text{ with verdict}=\text{'pass'}$. | Mandatory Critic gate & drop cap (2 attempts) |
| **No-Pivot-Without-Evidence** | Phase 5 | Every strategy version after v1 must have a parent, a triggering signal with numeric evidence, and a non-empty replay diff: $\forall s \text{ with } v > 1, s.\text{parent\_id} \neq \text{NULL} \land n.\text{evidence} \neq \emptyset \land \text{diff} > 0$. | Deterministic failure detection, bounded reflection & replay diff gating |
| **History-Is-Immutable** | Phases 1–5 | Past decisions and versions are never rewritten. Rollbacks create new versions forward; predictions are never altered after resolution. | Append-only models, immutable versions, rollback via forward creation |

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

### 12. Calibration, Brier Decomposition & ASCII Reliability Diagram
Inspects empirical calibration, Brier score decomposition (Reliability, Resolution, Uncertainty), skill vs. base rate, decile reliability, and ASCII diagram:
```bash
pilot calibration
pilot calibration --strategy v1
```

### 13. Strategy Lineage, Diffs & Non-Destructive Rollback
Inspects the auditable version lineage, parameter diffs, and performs non-destructive rollbacks:
```bash
pilot strategy history
pilot strategy diff v1 v2
pilot strategy rollback v1
```

### 14. Autonomous Learning Loop Simulation
Simulates the learning half of a decision cycle on its own (outcome resolution, calibration, hypothesis evaluation, failure detection, and reflection proposal):
```bash
pilot learn --dry-run
```


