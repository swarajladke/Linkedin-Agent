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

## Phase 1 Deliverables (In Progress)

- [x] **1. Scaffold + Docker Compose + Config**: Pyproject, Ruff, Pytest, Docker Compose (PostgreSQL 16 + pgvector), Pydantic Settings.
- [x] **2. Models + Migration 0001**: SQLAlchemy 2.0 models and Alembic migration for all core tables.
- [x] **3. Resume Reader + GitHub Client**: Deterministic raw text and document extraction without LLM.
- [x] **4. Grounded Extractor + Deduplication**: Structured LLM extraction with validation, content-hash deduplication, and strict provenance enforcement.
- [ ] **5. Goal Compiler**: Objective & constraints parser into target specs, numeric success criteria, and back-solved sub-goal timelines.
- [ ] **6. CLI**: `pilot init`, `pilot ingest`, `pilot goal set`, `pilot show`.
- [ ] **7. Tests**: Test suite for grounding validation, claim deduplication, and goal compiler timeline resolution.

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
