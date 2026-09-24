"""Pytest fixtures, scripted LLM fakes, and pipeline runners for Phase 1 integration tests."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from pilot.db.models import EvidenceClaim, Goal, Strategy, StrategyNote, User
from pilot.db.session import get_db_session
from pilot.extraction.extractor import ExtractionResult, GroundedExtractor
from pilot.extraction.llm import StructuredLLMClient
from pilot.extraction.schemas import ExtractedClaim, ExtractionBatch
from pilot.extraction.spans import github_to_spans
from pilot.goals.compiler import GoalCompiler
from pilot.goals.repository import persist_compiled_goal
from pilot.goals.schemas import CompiledGoalDraft, FunnelAssumptions
from pilot.ingestion.github import GitHubUserData
from pilot.ingestion.reader import ResumeReader, SourceSpan
from pilot.schemas.evidence import EvidenceClaimCreate
from pilot.schemas.goal import GoalCreate, TargetSpec

T = TypeVar("T", bound=BaseModel)

SAMPLE_RESUME_PATH = Path(__file__).parents[1] / "fixtures" / "sample_resume.txt"


class ScriptedLLM(StructuredLLMClient):
    """Deterministic scripted LLM fake supporting both extraction batches and goal compiler drafts."""

    def __init__(
        self,
        extraction_batches: list[ExtractionBatch | Exception] | None = None,
        goal_drafts: list[CompiledGoalDraft | Exception] | None = None,
        default_extraction: ExtractionBatch | None = None,
        default_goal_draft: CompiledGoalDraft | None = None,
    ) -> None:
        self.extraction_batches = list(extraction_batches or [])
        self.goal_drafts = list(goal_drafts or [])
        self.default_extraction = default_extraction or ExtractionBatch(
            claims=[
                ExtractedClaim(
                    claim="Deployed low-latency inference pipelines using vLLM and TensorRT-LLM",
                    kind="project",
                    source_excerpt="vLLM and TensorRT-LLM",
                    confidence=0.95,
                ),
                ExtractedClaim(
                    claim="Reduced inference latency by 45%",
                    kind="achievement",
                    source_excerpt="reducing latency by 45%",
                    confidence=0.90,
                ),
                ExtractedClaim(
                    claim="Spearheaded RAG architecture using pgvector and hybrid BM25 search",
                    kind="skill",
                    source_excerpt="pgvector and hybrid BM25 search",
                    confidence=0.92,
                ),
            ]
        )
        self.default_goal_draft = default_goal_draft or CompiledGoalDraft(
            target_spec=TargetSpec(
                must_have=["Senior Applied AI Engineer", "Remote or Bangalore", "Min INR 35L"],
                nice_to_have=["Series A/B startup", "Python & PyTorch stack"],
                unstated_but_real=[
                    "Can pass live DSA coding screen",
                    "Verifiable LLM evaluation project",
                ],
            ),
            success_criteria={
                "min_offers": 1,
                "min_salary": 3500000,
            },
            funnel_assumptions=FunnelAssumptions(
                application_to_response=0.10,
                response_to_interview=0.25,
                interview_to_offer=0.20,
                sourcing_to_application=0.50,
            ),
        )
        self.recorded_calls: list[tuple[str, str, type]] = []

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        self.recorded_calls.append((system_prompt, user_prompt, schema))
        if schema is ExtractionBatch or issubclass(schema, ExtractionBatch):
            if self.extraction_batches:
                resp = self.extraction_batches.pop(0)
                if isinstance(resp, Exception):
                    raise resp
                return resp  # type: ignore
            return self.default_extraction  # type: ignore

        if schema is CompiledGoalDraft or issubclass(schema, CompiledGoalDraft):
            if self.goal_drafts:
                resp = self.goal_drafts.pop(0)
                if isinstance(resp, Exception):
                    raise resp
                return resp  # type: ignore
            return self.default_goal_draft  # type: ignore

        raise ValueError(f"ScriptedLLM received unexpected schema: {schema}")


@dataclass
class Phase1PipelineResult:
    """Consolidated outcome of running the full Phase 1 ingestion and goal lifecycle."""

    user: User
    resume_spans: list[SourceSpan]
    github_spans: list[SourceSpan]
    extraction_result: ExtractionResult
    persisted_claim_count: int
    persisted_claims: list[EvidenceClaim]
    compiled_goal: GoalCreate | None = None
    persisted_goal: Goal | None = None
    strategy: Strategy | None = None
    strategy_note: StrategyNote | None = None


def assert_grounding_invariant(
    persisted_claims: list[EvidenceClaim],
    source_spans: list[SourceSpan],
) -> None:
    """Assert the aggregate invariant: every persisted source_excerpt appears verbatim in source spans."""
    assert len(persisted_claims) > 0, "No claims provided to check grounding invariant"
    for claim in persisted_claims:
        assert claim.source_excerpt, f"Claim {claim.id} has empty source_excerpt"
        matched = any(claim.source_excerpt in span.text for span in source_spans)
        assert matched, (
            f"Grounding invariant VIOLATED: source_excerpt '{claim.source_excerpt}' "
            f"for claim '{claim.claim}' does not appear verbatim in any source span!"
        )


def phase1_pipeline(
    session: Session,
    *,
    user: User | None = None,
    resume_path: Path | str | None = None,
    github_data: GitHubUserData | None = None,
    llm: StructuredLLMClient | None = None,
    objective_text: str = "Land Senior Applied AI Engineer role",
    constraints: dict[str, Any] | None = None,
    deadline: datetime | None = None,
    now: datetime | None = None,
    compile_goal: bool = True,
) -> Phase1PipelineResult:
    """Execute the shared Phase 1 pipeline from ingestion to extraction and goal compilation."""
    from pilot.extraction.repository import upsert_evidence_claims

    # 1. Resolve or seed user
    if user is None:
        user = session.scalars(select(User)).first()
        if user is None:
            user = User(
                email="alex.candidate@example.com",
                full_name="Alex Candidate",
                github_username="alexcandidate",
            )
            session.add(user)
            session.commit()
            session.refresh(user)

    # 2. Ingest Resume
    target_resume = Path(resume_path) if resume_path else SAMPLE_RESUME_PATH
    reader = ResumeReader()
    doc = reader.read(target_resume)
    resume_spans = doc.spans

    # 3. Ingest GitHub (optional)
    github_spans: list[SourceSpan] = []
    if github_data is not None:
        github_spans = github_to_spans(github_data)

    # 4. Extract Grounded Claims
    client = llm or ScriptedLLM()
    extractor = GroundedExtractor(llm=client)
    all_claims: list[EvidenceClaimCreate] = []
    all_dropped = []
    proposed_count = 0

    if resume_spans:
        res_resume = extractor.extract(
            spans=resume_spans,
            entity_type="user",
            entity_id=user.id,
            source="resume",
        )
        all_claims.extend(res_resume.claims)
        all_dropped.extend(res_resume.dropped)
        proposed_count += res_resume.proposed_count

    if github_spans:
        res_gh = extractor.extract(
            spans=github_spans,
            entity_type="user",
            entity_id=user.id,
            source="github",
        )
        all_claims.extend(res_gh.claims)
        all_dropped.extend(res_gh.dropped)
        proposed_count += res_gh.proposed_count

    extraction_result = ExtractionResult(
        claims=all_claims,
        dropped=all_dropped,
        proposed_count=proposed_count,
    )

    # 5. Persist Claims
    upserted_count = upsert_evidence_claims(session, all_claims)
    session.commit()
    persisted_claims = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == user.id)
    ).all()

    # 6. Compile & Persist Goal (if requested)
    compiled_goal: GoalCreate | None = None
    persisted_goal: Goal | None = None
    strategy: Strategy | None = None
    strategy_note: StrategyNote | None = None

    if compile_goal:
        ref_now = now or datetime.now(UTC)
        goal_deadline = deadline or (ref_now + timedelta(days=60))
        compiler = GoalCompiler(llm=client, now=ref_now)
        compiled_goal = compiler.compile(
            user_id=user.id,
            objective_text=objective_text,
            constraints=constraints or {},
            deadline=goal_deadline,
        )
        persisted_goal, strategy = persist_compiled_goal(session, compiled_goal)
        strategy_note = session.scalars(
            select(StrategyNote).where(StrategyNote.strategy_id == strategy.id)
        ).first()
        session.commit()

    return Phase1PipelineResult(
        user=user,
        resume_spans=resume_spans,
        github_spans=github_spans,
        extraction_result=extraction_result,
        persisted_claim_count=upserted_count,
        persisted_claims=list(persisted_claims),
        compiled_goal=compiled_goal,
        persisted_goal=persisted_goal,
        strategy=strategy,
        strategy_note=strategy_note,
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_schema(db_engine):
    """Ensure database schema is upgraded to head before running any integration tests."""
    from alembic import command
    from alembic.config import Config

    from pilot.config import get_settings

    settings = get_settings()
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", settings.database_url.get_secret_value())
    command.upgrade(cfg, "head")


@pytest.fixture(autouse=True)
def clean_db(ensure_schema):
    """Truncate Phase 1 tables before each integration test to guarantee order independence."""
    with get_db_session() as session:
        session.execute(
            text(
                "TRUNCATE TABLE users, evidence_claims, goals, strategies, strategy_notes CASCADE;"
            )
        )
        session.commit()
    yield


@pytest.fixture
def session():
    """Provide an active database session for integration tests."""
    with get_db_session() as sess:
        yield sess


@pytest.fixture
def seeded_user(session: Session) -> User:
    """Create and persist a standard test candidate user."""
    user = User(
        email="alex.candidate@example.com",
        full_name="Alex Candidate",
        github_username="alexcandidate",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    """Provide a deterministic scripted LLM fake."""
    return ScriptedLLM()
