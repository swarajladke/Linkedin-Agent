"""SQLAlchemy 2.0 mapped models for Pilot's world model and agent tracking."""

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.associationproxy import AssociationProxy, association_proxy
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from pilot.constants import EMBEDDING_DIM
from pilot.db.session import Base


# ============================================================================
# Enums (Python 3.11+ StrEnum)
# ============================================================================
class GoalStatus(enum.StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class ApplicationStage(enum.StrEnum):
    DISCOVERED = "discovered"
    PREPARING = "preparing"
    APPLIED = "applied"
    SCREENING = "screening"
    TECHNICAL = "technical"
    FINAL = "final"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class RoleStatus(enum.StrEnum):
    OPEN = "open"
    APPLIED = "applied"
    CLOSED = "closed"


class StrategyNoteStatus(enum.StrEnum):
    ACTIVE = "active"
    REFUTED = "refuted"
    GRADUATED = "graduated"


# ============================================================================
# Core World Models
# ============================================================================
class User(Base):
    """Registered job seeker and candidate."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    github_username: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    goals: Mapped[list["Goal"]] = relationship(
        "Goal", back_populates="user", cascade="all, delete-orphan"
    )


class Goal(Base):
    """Top-level single goal being actively pursued by the agent."""

    __tablename__ = "goals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    objective_text: Mapped[str] = mapped_column(Text, nullable=False)
    constraints_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    target_spec: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    success_criteria: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    sub_goals: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[GoalStatus] = mapped_column(
        Enum(
            GoalStatus,
            name="goal_status",
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        server_default=GoalStatus.ACTIVE.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="goals")
    strategies: Mapped[list["Strategy"]] = relationship(
        "Strategy", back_populates="goal", cascade="all, delete-orphan"
    )
    applications: Mapped[list["Application"]] = relationship(
        "Application", back_populates="goal", cascade="all, delete-orphan"
    )
    actions: Mapped[list["Action"]] = relationship(
        "Action", back_populates="goal", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="goal", cascade="all, delete-orphan"
    )
    strategy_notes: Mapped[list["StrategyNote"]] = relationship(
        "StrategyNote", back_populates="goal", cascade="all, delete-orphan"
    )
    escalations: Mapped[list["Escalation"]] = relationship(
        "Escalation", back_populates="goal", cascade="all, delete-orphan"
    )


class Strategy(Base):
    """Versioned tactical strategy for pursuing a goal, maintaining adaptation lineage."""

    __tablename__ = "strategies"
    __table_args__ = (UniqueConstraint("goal_id", "version", name="uq_strategies_goal_id_version"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    goal: Mapped["Goal"] = relationship("Goal", back_populates="strategies")
    parent_strategy: Mapped[Optional["Strategy"]] = relationship(
        "Strategy",
        remote_side=[id],
        foreign_keys=[parent_version_id],
    )
    actions: Mapped[list["Action"]] = relationship("Action", back_populates="strategy")
    calibrations: Mapped[list["Calibration"]] = relationship(
        "Calibration", back_populates="strategy"
    )
    notes: Mapped[list["StrategyNote"]] = relationship("StrategyNote", back_populates="strategy")


class Company(Base):
    """Target employer organization."""

    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    careers_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    headquarters: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    roles: Mapped[list["Role"]] = relationship(
        "Role", back_populates="company", cascade="all, delete-orphan"
    )
    people: Mapped[list["Person"]] = relationship("Person", back_populates="company")


class Person(Base):
    """Industry contacts, recruiters, hiring managers, and engineers."""

    __tablename__ = "people"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="SET NULL"),
        nullable=True,
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    public_profile_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    connection_degree: Mapped[str | None] = mapped_column(String(20), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    company: Mapped[Optional["Company"]] = relationship("Company", back_populates="people")
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="person"
    )


class Role(Base):
    """Specific job openings targeted by the agent."""

    __tablename__ = "roles"
    __table_args__ = (Index("ix_roles_company_id_status", "company_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    location_type: Mapped[str] = mapped_column(String(50), nullable=False)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    posting_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    requirements_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[RoleStatus] = mapped_column(
        Enum(
            RoleStatus,
            name="role_status",
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        server_default=RoleStatus.OPEN.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    company: Mapped["Company"] = relationship("Company", back_populates="roles")
    applications: Mapped[list["Application"]] = relationship(
        "Application", back_populates="role", cascade="all, delete-orphan"
    )


class Application(Base):
    """State tracking of an active candidacy for a specific role."""

    __tablename__ = "applications"
    __table_args__ = (Index("ix_applications_goal_id_stage", "goal_id", "stage"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage: Mapped[ApplicationStage] = mapped_column(
        Enum(
            ApplicationStage,
            name="application_stage",
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        server_default=ApplicationStage.DISCOVERED.value,
    )
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    portal_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    goal: Mapped["Goal"] = relationship("Goal", back_populates="applications")
    role: Mapped["Role"] = relationship("Role", back_populates="applications")
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="application"
    )


class Conversation(Base):
    """Communication touchpoints with individuals regarding an application."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="SET NULL"),
        nullable=True,
    )
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("people.id", ondelete="SET NULL"),
        nullable=True,
    )
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    thread_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    application: Mapped[Optional["Application"]] = relationship(
        "Application", back_populates="conversations"
    )
    person: Mapped[Optional["Person"]] = relationship("Person", back_populates="conversations")


# ============================================================================
# Actions, Predictions & Learning Loop
# ============================================================================
class Action(Base):
    """Every autonomous action executed by the agent with mandatory falsifiable predictions."""

    __tablename__ = "actions"
    __table_args__ = (
        CheckConstraint(
            "predicted_probability >= 0.0 AND predicted_probability <= 1.0",
            name="ck_actions_predicted_probability_bounds",
        ),
        Index("ix_actions_goal_id_created_at", "goal_id", "created_at"),
        Index("ix_actions_strategy_id", "strategy_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
    )
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_outcome: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        server_default=text("0.0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actual_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    goal: Mapped["Goal"] = relationship("Goal", back_populates="actions")
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="actions")
    outcome: Mapped[Optional["ActionOutcome"]] = relationship(
        "ActionOutcome",
        back_populates="action",
        uselist=False,
        cascade="all, delete-orphan",
    )
    escalations: Mapped[list["Escalation"]] = relationship("Escalation", back_populates="action")

    # Read-only association proxy to human-readable strategy version (single source of truth)
    strategy_version: AssociationProxy[int | None] = association_proxy("strategy", "version")


class ActionOutcome(Base):
    """Resolution record for an executed action; brier_score is computed automatically via DB trigger."""

    __tablename__ = "action_outcomes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    action_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("actions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    binary_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    actual_outcome_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    diagnosis: Mapped[str | None] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    action: Mapped["Action"] = relationship("Action", back_populates="outcome")


class EvidenceClaim(Base):
    """Verifiable atomic factual claim about an entity with strict provenance and embedding."""

    __tablename__ = "evidence_claims"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_evidence_claims_confidence_bounds",
        ),
        UniqueConstraint("entity_id", "content_hash", name="uq_evidence_claims_entity_hash"),
        Index("ix_evidence_claims_entity_type_entity_id", "entity_type", "entity_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    verified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class Prediction(Base):
    """Funnel-level probabilistic forecasts only (e.g. milestones or conversion probabilities).

    NOTE: Per-action predictions belong in `actions.predicted_*`, NOT here.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "predicted_probability >= 0.0 AND predicted_probability <= 1.0",
            name="ck_predictions_predicted_probability_bounds",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
    )
    claim_statement: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    resolution_deadline: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    outcome_truth: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    goal: Mapped["Goal"] = relationship("Goal", back_populates="predictions")


class Calibration(Base):
    """Rolling or periodic probability calibration curves for a strategy version."""

    __tablename__ = "calibration"
    __table_args__ = (Index("ix_calibration_strategy_id", "strategy_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_name: Mapped[str] = mapped_column(String(50), nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    calibration_buckets: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="calibrations")


class StrategyNote(Base):
    """Hypotheses, reflections, and strategic pivots."""

    __tablename__ = "strategy_notes"
    __table_args__ = (
        Index("ix_strategy_notes_goal_id", "goal_id"),
        Index("ix_strategy_notes_strategy_id", "strategy_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
    )
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    reflection: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[StrategyNoteStatus] = mapped_column(
        Enum(
            StrategyNoteStatus,
            name="strategy_note_status",
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        server_default=StrategyNoteStatus.ACTIVE.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    goal: Mapped["Goal"] = relationship("Goal", back_populates="strategy_notes")
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="notes")


class Escalation(Base):
    """Situations requiring human intervention, confirmation, or interview scheduling."""

    __tablename__ = "escalations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
    )
    action_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("actions.id", ondelete="SET NULL"),
        nullable=True,
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    goal: Mapped["Goal"] = relationship("Goal", back_populates="escalations")
    action: Mapped[Optional["Action"]] = relationship("Action", back_populates="escalations")
