"""Pydantic schemas for external world entities: Companies, Roles, People, Applications, and Conversations."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pilot.db.models import ApplicationStage, RoleStatus


# ============================================================================
# Company
# ============================================================================
class CompanyBase(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    domain: str | None = Field(default=None, max_length=255)
    careers_url: str | None = None
    headquarters: str | None = Field(default=None, max_length=255)


class CompanyCreate(CompanyBase):
    pass


class CompanyRead(CompanyBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Person
# ============================================================================
class PersonBase(BaseModel):
    company_id: UUID | None = None
    full_name: str = Field(min_length=1, max_length=255)
    role_title: str | None = Field(default=None, max_length=255)
    public_profile_url: str | None = None
    connection_degree: str | None = Field(default=None, max_length=20)
    notes: str | None = None


class PersonCreate(PersonBase):
    pass


class PersonRead(PersonBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Role
# ============================================================================
class RoleBase(BaseModel):
    company_id: UUID
    title: str = Field(min_length=1, max_length=255)
    location_type: str = Field(min_length=1, max_length=50)
    location: str | None = Field(default=None, max_length=255)
    posting_url: str | None = None
    requirements_summary: str | None = None
    status: RoleStatus = RoleStatus.OPEN
    source: str = Field(default="manual", max_length=50)
    external_id: str | None = Field(default=None, max_length=255)
    raw_posting: dict[str, object] | None = None


class RoleCreate(RoleBase):
    pass


class RoleRead(RoleBase):
    id: UUID
    first_seen_at: datetime
    last_seen_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Application
# ============================================================================
class ApplicationBase(BaseModel):
    goal_id: UUID
    role_id: UUID
    stage: ApplicationStage = ApplicationStage.DISCOVERED
    applied_at: datetime | None = None
    portal_url: str | None = None
    notes: str | None = None


class ApplicationCreate(ApplicationBase):
    pass


class ApplicationRead(ApplicationBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Conversation
# ============================================================================
class ConversationBase(BaseModel):
    application_id: UUID | None = None
    person_id: UUID | None = None
    channel: str = Field(min_length=1, max_length=50)
    thread_summary: str | None = None
    last_message_at: datetime | None = None


class ConversationCreate(ConversationBase):
    pass


class ConversationRead(ConversationBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
