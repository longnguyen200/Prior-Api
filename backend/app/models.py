import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=40)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=40)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(UserBase):
    email: EmailStr | None = Field(default=None, max_length=255)  # type: ignore
    password: str | None = Field(default=None, min_length=8, max_length=40)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=40)
    new_password: str = Field(min_length=8, max_length=40)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    items: list["Item"] = Relationship(back_populates="owner", cascade_delete=True)
    pdf_extracted_data: list["PDFExtractedData"] = Relationship(
        back_populates="owner", cascade_delete=True
    )
    policy_evaluations: list["PolicyEvaluation"] = Relationship(
        back_populates="owner", cascade_delete=True
    )


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# Shared properties
class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Properties to receive on item creation
class ItemCreate(ItemBase):
    pass


# Properties to receive on item update
class ItemUpdate(ItemBase):
    title: str | None = Field(default=None, min_length=1, max_length=255)  # type: ignore


# Database model, database table inferred from class name
class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")


# Properties to return via API, id is always required
class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=40)


# PDF Extraction models
class PDFExtractionRequest(SQLModel):
    config_override: dict[str, Any] | None = Field(
        default=None, description="Override default extraction config"
    )


class PDFExtractionResponse(SQLModel):
    success: bool
    extracted_data: dict[str, Any] | None = None
    error: str | None = None
    filename: str
    id: str | None = None  # ID of saved record when save_to_db=True


class PDFBatchExtractionResponse(SQLModel):
    success: bool
    extracted_data: dict[str, Any] | None = None
    error: str | None = None
    filenames: list[str]  # List of all processed filenames
    total_files: int
    id: str | None = None  # ID of saved record when save_to_db=True


# Database model for storing extracted PDF data
class PDFExtractedData(SQLModel, table=True):
    __tablename__ = "pdf_extracted_data"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    filename: str = Field(max_length=255)  # Can store JSON array for multiple files
    file_hash: str = Field(max_length=64, index=True)  # SHA256 hash of combined file content
    extracted_data: str  # JSON string
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship()


# Policy Evaluation Models
class ApprovalReason(SQLModel):
    """Individual reason for approval or denial"""

    criterion: str = Field(description="The specific policy criterion being evaluated")
    status: str = Field(description="met, not_met, or insufficient_data")
    summary: str = Field(description="Short summary explaining the evaluation")
    details: str | None = Field(
        default=None, description="Additional details or evidence"
    )


class PolicyEvaluationResult(SQLModel):
    """Structured policy evaluation result"""

    final_determination: str = Field(description="approved, denied, or needs_more_info")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence level (0-1)"
    )
    approval_reasons: list[ApprovalReason] = Field(
        default_factory=list, description="Reasons supporting approval"
    )
    denial_reasons: list[ApprovalReason] = Field(
        default_factory=list, description="Reasons supporting denial"
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="List of missing data points needed for evaluation",
    )
    policy_references: list[str] = Field(
        default_factory=list, description="References to specific policy sections"
    )
    summary: str = Field(description="Overall summary of the evaluation")
    questions_for_provider: list[str] = Field(
        default_factory=list,
        description="Questions to ask provider if information is missing",
    )


class PolicyEvaluationRequest(SQLModel):
    """Request for policy evaluation"""

    extracted_data_id: str = Field(description="ID of the extracted patient data")
    policy_documents: list[str] | None = Field(
        default=None,
        description="Optional: specific policy document names to use for evaluation",
    )


class PolicyEvaluationResponse(SQLModel):
    """Response from policy evaluation"""

    success: bool
    evaluation_result: PolicyEvaluationResult | None = None
    error: str | None = None
    evaluation_id: str | None = None  # ID of saved evaluation record


# Database model for storing policy evaluations
class PolicyEvaluation(SQLModel, table=True):
    __tablename__ = "policy_evaluations"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    patient_data: str = Field(description="JSON string of extracted patient data")
    pa_form_data: str = Field(description="JSON string of PA form data")
    evaluation_result: str = Field(description="JSON string of evaluation result")
    final_determination: str = Field(max_length=20)  # approved, denied, needs_more_info
    confidence_score: float
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship()
