"""Core data models for the content generation pipeline."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class WorkflowStage(str, Enum):
    """Stages in the content generation workflow."""

    PLANNING = "planning"
    RESEARCH = "research"
    VALIDATION = "validation"
    STRUCTURING = "structuring"
    SCRIPT_GENERATION = "script_generation"
    COMPLETED = "completed"
    ERROR = "error"


# Removed AudienceType enum - now using free-form string for audience
# This allows users to specify any custom audience (e.g., "healthcare professionals", 
# "marketing managers", "high school teachers", etc.)


class ContentTone(str, Enum):
    """Content tone variations."""

    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    ENGAGING = "engaging"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"


class SearchResult(BaseModel):
    """Model for web search results."""

    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Extracted text content")
    summary: str = Field(..., description="AI-generated summary")
    embedding: List[float] = Field(..., description="Vector embedding for semantic search")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Source credibility score")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidatedContent(BaseModel):
    """Model for validated and fact-checked content."""

    original_claim: str = Field(..., description="Original claim or statement")
    validation_status: str = Field(..., description="Validation result")
    supporting_sources: List[str] = Field(..., description="URLs supporting the claim")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in validation")
    notes: Optional[str] = Field(None, description="Additional validation notes")


class ContentSection(BaseModel):
    """Model for content structure sections."""

    title: str = Field(..., description="Section title")
    description: str = Field(..., description="Section description")
    estimated_duration_seconds: int = Field(..., description="Estimated speaking time")
    key_points: List[str] = Field(..., description="Main points to cover")
    transition_notes: Optional[str] = Field(None, description="Transition guidance")


class ContentPlan(BaseModel):
    """Model for complete content structure plan."""

    subject: str = Field(..., description="Main topic")
    scope: str = Field(..., description="Content scope")
    target_audience: str = Field(..., description="Target audience (free-form description)")
    target_length_minutes: int = Field(..., description="Target script length")
    sections: List[ContentSection] = Field(..., description="Planned content sections")
    overall_tone: ContentTone = Field(..., description="Overall content tone")
    key_messages: List[str] = Field(..., description="Key messages to convey")
    estimated_total_duration: int = Field(..., description="Total estimated duration")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ScriptSection(BaseModel):
    """Model for individual script sections."""

    section_title: str = Field(..., description="Section title")
    content: str = Field(..., description="Script content for this section")
    word_count: int = Field(..., description="Word count for this section")
    estimated_duration_seconds: int = Field(..., description="Estimated speaking time")


class GeneratedScript(BaseModel):
    """Model for final generated script."""

    title: str = Field(..., description="Script title")
    subject: str = Field(..., description="Main topic")
    target_audience: str = Field(..., description="Target audience (free-form description)")
    target_length_minutes: int = Field(..., description="Target length")
    full_content: str = Field(..., description="Complete script text")
    sections: List[ScriptSection] = Field(..., description="Script sections")
    total_word_count: int = Field(..., description="Total word count")
    estimated_read_time_seconds: int = Field(..., description="Estimated speaking time")
    citations: List[str] = Field(default_factory=list, description="APA-style citations for sources")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowMetadata(BaseModel):
    """Model for workflow execution metadata."""

    session_id: str = Field(..., description="Unique session identifier")
    workflow_stage: WorkflowStage = Field(..., description="Current workflow stage")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)
    search_results_count: int = Field(default=0)
    validated_content_count: int = Field(default=0)
    script_generated: bool = Field(default=False)


class WorkflowState(TypedDict):
    """Complete workflow state for LangGraph."""

    session_id: str
    subject: str
    scope: str
    target_audience: str
    target_length: int
    current_stage: WorkflowStage
    search_results: List[SearchResult]
    validated_content: List[ValidatedContent]
    content_plan: Optional[ContentPlan]
    final_script: Optional[GeneratedScript]
    metadata: WorkflowMetadata
    created_at: datetime
    updated_at: datetime


class UserInput(BaseModel):
    """Model for user input to the pipeline."""

    subject: str = Field(..., description="Main topic or subject")
    scope: str = Field(..., description="Specific scope or angle")
    target_audience: str = Field(..., description="Target audience (e.g., 'healthcare professionals', 'marketing managers')")
    target_length_minutes: int = Field(..., ge=1, le=30, description="Target script length in minutes")
    additional_instructions: Optional[str] = Field(None, description="Additional specific instructions")
