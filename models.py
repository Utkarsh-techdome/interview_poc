from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sessions / Interview schemas
# ---------------------------------------------------------------------------

class InterviewConfig(BaseModel):
    role: str = Field(default="Software Engineer")
    candidate_name: str = Field(default="Candidate")
    # Legacy flat list — still accepted for backward compat
    questions: list[str] = Field(default=[])
    # Structured question bank from generate_question_bank()
    # Each item: {id, question_type, text, anchor, follow_up_seeds, depth_gate}
    question_bank: list[dict] = Field(default=[])
    # Resume-derived fields
    candidate_skills: list[str] = Field(default=[])
    candidate_experience: list[dict] = Field(default=[])
    jd_skills: list[str] = Field(default=[])

class GenerateQuestionsRequest(BaseModel):
    role: str = Field(..., description="Job role being interviewed for")
    candidate_name: str = Field(..., description="Candidate's name")
    candidate_skills: list[str] = Field(default=[], description="Skills extracted from resume")
    candidate_experience: list[dict] = Field(default=[], description="Experience extracted from resume")
    jd_skills: list[str] = Field(default=[], description="Skills required by job description")

class GenerateQuestionsResponse(BaseModel):
    role: str
    candidate_name: str
    question_count: int
    question_bank: list[dict] 

class SessionResponse(BaseModel):
    session_id: str
    role: str
    candidate_name: str
    question_count: int
    status: str
    created_at: str

class SessionStatus(BaseModel):
    session_id: str
    status: str
    conversation_turns: int
    questions_total: int
    duration_seconds: Optional[float]
    created_at: str

# ---------------------------------------------------------------------------
# Resume Screening schemas
# ---------------------------------------------------------------------------

class CandidateEvalResponse(BaseModel):
    candidate_id: Optional[str] = None
    experience_similarity: float = Field(..., ge=0.0, le=1.0)
    experience_strength: float = Field(..., ge=0.0, le=1.0)
    experience_strength_breakdown: dict = Field(default_factory=dict)
    experience_strength_reason: str = Field(default="")
    skill_coverage: float = Field(..., ge=0.0, le=1.0)
    fit_score: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="'interview' or 'reject'")
    threshold: float
    # Extracted data for question generation
    candidate_name: Optional[str] = None
    role: Optional[str] = None
    candidate_skills: list[str] = Field(default_factory=list)
    candidate_experience: list[dict] = Field(default_factory=list)
    jd_skills: list[str] = Field(default_factory=list)
