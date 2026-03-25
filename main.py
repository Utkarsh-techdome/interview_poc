# =============================================================================
# Deepgram Voice Interviewing Agent — FastAPI Server
# =============================================================================
#
# Architecture:
#
#   Browser  ──── REST ────►  FastAPI  (session management, reports)
#   Browser  ── WebSocket ──► FastAPI  ── WebSocket ──► Deepgram Agent API
#                  ▲                                         │
#                  └──────── audio PCM back ─────────────────┘
#
# The browser sends raw PCM audio chunks over WS and receives agent PCM audio
# back in real time. No PyAudio / server microphone is needed.
#
# Requirements:
#   pip install fastapi uvicorn deepgram-sdk python-multipart websockets
#
# Run:
#   export DEEPGRAM_API_KEY="your_key"
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Docs: http://localhost:8000/docs
# =============================================================================

from __future__ import annotations

import asyncio
import base64
from dotenv import load_dotenv
load_dotenv()
import datetime
import json
import logging
import os
import traceback
import uuid
import wave
from pathlib import Path
from typing import Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import InterviewSession, build_system_prompt, DEFAULT_CONFIG, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH
from resume_screening import evaluate_candidate, DEFAULT_THRESHOLD

from question_generator import generate_question_bank
# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Interview Agent API",
    description="AI Interviewer and Resume Screening Pipeline",
    version="1.0.0",
)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the browser UI)
Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Sessions store: session_id → InterviewSession
SESSIONS: dict[str, InterviewSession] = {}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class InterviewConfig(BaseModel):
    role: str = Field(default="Software Engineer")
    candidate_name: str = Field(default="Candidate")
 
    # Legacy flat list — still accepted for backward compat.
    # Ignored when question_bank is provided.
    questions: list[str] = Field(default=[])
 
    # Structured question bank from generate_question_bank().
    # Each item: {id, question_type, text, anchor, follow_up_seeds, depth_gate}
    question_bank: list[dict] = Field(default=[])
 
    # Resume-derived fields (from /evaluate-candidate response)
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
    experience_similarity: float = Field(..., ge=0.0, le=1.0,
                                         description="Cosine similarity of experience embeddings (0.35 weight)")
    experience_strength: float = Field(..., ge=0.0, le=1.0,
                                       description="LLM quality score for experience (0.40 weight)")
    experience_strength_breakdown: dict = Field(default_factory=dict,
                                                description="Relevance/depth/impact/complexity sub-scores")
    experience_strength_reason: str = Field(default="",
                                            description="Short LLM explanation")
    skill_coverage: float = Field(..., ge=0.0, le=1.0,
                                  description="Fraction of JD skills matched in resume (0.25 weight)")
    fit_score: float = Field(..., ge=0.0, le=1.0,
                             description="0.35*exp_sim + 0.40*exp_str + 0.25*skill_cov")
    decision: str = Field(..., description="'interview' if fit_score >= threshold, else 'reject'")
    threshold: float

    # Extracted data for question generation
    candidate_skills: list[str] = Field(default_factory=list)
    candidate_experience: list[dict] = Field(default_factory=list)
    jd_skills: list[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _session_or_404(session_id: str) -> InterviewSession:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session

# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the browser UI."""
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Resume Screening endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/evaluate-candidate",
    response_model=CandidateEvalResponse,
    summary="Screen a candidate resume against a job description",
    tags=["Resume Screening"],
)
async def evaluate_candidate_endpoint(
    jd_file: UploadFile = File(..., description="Job description PDF"),
    resume_file: UploadFile = File(..., description="Candidate resume PDF"),
    candidate_id: Optional[str] = Form(default=None, description="Optional candidate identifier"),
    threshold: float = Form(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0,
                            description="Minimum fit_score to recommend interview (default 0.5)"),
):
    """
    Evaluate a candidate resume against a job description.

    **Pipeline:**
    1. Extract text from both PDFs via pdfplumber
    2. Use `llama3.2:3b` (Ollama) to extract structured JSON {skills, experience}
       from JD and Resume **concurrently**
    3. Generate embeddings via `nomic-embed-text` (Ollama)
    4. Cosine similarity: skill_sim + experience_sim
    5. fit_score = 0.2 * skill_sim + 0.8 * exp_sim

    **Requirements:**
    - Ollama must be running locally with `llama3.2:3b` and `nomic-embed-text` pulled
    """
    # Validate MIME types
    for f, label in [(jd_file, "jd_file"), (resume_file, "resume_file")]:
        if f.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{label} must be a PDF (got {f.content_type})",
            )

    jd_bytes = await jd_file.read()
    resume_bytes = await resume_file.read()

    try:
        result = await evaluate_candidate(
            jd_bytes=jd_bytes,
            resume_bytes=resume_bytes,
            threshold=threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        logger.error(f"Pipeline failure: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return CandidateEvalResponse(
        candidate_id=candidate_id,
        experience_similarity=result["experience_similarity"],
        experience_strength=result["experience_strength"],
        experience_strength_breakdown=result.get("experience_strength_breakdown", {}),
        experience_strength_reason=result.get("experience_strength_reason", ""),
        skill_coverage=result["skill_coverage"],
        fit_score=result["fit_score"],
        decision=result["decision"],
        threshold=threshold,
        candidate_skills=result.get("candidate_skills", []),
        candidate_experience=result.get("candidate_experience", []),
        jd_skills=result.get("jd_skills", []),
    )

@app.post(
    "/generate-questions",
    response_model=GenerateQuestionsResponse,
    summary="Generate a personalised question bank for a candidate",
    tags=["Interview"],
)
async def generate_questions_endpoint(req: GenerateQuestionsRequest):
    """
    Generate a resume-anchored, role-aware question bank.
 
    **Typical flow:**
    1. POST /evaluate-candidate  → get candidate_skills, candidate_experience
    2. POST /generate-questions  → get question_bank   ← THIS ENDPOINT
    3. POST /sessions            → pass question_bank in the body
 
    The question_bank returned here should be passed verbatim as the
    `question_bank` field when creating a session.
 
    **Requirements:**
    - Ollama must be running with the extraction model available.
    """
    try:
        bank = await generate_question_bank(
            role=req.role,
            candidate_name=req.candidate_name,
            candidate_skills=req.candidate_skills,
            candidate_experience=req.candidate_experience,
            jd_skills=req.jd_skills,
        )
    except Exception as exc:
        # Never block session creation — fall back gracefully
        logger.error(f"Question generation error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {exc}. Use /sessions with a manual questions list as fallback.",
        )
 
    return GenerateQuestionsResponse(
        role=req.role,
        candidate_name=req.candidate_name,
        question_count=len(bank),
        question_bank=bank,
    )


@app.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new interview session",
)
async def create_session(cfg: InterviewConfig):
    """
    Create a new interview session.
 
    **Recommended:** pass `question_bank` (from /generate-questions) rather
    than the legacy `questions` list. When `question_bank` is present,
    `questions` is ignored.
 
    If neither is provided, a generic fallback question set is used.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DEEPGRAM_API_KEY is not configured on the server",
        )
 
    # Ensure at least a fallback question list exists
    cfg_dict = cfg.model_dump()
    if not cfg_dict.get("question_bank") and not cfg_dict.get("questions"):
        from agent import DEFAULT_CONFIG
        cfg_dict["questions"] = DEFAULT_CONFIG["questions"]
 
    session_id = str(uuid.uuid4())
    session = InterviewSession(session_id=session_id, cfg=cfg_dict)
    SESSIONS[session_id] = session
 
    question_count = (
        len(cfg_dict["question_bank"]) if cfg_dict.get("question_bank")
        else len(cfg_dict.get("questions", []))
    )
 
    return SessionResponse(
        session_id=session_id,
        role=cfg.role,
        candidate_name=cfg.candidate_name,
        question_count=question_count,
        status=session.status,
        created_at=session.created_at.isoformat(),
    )


@app.get(
    "/sessions",
    summary="List all sessions",
)
async def list_sessions():
    return [
        {
            "session_id": sid,
            "role": s.cfg["role"],
            "candidate_name": s.cfg["candidate_name"],
            "status": s.status,
            "created_at": s.created_at.isoformat(),
        }
        for sid, s in SESSIONS.items()
    ]


@app.get(
    "/sessions/{session_id}",
    response_model=SessionStatus,
    summary="Get session status",
)
async def get_session(session_id: str):
    s = _session_or_404(session_id)
    duration = None
    if s.end_time:
        duration = (s.end_time - s.created_at).total_seconds()
    return SessionStatus(
        session_id=session_id,
        status=s.status,
        conversation_turns=len(s.conversation),
        questions_total=len(s.cfg["questions"]),
        duration_seconds=duration,
        created_at=s.created_at.isoformat(),
    )


@app.get(
    "/sessions/{session_id}/transcript",
    summary="Get the conversation transcript",
)
async def get_transcript(session_id: str):
    s = _session_or_404(session_id)
    return {"session_id": session_id, "conversation": s.conversation}


@app.get(
    "/sessions/{session_id}/report",
    summary="Download the Markdown interview report",
)
async def get_report(session_id: str):
    s = _session_or_404(session_id)
    report_path = Path(f"data/{session_id}/interview_report.md")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not yet generated")
    return FileResponse(
        path=str(report_path),
        media_type="text/markdown",
        filename=f"interview_report_{session_id[:8]}.md",
    )


@app.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
)
async def delete_session(session_id: str):
    _session_or_404(session_id)
    del SESSIONS[session_id]


# ---------------------------------------------------------------------------
# WebSocket — real-time audio bridge
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_bridge(websocket: WebSocket, session_id: str):
    """
    WebSocket bridge between the browser and Deepgram's Agent API.

    Browser → server:  raw PCM audio bytes (linear16, 24 kHz, mono)
    Server  → browser: JSON events  { type, ... }
                        binary audio chunks (agent speech PCM)
    """
    await websocket.accept()

    session = SESSIONS.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    if session.status not in ("created", "idle"):
        await websocket.send_json({"type": "error", "message": f"Session is already {session.status}"})
        await websocket.close()
        return

    try:
        await session.run_async(websocket)
    except WebSocketDisconnect:
        session._log("Browser WebSocket disconnected")
        session.status = "ended"
    except Exception as e:
        session._log(f"Bridge error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass