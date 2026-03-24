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
import datetime
import json
import os
import uuid
import wave
from pathlib import Path
from typing import Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import InterviewSession, build_system_prompt, DEFAULT_CONFIG, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Interview Agent",
    description="Real-time voice interviewing agent powered by Deepgram",
    version="1.0.0",
)

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
    role: str = Field(default="Software Engineer", description="Job role being interviewed for")
    candidate_name: str = Field(default="Candidate", description="Candidate's name")
    questions: list[str] = Field(
        default=DEFAULT_CONFIG["questions"],
        description="Ordered list of interview questions",
    )

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


@app.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new interview session",
)
async def create_session(cfg: InterviewConfig):
    """
    Create a new interview session with the given configuration.
    Returns a session_id used for all subsequent calls.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DEEPGRAM_API_KEY is not configured on the server",
        )

    session_id = str(uuid.uuid4())
    session = InterviewSession(
        session_id=session_id,
        cfg=cfg.model_dump(),
    )
    SESSIONS[session_id] = session

    return SessionResponse(
        session_id=session_id,
        role=cfg.role,
        candidate_name=cfg.candidate_name,
        question_count=len(cfg.questions),
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