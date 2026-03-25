# =============================================================================
# main.py -- FastAPI Server Entry Point
# =============================================================================

from __future__ import annotations
import os
import uuid
import logging
import traceback
from typing import Optional
from pathlib import Path

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
from dotenv import load_dotenv

# Modular imports
load_dotenv()
from models import (
    InterviewConfig, GenerateQuestionsRequest, GenerateQuestionsResponse,
    SessionResponse, SessionStatus, CandidateEvalResponse
)
from agent import InterviewSession
from resume_screening import evaluate_candidate, DEFAULT_THRESHOLD
from question_generator import generate_question_bank

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview_agent.main")

app = FastAPI(title="Deepgram Voice Interviewing Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: dict[str, InterviewSession] = {}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/evaluate-candidate", response_model=CandidateEvalResponse)
async def evaluate_candidate_endpoint(
    jd_file: UploadFile = File(...),
    resume_file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
):
    try:
        jd_bytes = await jd_file.read()
        res_bytes = await resume_file.read()
        
        result = await evaluate_candidate(jd_bytes, res_bytes, threshold)
        
        return CandidateEvalResponse(
            candidate_id=str(uuid.uuid4()),
            experience_similarity=result["experience_similarity"],
            experience_strength=result["experience_strength"],
            experience_strength_breakdown=result["experience_strength_breakdown"],
            experience_strength_reason=result["experience_strength_reason"],
            skill_coverage=result["skill_coverage"],
            fit_score=result["fit_score"],
            decision=result["decision"],
            threshold=threshold,
            candidate_name=result.get("candidate_name"),
            role=result.get("role"),
            candidate_skills=result.get("candidate_skills", []),
            candidate_experience=result.get("candidate_experience", []),
            jd_skills=result.get("jd_skills", []),
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions_endpoint(req: GenerateQuestionsRequest):
    try:
        bank = await generate_question_bank(
            role=req.role,
            candidate_name=req.candidate_name,
            candidate_skills=req.candidate_skills,
            candidate_experience=req.candidate_experience,
            jd_skills=req.jd_skills
        )
        return GenerateQuestionsResponse(
            role=req.role,
            candidate_name=req.candidate_name,
            question_count=len(bank),
            question_bank=bank
        )
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions", response_model=SessionResponse)
async def create_session(cfg: InterviewConfig):
    session_id = str(uuid.uuid4())
    session = InterviewSession(session_id, cfg.model_dump())
    sessions[session_id] = session
    
    return SessionResponse(
        session_id=session_id,
        role=session.cfg["role"],
        candidate_name=session.cfg["candidate_name"],
        question_count=len(session.cfg["question_bank"]) or len(session.cfg["questions"]),
        status=session.status,
        created_at=session.created_at.isoformat()
    )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session = sessions.get(session_id)
    if not session:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    await websocket.accept()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        await websocket.send_json({"type": "error", "message": "DEEPGRAM_API_KEY missing on server"})
        await websocket.close()
        return

    try:
        await session.run_async(websocket, api_key)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WS error: {e}")
    finally:
        if session_id in sessions:
            # We keep it in memory for status/report retrieval
            pass

@app.get("/sessions/{session_id}/report")
async def get_report(session_id: str):
    report_path = Path(f"data/{session_id}/interview_report.md")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet")
    return FileResponse(report_path)

# Static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)