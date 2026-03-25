# =============================================================================
# resume_screening.py — Candidate fit evaluation & data extraction
# =============================================================================

import asyncio
import json
import logging
import re
from typing import Any
import numpy as np
import ollama

from prompts import EXTRACTION_SCHEMA, EXTRACTION_PROMPT, EXPERIENCE_STRENGTH_PROMPT
from utils import extract_text_from_pdf, safe_json_parse, format_experience
from constants import (
    EXTRACTION_MODEL, EMBEDDING_MODEL, STRENGTH_MODEL,
    EXP_SIM_WEIGHT, EXP_STR_WEIGHT, SKILL_COV_WEIGHT,
    DEFAULT_THRESHOLD
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("interview_agent.screening")

# ---------------------------------------------------------------------------
# Step 2: Structured Extraction
# ---------------------------------------------------------------------------

async def extract_structured_data(text: str) -> dict:
    """Uses LLM to extract structured skills and experience from text."""
    if not text.strip():
        return {"skills": [], "experience": []}

    prompt = EXTRACTION_PROMPT.format(schema=EXTRACTION_SCHEMA, text=text)
    
    try:
        response = await asyncio.to_thread(
            ollama.chat,
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0}
        )
        raw_json = response["message"]["content"]
        extracted = safe_json_parse(raw_json)
        
        # Normalize skills
        skills = [s.lower().strip() for s in extracted.get("skills", []) if isinstance(s, str)]
        # Deduplicate
        skills = list(dict.fromkeys(skills))
        
        return {
            "candidate_name": extracted.get("candidate_name"),
            "role": extracted.get("role"),
            "skills": skills,
            "experience": extracted.get("experience", [])
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"skills": [], "experience": []}

# ---------------------------------------------------------------------------
# Step 3 & 4: Embeddings & Similarity
# ---------------------------------------------------------------------------

async def generate_embeddings(texts: list[str]) -> list[np.ndarray]:
    """Batch-embed a list of texts via Ollama."""
    try:
        results = []
        for text in texts:
            # Simple wrapper since Ollama SDK doesn't have a direct batch embed in some versions
            resp = await asyncio.to_thread(ollama.embeddings, model=EMBEDDING_MODEL, prompt=text)
            results.append(np.array(resp["embedding"], dtype=np.float32))
        return results
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Return zero vectors if failed to avoid crashing, but score will be 0.
        return [np.zeros(768, dtype=np.float32) for _ in texts]

def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity normalized to [0, 1]."""
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    raw = float(np.dot(vec1, vec2) / (n1 * n2))
    # Normalize cosine [-1, 1] to [0, 1]
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

# ---------------------------------------------------------------------------
# Step 5: Skill Coverage
# ---------------------------------------------------------------------------

def compute_skill_coverage(res_skills: list[str], jd_skills: list[str]) -> float:
    """
    Computes skill match ratio with substring fallback.
    e.g., 'python' matches 'python 3.8' or 'ml' matches 'machine learning'.
    """
    if not jd_skills:
        return 1.0
    
    res_set = set(res_skills)
    matched = 0
    for jd_skill in jd_skills:
        if jd_skill in res_set:
            matched += 1
        elif any(jd_skill in rs or rs in jd_skill for rs in res_set):
            matched += 1
            
    return float(np.clip(matched / len(jd_skills), 0.0, 1.0))

# ---------------------------------------------------------------------------
# Step 6: Experience Strength (LLM)
# ---------------------------------------------------------------------------

async def _call_experience_strength(
    jd_data: dict[str, Any],
    res_data: dict[str, Any],
) -> dict[str, Any]:
    """Asynchronous LLM call for experience quality scoring."""
    jd_str = format_experience(jd_data["experience"])
    res_str = format_experience(res_data["experience"])
    
    try:
        response = await asyncio.to_thread(
            ollama.chat,
            model=STRENGTH_MODEL,
            messages=[{"role": "user", "content": EXPERIENCE_STRENGTH_PROMPT.format(
                jd_experience=jd_str,
                resume_experience=res_str
            )}],
            format="json",
            options={"temperature": 0.0}
        )
        raw = response["message"]["content"]
        data = safe_json_parse(raw)
        
        return {
            "experience_strength": float(data.get("experience_strength", 0.0)),
            "breakdown": data.get("breakdown", {}),
            "reason": data.get("reason", "N/A")
        }
    except Exception as e:
        logger.error(f"Strength evaluation failed: {e}")
        return {"experience_strength": 0.0, "reason": "Evaluation failed"}

# ---------------------------------------------------------------------------
# Core Evaluation Pipeline
# ---------------------------------------------------------------------------

def _experience_string(experience: list[dict]) -> str:
    parts = [e["description"] for e in experience if e.get("description")]
    return " ".join(parts) if parts else "no experience listed"

async def evaluate_candidate(
    jd_bytes: bytes,
    resume_bytes: bytes,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Complete screening pipeline using original weighted scoring logic.
    """
    # 1. Extraction from PDF
    jd_text = extract_text_from_pdf(jd_bytes)
    resume_text = extract_text_from_pdf(resume_bytes)

    if not jd_text or not resume_text:
        return {
            "fit_score": 0.0, "decision": "reject", 
            "reason": "Failed to read PDF content"
        }

    # 2. Structured data extraction
    jd_data, res_data = await asyncio.gather(
        extract_structured_data(jd_text),
        extract_structured_data(resume_text)
    )

    # 3. Parallel tasks: Embeddings and LLM Strength
    resume_exp_str = _experience_string(res_data["experience"])
    jd_exp_str     = _experience_string(jd_data["experience"])
    
    exp_embs_task = generate_embeddings([resume_exp_str, jd_exp_str])
    strength_task = _call_experience_strength(jd_data, res_data)
    
    exp_embs, strength_result = await asyncio.gather(exp_embs_task, strength_task)
    res_emb, jd_emb = exp_embs

    # 4. Component scores
    experience_similarity = compute_similarity(res_emb, jd_emb)
    experience_strength   = strength_result["experience_strength"]
    skill_coverage        = compute_skill_coverage(res_data["skills"], jd_data["skills"])

    # 5. Weighted Final Score
    fit_score = (
        (experience_similarity * EXP_SIM_WEIGHT) +
        (experience_strength   * EXP_STR_WEIGHT) +
        (skill_coverage        * SKILL_COV_WEIGHT)
    )
    fit_score = float(np.clip(fit_score, 0.0, 1.0))

    return {
        "experience_similarity": round(experience_similarity, 4),
        "experience_strength": round(experience_strength, 4),
        "experience_strength_breakdown": strength_result.get("breakdown", {}),
        "experience_strength_reason": strength_result.get("reason", "N/A"),
        "skill_coverage": round(skill_coverage, 4),
        "fit_score": round(fit_score, 4),
        "decision": "interview" if fit_score >= threshold else "reject",
        "threshold": threshold,
        "candidate_skills": res_data["skills"],
        "candidate_experience": res_data["experience"],
        "candidate_name": res_data.get("candidate_name"),
        "role": jd_data.get("role") or res_data.get("role"),
        "jd_skills": jd_data["skills"]
    }
