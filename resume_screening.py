"""
resume_screening.py — Structured Extraction + Weighted Semantic Similarity Pipeline
=====================================================================================

Pipeline:
  1. PDF → pdfplumber → raw text (JD + Resume, concurrent)
  2. raw text → llama3.2:3b (Ollama) → structured JSON {skills, experience}
  3. skills list → embedding string → nomic-embed-text → vector
  4. experience descriptions → concatenated string → nomic-embed-text → vector
  5. cosine_similarity per component
  6. fit_score = 0.2 * skill_similarity + 0.8 * experience_similarity

fit_score is normalized to [0, 1].
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from typing import Any

import numpy as np
import ollama
import pdfplumber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXTRACTION_MODEL  = "minimax-m2.7:cloud"
EMBEDDING_MODEL   = "nomic-embed-text"
EXP_SIM_WEIGHT    = 0.35
EXP_STR_WEIGHT    = 0.40
SKILL_COV_WEIGHT  = 0.25
DEFAULT_THRESHOLD = 0.50

EXTRACTION_SCHEMA = """{
  "skills": ["string"],
  "experience": [
    {
      "role": "string",
      "description": "string",
      "years": number
    }
  ]
}"""

EXTRACTION_PROMPT = """\
Extract structured information from the following text and return ONLY valid JSON \
matching this exact schema (no extra text, no markdown fences):

{schema}

Rules:
- "skills": technical and domain-relevant skills only, deduplicated, lowercase
- "experience": each role as a separate entry
- "description": concise 1-2 line summary of what was done
- "years": estimated duration as a number (use 0 if unknown)
- Ignore empty/null fields

Text:
{text}

JSON output:"""

# ---------------------------------------------------------------------------
# Step 1: PDF → text
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF byte string using pdfplumber."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            "Could not extract any text from the PDF. "
            "Make sure it is not a scanned/image-only PDF."
        )
    return text


# ---------------------------------------------------------------------------
# Step 2: Structured extraction via llama3.2:3b (async)
# ---------------------------------------------------------------------------

def _call_ollama_chat(model: str, prompt: str) -> str:
    """Synchronous Ollama chat call (run in threadpool for async contexts)."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


def _robust_get(data: dict, key: str, default: Any = None) -> Any:
    """
    Robustly get a value from a dict even if the key has literal quotes,
    different casing, or extra whitespace.
    """
    if not isinstance(data, dict):
        return default
    # Try exact match first
    if key in data:
        return data[key]
    # Try case-insensitive and quote-stripped search
    target = key.lower().strip().strip('"').strip("'")
    for k, v in data.items():
        clean_k = str(k).lower().strip().strip('"').strip("'")
        if clean_k == target:
            return v
    return default


def _parse_structured(raw: str) -> dict[str, Any]:
    """
    Parse the LLM's JSON output into a validated dict.
    Falls back gracefully on malformed output.
    """
    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(raw)
        logger.info(f"Parsed structured JSON keys: {list(data.keys())}")
    except json.JSONDecodeError:
        logger.warning(f"LLM returned malformed JSON: {raw[:200]}")
        return {"skills": [], "experience": []}

    # Normalize
    skills_raw = _robust_get(data, "skills", [])
    skills = [
        s.lower().strip()
        for s in skills_raw
        if s and isinstance(s, str)
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped_skills: list[str] = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            deduped_skills.append(s)

    experience = []
    experience_raw = _robust_get(data, "experience", [])
    for exp in experience_raw:
        if not isinstance(exp, dict):
            continue
        role = str(_robust_get(exp, "role", "")).strip()
        desc = str(_robust_get(exp, "description", "")).strip()
        try:
            years = float(_robust_get(exp, "years", 0))
        except (TypeError, ValueError):
            years = 0.0
        if role or desc:
            experience.append({"role": role, "description": desc, "years": years})

    return {"skills": deduped_skills, "experience": experience}


async def extract_structured_data(text: str) -> dict[str, Any]:
    """
    Async wrapper: sends text to llama3.2:3b and returns structured JSON.
    Retries once on malformed output.
    """
    prompt = EXTRACTION_PROMPT.format(schema=EXTRACTION_SCHEMA, text=text)
    loop = asyncio.get_event_loop()

    raw = await loop.run_in_executor(None, _call_ollama_chat, EXTRACTION_MODEL, prompt)
    result = _parse_structured(raw)

    # If both keys are empty, retry once
    if not result["skills"] and not result["experience"]:
        logger.warning("Empty extraction result; retrying once...")
        raw = await loop.run_in_executor(None, _call_ollama_chat, EXTRACTION_MODEL, prompt)
        result = _parse_structured(raw)

    return result


# ---------------------------------------------------------------------------
# Step 3: Embedding generation (single or batched)
# ---------------------------------------------------------------------------

def _embed_sync(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed a list of texts via Ollama.
    Ollama's embed endpoint accepts a single prompt, so we call it per-text
    but gather results efficiently.
    """
    results = []
    for text in texts:
        resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        results.append(resp["embedding"])
    return results


async def generate_embeddings(texts: list[str]) -> list[np.ndarray]:
    """Async wrapper for batch embedding via Ollama."""
    loop = asyncio.get_event_loop()
    vecs = await loop.run_in_executor(None, _embed_sync, texts)
    return [np.array(v, dtype=np.float32) for v in vecs]


# ---------------------------------------------------------------------------
# Step 4: Cosine similarity
# ---------------------------------------------------------------------------

def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity normalized to [0, 1]."""
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    raw = float(np.dot(vec1, vec2) / (n1 * n2))
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Skill coverage  (exact set-intersection ratio)
# ---------------------------------------------------------------------------

def compute_skill_coverage(
    resume_skills: list[str],
    jd_skills: list[str],
) -> float:
    """
    skill_coverage = len(matched_skills) / len(jd_skills)
    Skills are already normalized (lowercase, trimmed) from extraction.
    Matching uses exact equality first, then substring containment as fallback.
    """
    if not jd_skills:
        return 0.0
    resume_set = set(resume_skills)
    matched = 0
    for jd_skill in jd_skills:
        if jd_skill in resume_set:
            matched += 1
        else:
            # substring fallback: e.g. "ml" matches "machine learning"
            if any(jd_skill in rs or rs in jd_skill for rs in resume_set):
                matched += 1
    return float(np.clip(matched / len(jd_skills), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Experience strength  (LLM-powered quality score)
# ---------------------------------------------------------------------------

EXPERIENCE_STRENGTH_PROMPT = """\
You are an expert technical interviewer.

Your task is to evaluate the QUALITY of a candidate's experience relative to a Job Description.

## Input:

1. Job Description (JD):
{jd_experience}

2. Candidate Experience:
{resume_experience}

## Evaluation Criteria (strict):

Score the candidate from 0 to 1 based on:

1. Relevance (0-0.3)
   * How closely does the experience match the JD requirements?

2. Depth (0-0.3)
   * Does the candidate demonstrate strong technical ownership?
   * Are they building/designing vs just "working on"?

3. Impact (0-0.2)
   * Are there measurable outcomes? (metrics, scale, improvements)

4. Complexity (0-0.2)
   * Does the work involve non-trivial systems, tools, or challenges?

## Instructions:
* Be strict. Avoid giving high scores unless clearly justified.
* Penalize vague phrases like "worked on", "familiar with"
* Reward specific tools, metrics, and ownership

## Output (STRICT JSON ONLY):
{{"experience_strength": 0.0, "breakdown": {{"relevance": 0.0, "depth": 0.0, "impact": 0.0, "complexity": 0.0}}, "reason": "string"}}"""


def _format_experience(experience: list[dict]) -> str:
    if not experience:
        return "No experience listed."
    lines = []
    for e in experience:
        role = e.get("role", "Unknown Role")
        desc = e.get("description", "")
        years = e.get("years", 0)
        lines.append(f"- {role} ({years}y): {desc}")
    return "\n".join(lines)


def _call_experience_strength(
    jd_experience: str,
    resume_experience: str,
) -> dict[str, Any]:
    """Synchronous LLM call for experience quality scoring."""
    prompt = EXPERIENCE_STRENGTH_PROMPT.format(
        jd_experience=jd_experience,
        resume_experience=resume_experience,
    )
    response = ollama.chat(
        model=EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"]
    raw = re.sub(r"^```(?:json)?\s*|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(raw)
        print(f"DEBUG: LLM Experience Strength Keys: {list(data.keys())}")
    except json.JSONDecodeError:
        print(f"DEBUG: LLM Experience Strength JSON error. Raw: {raw[:200]}")
        logger.warning("experience_strength LLM returned malformed JSON; defaulting to 0.")
        return {"experience_strength": 0.0, "breakdown": {}, "reason": "parse error"}
    
    strength = float(np.clip(_robust_get(data, "experience_strength", 0.0), 0.0, 1.0))
    breakdown = _robust_get(data, "breakdown", {})
    reason = str(_robust_get(data, "reason", "")).strip()

    res = {
        "experience_strength": round(strength, 4),
        "breakdown": breakdown,
        "reason": reason,
    }
    print(f"DEBUG: Final strength_result keys: {list(res.keys())}")
    return res


async def evaluate_experience_strength(
    jd_struct: dict[str, Any],
    resume_struct: dict[str, Any],
) -> dict[str, Any]:
    """Async wrapper: calls LLM in threadpool to score experience quality."""
    jd_exp_str  = _format_experience(jd_struct.get("experience", []))
    res_exp_str = _format_experience(resume_struct.get("experience", []))
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _call_experience_strength, jd_exp_str, res_exp_str
    )


# ---------------------------------------------------------------------------
# Step 5: Aggregate into fit_score
# ---------------------------------------------------------------------------

def _experience_string(experience: list[dict]) -> str:
    parts = [e["description"] for e in experience if e.get("description")]
    return " ".join(parts) if parts else "no experience listed"


async def compute_fit_score(
    resume_struct: dict[str, Any],
    jd_struct: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute weighted fit_score:
      fit_score = 0.35 * experience_similarity
               + 0.40 * experience_strength   (LLM)
               + 0.25 * skill_coverage        (set intersection)
    """
    resume_exp_str = _experience_string(resume_struct["experience"])
    jd_exp_str     = _experience_string(jd_struct["experience"])

    # Run experience embedding + LLM strength evaluation concurrently
    exp_embs, strength_result = await asyncio.gather(
        generate_embeddings([resume_exp_str, jd_exp_str]),
        evaluate_experience_strength(jd_struct, resume_struct),
    )
    resume_exp_emb, jd_exp_emb = exp_embs

    experience_similarity = compute_similarity(resume_exp_emb, jd_exp_emb)
    experience_strength   = strength_result["experience_strength"]
    skill_coverage        = compute_skill_coverage(
        resume_struct["skills"], jd_struct["skills"]
    )

    fit_score = (
        EXP_SIM_WEIGHT   * experience_similarity
        + EXP_STR_WEIGHT * experience_strength
        + SKILL_COV_WEIGHT * skill_coverage
    )
    fit_score = float(np.clip(fit_score, 0.0, 1.0))

    return {
        "experience_similarity":       round(experience_similarity, 4),
        "experience_strength":         round(experience_strength, 4),
        "experience_strength_breakdown": strength_result["breakdown"],
        "experience_strength_reason":  strength_result["reason"],
        "skill_coverage":              round(skill_coverage, 4),
        "fit_score":                   round(fit_score, 4),
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

async def evaluate_candidate(
    jd_bytes: bytes,
    resume_bytes: bytes,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    End-to-end evaluation pipeline.

    Returns:
        {
          "experience_similarity": float,
          "experience_strength": float,
          "experience_strength_breakdown": dict,
          "experience_strength_reason": str,
          "skill_coverage": float,
          "fit_score": float,
          "decision": "interview" | "reject",
        }
    """
    # 1. Extract text from PDFs
    jd_text     = extract_text_from_pdf(jd_bytes)
    resume_text = extract_text_from_pdf(resume_bytes)

    # 2. Extract structured data from both concurrently
    logger.info("Extracting structured data from JD and Resume concurrently...")
    jd_struct, resume_struct = await asyncio.gather(
        extract_structured_data(jd_text),
        extract_structured_data(resume_text),
    )
    logger.info(f"JD skills: {len(jd_struct['skills'])}, "
                f"Resume skills: {len(resume_struct['skills'])}")

    # 3-5. Score
    logger.info("Computing fit score...")
    scores = await compute_fit_score(resume_struct, jd_struct)

    # 6. Decision
    decision = "interview" if scores["fit_score"] >= threshold else "reject"
    logger.info(f"fit_score={scores['fit_score']}, decision={decision}")

    return {
        **scores,
        "decision": decision,
        "candidate_skills": resume_struct["skills"],
        "candidate_experience": resume_struct["experience"],
        "jd_skills": jd_struct["skills"],
    }
