# =============================================================================
# question_generator.py — Resume-anchored, role-aware question bank generator
# =============================================================================
#
# Called ONCE before a session starts. Takes the structured resume data that
# resume_screening.py already produces and generates a typed question bank
# with per-question follow-up seeds and depth gates.
#
# The LLM is only used here (and in resume_screening). The Deepgram agent
# (GPT-4o-mini) receives the final bank via the system prompt — it does NOT
# call this itself.
#
# Usage:
#   bank = await generate_question_bank(
#       role="ML Engineer",
#       candidate_name="Alex",
#       candidate_skills=["python", "pytorch", "langchain"],
#       candidate_experience=[{"role": "MLE", "description": "...", "years": 2}],
#       jd_skills=["python", "kubernetes", "mlops"],
#   )
# =============================================================================

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import ollama

from prompts import GENERATION_PROMPT, QUESTION_COUNTS
from utils import safe_json_parse, format_skills, format_experience
from constants import GENERATION_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_skill_gaps(
    candidate_skills: list[str],
    jd_skills: list[str],
) -> list[str]:
    """Return JD skills not found in the candidate's resume (case-insensitive)."""
    resume_set = {s.lower().strip() for s in candidate_skills}
    gaps = []
    for jd_skill in jd_skills:
        norm = jd_skill.lower().strip()
        if norm not in resume_set and not any(norm in rs or rs in norm for rs in resume_set):
            gaps.append(jd_skill)
    return gaps


def _parse_bank(raw: str) -> list[dict[str, Any]]:
    """Parse and validate the LLM's JSON output."""
    raw = re.sub(r"^```(?:json)?\s*|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Question generator returned malformed JSON: {raw[:300]}")
        return []

    questions = data.get("questions", [])
    validated = []
    seen_texts = set()

    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
            
        q_text = str(q.get("text", "")).strip()
        if not q_text:
            continue
            
        # Deduplication check: normalise and check if we've seen a very similar question
        # (Llama 3.2 3B sometimes copy-pastes JSON objects to pad the array)
        norm_text = re.sub(r'[^a-z0-9]', '', q_text.lower())
        if norm_text in seen_texts:
            logger.warning(f"Generator duplicate question stripped: {q_text[:30]}")
            continue
        seen_texts.add(norm_text)

        # Normalise required fields with safe defaults
        validated.append({
            "id": f"q{len(validated)+1}",
            "question_type": q.get("question_type", "behavioural"),
            "text": q_text,
            "anchor": str(q.get("anchor", "")).strip(),
            "follow_up_seeds": [
                str(s) for s in q.get("follow_up_seeds", []) if s
            ][:2],   # cap at 2
            "depth_gate": {
                "requires_concrete_example": bool(
                    q.get("depth_gate", {}).get("requires_concrete_example", False)
                ),
                "requires_metric": bool(
                    q.get("depth_gate", {}).get("requires_metric", False)
                ),
            },
        })
    return validated


def _call_ollama(prompt: str) -> str:
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.3},   # slight creativity, still deterministic enough
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Fallback bank  (used if LLM call fails or returns empty)
# ---------------------------------------------------------------------------

def _fallback_bank(role: str) -> list[dict[str, Any]]:
    """
    Generic question bank used when LLM generation fails.
    Better than crashing — the interview still runs, just not personalised.
    """
    logger.warning("Using fallback question bank (LLM generation failed)")
    return [
        {
            "id": "q1",
            "question_type": "behavioural",
            "text": "Can you tell me a bit about yourself and what drew you to this role?",
            "anchor": "Fallback opener",
            "follow_up_seeds": [
                "What has been the most technically challenging project you've worked on?",
                "How did that experience shape where you want to go next?",
            ],
            "depth_gate": {"requires_concrete_example": False, "requires_metric": False},
        },
        {
            "id": "q2",
            "question_type": "technical",
            "text": f"Walk me through a technical problem you solved recently that's relevant to the {role} role.",
            "anchor": "Fallback technical opener",
            "follow_up_seeds": [
                "What alternatives did you consider, and why did you go with this approach?",
                "How did you measure whether it was successful?",
            ],
            "depth_gate": {"requires_concrete_example": True, "requires_metric": False},
        },
        {
            "id": "q3",
            "question_type": "technical",
            "text": "Which tools or technologies in your stack do you feel strongest in, and why?",
            "anchor": "Fallback skills question",
            "follow_up_seeds": [
                "Can you give me a specific example of where that strength made a real difference?",
                "What's something in that area you're still actively learning?",
            ],
            "depth_gate": {"requires_concrete_example": True, "requires_metric": False},
        },
        {
            "id": "q4",
            "question_type": "behavioural",
            "text": "Tell me about a time you had to deliver something under a tight deadline. How did you handle it?",
            "anchor": "Fallback pressure question",
            "follow_up_seeds": [
                "What trade-offs did you make, and how did you decide what to cut?",
                "What would you do differently if you faced the same situation again?",
            ],
            "depth_gate": {"requires_concrete_example": True, "requires_metric": False},
        },
        {
            "id": "q5",
            "question_type": "motivational",
            "text": f"Where do you see yourself in the next few years, and how does the {role} role fit into that?",
            "anchor": "Fallback motivational closer",
            "follow_up_seeds": [
                "What kind of problems do you most want to be working on?",
                "What does growth look like for you in the next role?",
            ],
            "depth_gate": {"requires_concrete_example": False, "requires_metric": False},
        },
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_question_bank(
    role: str,
    candidate_name: str,
    candidate_skills: list[str],
    candidate_experience: list[dict],
    jd_skills: list[str],
) -> list[dict[str, Any]]:
    """
    Generate a personalised, structured question bank for the candidate.

    Returns a list of question dicts, each containing:
        id, question_type, text, anchor, follow_up_seeds, depth_gate

    Falls back to a generic bank if LLM call fails.
    """
    skill_gaps = _compute_skill_gaps(candidate_skills, jd_skills)
    total_questions = sum(QUESTION_COUNTS.values())

    prompt = GENERATION_PROMPT.format(
        role=role,
        candidate_name=candidate_name,
        candidate_skills=format_skills(candidate_skills),
        candidate_experience=format_experience(candidate_experience),
        jd_skills=format_skills(jd_skills),
        skill_gaps=format_skills(skill_gaps) if skill_gaps else "None — strong skill match",
        total_questions=total_questions,
        behavioural_count=QUESTION_COUNTS["behavioural"],
        technical_count=QUESTION_COUNTS["technical"],
        motivational_count=QUESTION_COUNTS["motivational"],
    )

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, _call_ollama, prompt)
        bank = _parse_bank(raw)
        if not bank:
            raise ValueError("Empty question bank returned")
        logger.info(
            f"Generated {len(bank)} questions for {candidate_name} "
            f"({role}): {[q['question_type'] for q in bank]}"
        )
        return bank
    except Exception as exc:
        logger.error(f"Question generation failed: {exc}")
        return _fallback_bank(role)