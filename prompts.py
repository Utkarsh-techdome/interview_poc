# =============================================================================
# prompts.py — Centralized prompt engineering
# =============================================================================

import datetime

# ---------------------------------------------------------------------------
# Constants / Config Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "role": "Software Engineer",
    "candidate_name": "Candidate",
    "questions": [
        "Can you start by telling me a little about yourself and your background?",
        "What programming languages or technologies are you most comfortable with, and why?",
        "Describe a challenging technical problem you solved recently. What was your approach?",
        "How do you handle working under pressure or tight deadlines?",
        "Where do you see yourself professionally in the next three to five years?",
    ],
    "question_bank": [],
    "candidate_skills": [],
    "candidate_experience": [],
    "jd_skills": [],
}

GREETING_TEMPLATES = [
    "Hello {name}! Welcome to your interview for the {role} position. I'm your AI interviewer. Whenever you're ready, let's begin!",
    "Hi {name}, thanks for joining us today for the {role} interview. I'll be guiding you through a few technical and behavioral questions. Shall we start?",
    "Welcome, {name}! It's great to have you here for the {role} role. I'm looking forward to learning more about your background. Let's dive in whenever you're ready."
]

FAREWELL_PHRASES = [
    "goodbye", "best of luck with", "interview is now complete",
    "that concludes our interview", "we'll be in touch", "this concludes",
]

# ---------------------------------------------------------------------------
# Resume Screening Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA = """{
  "candidate_name": "string",
  "role": "string",
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
- "candidate_name": the person's full name (if found)
- "role": the specific job title they are applying for or currently have
- "skills": technical and domain-relevant skills only, deduplicated, lowercase
- "experience": each role as a separate entry
- "description": concise 1-2 line summary of what was done
- "years": estimated duration as a number (use 0 if unknown)
- Ignore empty/null fields

Text:
{text}

JSON output:"""

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

# ---------------------------------------------------------------------------
# Question Generator Prompts
# ---------------------------------------------------------------------------

QUESTION_COUNTS = {
    "behavioural": 2,
    "technical": 3,
    "motivational": 1,
}

GENERATION_PROMPT = """\
You are an expert technical interviewer designing a structured interview question bank.

## Context

Role: {role}
Candidate name: {candidate_name}

Candidate's skills (from resume):
{candidate_skills}

Candidate's experience (from resume):
{candidate_experience}

Job description required skills:
{jd_skills}

Skill gaps (in JD but not in resume):
{skill_gaps}

## Your task

Generate exactly {total_questions} interview questions in the JSON format below.

Distribution:
- {behavioural_count} behavioural questions (background, pressure handling, collaboration)
- {technical_count} technical questions (MUST be anchored to specific skills or projects from the candidate's resume — use their actual experience as the jumping-off point, not generic questions)
- {motivational_count} motivational question (career goals, why this role)

Rules:
1. Technical questions MUST reference something specific from the candidate's resume (a project, a tool, a technology they listed). Never ask "describe a project" generically — ask about THEIR project.
2. Skill gaps: if a JD skill is missing from the resume, include ONE probing question about it (fits within technical count).
3. follow_up_seeds: 2 probing follow-ups per question. These must explore a DIFFERENT dimension than the main question. Good dimensions: scale/metrics, challenges faced, architectural decisions, team dynamics, what they'd do differently.
4. depth_gate: set requires_concrete_example=true for all technical questions. Set requires_metric=true only if the question is about impact or scale.
5. question_type: "behavioural" | "technical" | "motivational"
6. Keep question text conversational, as if spoken aloud. No bullet points inside questions.

## Output format

Return ONLY valid JSON, no markdown fences, no preamble:

{{
  "questions": [
    {{
      "id": "q1",
      "question_type": "behavioural",
      "text": "...",
      "anchor": "why this question was chosen (1 sentence)",
      "follow_up_seeds": ["...", "..."],
      "depth_gate": {{
        "requires_concrete_example": true,
        "requires_metric": false
      }}
    }}
  ]
}}"""

# ---------------------------------------------------------------------------
# Interview Agent Prompts
# ---------------------------------------------------------------------------

def build_system_prompt(cfg: dict) -> str:
    bank: list[dict] = cfg.get("question_bank", [])
    if bank:
        questions_block = _render_question_bank(bank)
        personalization_block = _render_personalization(cfg)
    else:
        questions_block = "\n".join(
            f"  Q{i+1}: {q}" for i, q in enumerate(cfg.get("questions", []))
        )
        personalization_block = ""

    return (
        f"You are a professional, friendly, and rigorous interviewer conducting a job interview "
        f"for the role of {cfg.get('role', 'the position')}.\n"
        f"The candidate's name is {cfg.get('candidate_name', 'Candidate')}.\n\n"
        + personalization_block
        + _core_rules()
        + _flow_control(bank, cfg.get('candidate_name', 'Candidate'))
        + "## Questions to cover\n\n"
        + questions_block
        + "\n\n"
        "Important: Keep responses conversational and concise. "
        "Follow the Flow Control rules strictly. "
        "Never read out the anchor or depth_gate fields to the candidate."
    )

def _render_personalization(cfg: dict) -> str:
    skills = cfg.get("candidate_skills", [])
    experience = cfg.get("candidate_experience", [])
    if not skills and not experience:
        return ""
    skill_str = ", ".join(skills[:8]) if skills else "not specified"
    exp_lines = [f"  - {e.get('role', '')}: {e.get('description', '')}" for e in experience[:3]]
    exp_str = "\n".join(exp_lines) if exp_lines else "  Not specified"
    return (
        "## Candidate background\n\n"
        f"Skills: {skill_str}\n\n"
        f"Experience:\n{exp_str}\n\n"
    )

def _render_question_bank(bank: list[dict]) -> str:
    lines = []
    for q in bank:
        qid = q.get("id", "?")
        qtype = q.get("question_type", "behavioural")
        text = q.get("text", "")
        seeds = q.get("follow_up_seeds", [])
        gate = q.get("depth_gate", {})
        gate_parts = []
        if gate.get("requires_concrete_example"): gate_parts.append("candidate must give a concrete example")
        if gate.get("requires_metric"): gate_parts.append("ask for a quantitative outcome")
        gate_str = "; ".join(gate_parts) or "no hard gate"
        seed_lines = "\n".join(f"       - {s}" for s in seeds) if seeds else "       (none)"
        lines.append(
            f"  [{qid}] ({qtype.upper()})\n"
            f"     Ask: \"{text}\"\n"
            f"     Follow-up seeds:\n{seed_lines}\n"
            f"     Depth gate: {gate_str}\n"
        )
    return "\n".join(lines)

def _core_rules() -> str:
    return (
        "## Core rules\n\n"
        "1. STAY IN ROLE.\n"
        "2. NO GENERIC PRAISE.\n"
        "3. CLARIFICATION BEFORE MOVING ON.\n"
        "4. FRAGMENTED SPEECH IS ONE ANSWER.\n"
    )

def _flow_control(bank: list[dict], candidate_name: str) -> str:
    total = len(bank)
    return (
        "## Interview flow control\n\n"
        f"Total questions: {total}.\n"
        "### Closing\n"
        f"- Close with: 'That concludes our interview. Best of luck with the next steps, {candidate_name}.'\n"
    )
