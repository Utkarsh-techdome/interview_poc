# =============================================================================
# utils.py — Common helper functions
# =============================================================================

import os
import json
import wave
import logging
import pdfplumber
import io
from pathlib import Path

logger = logging.getLogger("interview_agent.utils")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts raw text from a PDF file (bytes)."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {e}")
    return text.strip()

def save_wav(filename: str, audio_data: bytes, sample_rate: int, channels: int, sample_width: int):
    """Saves raw PCM data to a WAV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

def save_report(session_id: str, report_content: str, data_dir: str = "data") -> str:
    """Saves the interview report to a markdown file."""
    report_path = Path(data_dir) / session_id / "interview_report.md"
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return str(report_path)

def safe_json_parse(text: str) -> dict:
    """Tries to extract and parse JSON from messy LLM output."""
    text = text.strip()
    # Basic cleaning if LLM included markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```json"):
            text = "\n".join(lines[1:-1])
        elif lines[0].startswith("```"):
            text = "\n".join(lines[1:-1])
    
    # Try to find { ... } if there's surrounding text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def format_skills(skills: list[str]) -> str:
    """Formats a list of skills into a comma-separated string."""
    if not skills:
        return "None listed"
    return ", ".join(skills)

def format_experience(experience: list[dict]) -> str:
    """Formats a list of experience dicts into a readable string."""
    if not experience:
        return "No experience listed"
    lines = []
    for e in experience:
        role = e.get("role", "Unknown role")
        desc = e.get("description", "")
        years = e.get("years", 0)
        lines.append(f"- {role} ({years}y): {desc}")
    return "\n".join(lines)
