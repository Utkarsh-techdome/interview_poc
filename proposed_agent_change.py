# =============================================================================
# interview_state.py — Per-session interview state tracker
# =============================================================================
#
# Owns all mutable interview state. Called after every candidate turn.
# Produces a concise state snapshot that gets injected into GPT-4o-mini
# via Deepgram's UpdateInstructions message.
#
# Design principles:
#   - No LLM calls here. Analysis is heuristic + keyword-based.
#     Fast, deterministic, zero-latency on the hot path.
#   - One source of truth for what question we're on, how deep we've gone,
#     and what the agent should do next.
#   - The snapshot it produces is intentionally terse — prompt tokens are
#     precious inside a live voice session.
# =============================================================================

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("interview_agent.state")

# ---------------------------------------------------------------------------
# Signal detection helpers
# ---------------------------------------------------------------------------

# Phrases that indicate a concrete example was given
_EXAMPLE_SIGNALS = [
    r"\b(for example|for instance|specifically|in particular)\b",
    r"\bi (built|created|designed|developed|implemented|wrote|architected|led|owned)\b",
    r"\bwe (built|created|designed|developed|implemented)\b",
    r"\bat (my last|my previous|my current|my role)\b",
    r"\bthe (project|system|service|pipeline|model|api|app)\b",
    r"\busing (python|java|go|rust|typescript|kubernetes|postgres|redis|kafka|pytorch|tensorflow|langchain|fastapi|django|react|aws|gcp|azure)\b",
]

# Phrases that indicate a metric or quantitative outcome
_METRIC_SIGNALS = [
    r"\b\d+\s*(%|percent|ms|milliseconds|seconds|hours|days|users|requests|rps|qps|gb|tb|mb)\b",
    r"\b(reduced|improved|increased|decreased|cut|saved|achieved|scaled|handled)\b.{0,40}\b\d+\b",
    r"\b(by \d+|from \d+|to \d+|over \d+|under \d+)\b",
    r"\b(latency|throughput|uptime|accuracy|precision|recall|f1|error rate|response time)\b",
]

# Phrases that indicate the candidate is struggling / deflecting
_STRUGGLE_SIGNALS = [
    r"\b(i don'?t know|not sure|no idea|can'?t think of|haven'?t|never (done|used|worked))\b",
    r"\b(i'?m not (sure|familiar|experienced)|outside my (experience|expertise))\b",
    r"\b(pass|skip|next question|move on)\b",
]

# Phrases that indicate a clarification request is needed (ASR noise patterns)
_GARBLE_PATTERNS = [
    r"\b[A-Z]{4,}\b",              # Unexplained acronym blocks — potential ASR artifact
    r"\w{15,}",                     # Suspiciously long single token
    r"(\w+\s+){0,3}\?$",           # Very short turn ending in question mark
]

_MIN_SUBSTANTIVE_WORDS = 8          # Below this → likely incomplete utterance


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)


def _word_count(text: str) -> int:
    return len(text.split())


def _is_substantive(text: str) -> bool:
    """True if the turn looks like a real answer, not a filler or fragment."""
    return _word_count(text) >= _MIN_SUBSTANTIVE_WORDS


# ---------------------------------------------------------------------------
# Topic state
# ---------------------------------------------------------------------------

@dataclass
class TopicState:
    question_id: str
    question_type: str                    # "technical" | "behavioural" | "motivational"
    question_text: str
    follow_up_seeds: list[str]
    requires_concrete_example: bool
    requires_metric: bool

    # Mutable tracking fields
    follow_up_count: int = 0
    got_concrete_example: bool = False
    got_metric: bool = False
    struggle_count: int = 0               # consecutive struggle signals
    dimensions_covered: list[str] = field(default_factory=list)
    signal_quality: str = "unknown"       # "strong" | "adequate" | "weak" | "unknown"

    @property
    def max_follow_ups(self) -> int:
        return 2 if self.question_type == "technical" else 1

    @property
    def depth_satisfied(self) -> bool:
        """True when depth gate conditions are met — safe to advance."""
        example_ok = (not self.requires_concrete_example) or self.got_concrete_example
        metric_ok = (not self.requires_metric) or self.got_metric
        min_followup_ok = (
            self.follow_up_count >= 1
            if self.question_type == "technical"
            else True
        )
        return example_ok and metric_ok and min_followup_ok

    @property
    def exhausted(self) -> bool:
        """True when we've hit the follow-up ceiling."""
        return self.follow_up_count >= self.max_follow_ups

    def update_signal_quality(self) -> None:
        score = 0
        if self.got_concrete_example:
            score += 2
        if self.got_metric:
            score += 1
        if self.follow_up_count >= 1:
            score += 1
        self.signal_quality = (
            "strong" if score >= 3
            else "adequate" if score >= 2
            else "weak"
        )


# ---------------------------------------------------------------------------
# Next-action decisions
# ---------------------------------------------------------------------------

# What should the agent do next for this topic?
# "probe"   → ask a follow-up (use a seed)
# "advance" → move to the next question
# "close"   → ask closing question (all topics done)
# "clarify" → ask candidate to repeat / clarify

NEXT_ACTIONS = ("probe", "advance", "close", "clarify")


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class InterviewStateTracker:
    """
    Owns all mutable interview state for one session.

    Usage pattern (inside agent.py _handle_dg_event):
        state = InterviewStateTracker(question_bank)

        # On every candidate ConversationText:
        snapshot = state.process_candidate_turn(candidate_text)
        # → push snapshot via Deepgram UpdateInstructions
    """

    def __init__(self, question_bank: list[dict]):
        self.question_bank = question_bank
        self._index: int = 0                          # index into question_bank
        self._topics_done: list[str] = []
        self._closed: bool = False                    # closing question asked?

        # Current topic state — None before first question is asked
        self._current: Optional[TopicState] = None

        # Initialise first topic
        if question_bank:
            self._current = self._make_topic(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        return self._closed

    @property
    def current_question_id(self) -> Optional[str]:
        return self._current.question_id if self._current else None

    def process_candidate_turn(self, text: str) -> dict:
        """
        Analyse a candidate utterance and return a state snapshot dict
        ready to be injected into the system prompt via UpdateInstructions.

        Call this after every ConversationText where role == "user".
        """
        if self._closed or not self._current:
            return self._snapshot("close")

        topic = self._current

        # 1. Check if we need to ask for clarification first
        if self._needs_clarification(text):
            logger.debug(f"[state] Clarification needed for: {text[:60]}")
            return self._snapshot("clarify")

        # 2. Only analyse substantive turns
        if _is_substantive(text):
            self._analyse_answer(text, topic)

        # 3. Decide next action
        action = self._decide_action(text, topic)

        # 4. If advancing, roll state forward
        if action == "advance":
            self._advance_topic()
        elif action == "close":
            self._closed = True

        return self._snapshot(action)

    def mark_agent_asked_followup(self) -> None:
        """
        Call this when a ConversationText from the agent contains a follow-up
        question (as opposed to just an acknowledgement + main question re-ask).
        Increments the follow_up_count for the current topic.
        """
        if self._current:
            self._current.follow_up_count += 1
            logger.debug(
                f"[state] follow_up_count={self._current.follow_up_count} "
                f"for {self._current.question_id}"
            )

    def advance_to_next(self) -> None:
        """Force-advance to next topic (e.g. after struggle threshold)."""
        self._advance_topic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_topic(self, index: int) -> TopicState:
        q = self.question_bank[index]
        gate = q.get("depth_gate", {})
        return TopicState(
            question_id=q["id"],
            question_type=q.get("question_type", "behavioural"),
            question_text=q.get("text", ""),
            follow_up_seeds=q.get("follow_up_seeds", []),
            requires_concrete_example=gate.get("requires_concrete_example", False),
            requires_metric=gate.get("requires_metric", False),
        )

    def _analyse_answer(self, text: str, topic: TopicState) -> None:
        """Update topic state based on what the candidate said."""
        if not topic.got_concrete_example and _matches_any(text, _EXAMPLE_SIGNALS):
            topic.got_concrete_example = True
            logger.debug(f"[state] concrete example detected for {topic.question_id}")

        if not topic.got_metric and _matches_any(text, _METRIC_SIGNALS):
            topic.got_metric = True
            logger.debug(f"[state] metric detected for {topic.question_id}")

        if _matches_any(text, _STRUGGLE_SIGNALS):
            topic.struggle_count += 1
            logger.debug(
                f"[state] struggle signal #{topic.struggle_count} for {topic.question_id}"
            )

        topic.update_signal_quality()

    def _decide_action(self, text: str, topic: TopicState) -> str:
        # Candidate explicitly struggling twice → move on
        if topic.struggle_count >= 2:
            logger.debug(f"[state] struggle threshold hit — advancing")
            return "advance"

        # Hit the follow-up ceiling
        if topic.exhausted:
            return "advance"

        # Depth gate not satisfied and still have follow-up budget
        if not topic.depth_satisfied and not topic.exhausted:
            return "probe"

        # Depth satisfied — advance
        if topic.depth_satisfied:
            # Check if there are more topics
            next_index = self._index + 1
            if next_index >= len(self.question_bank):
                return "close"
            return "advance"

        return "probe"   # safe default

    def _advance_topic(self) -> None:
        if self._current:
            self._topics_done.append(self._current.question_id)
            logger.debug(f"[state] completed topic {self._current.question_id}")

        self._index += 1
        if self._index < len(self.question_bank):
            self._current = self._make_topic(self._index)
            logger.debug(f"[state] advanced to {self._current.question_id}")
        else:
            self._current = None
            logger.debug("[state] all topics exhausted")

    def _needs_clarification(self, text: str) -> bool:
        """
        True if the text looks like ASR noise / gibberish.
        Conservative — only triggers on clear signal.
        """
        if _word_count(text) <= 2:
            return False   # too short to judge — let the agent handle it
        # Flag if more than 40% of tokens look like garble
        words = text.split()
        garble_hits = sum(
            1 for w in words
            if len(w) > 12 or (w.isupper() and len(w) >= 5)
        )
        return (garble_hits / len(words)) > 0.4

    def _snapshot(self, next_action: str) -> dict:
        """
        Build the terse state dict that gets serialised into the
        UpdateInstructions prompt injection.
        """
        topic = self._current
        remaining_ids = [
            self.question_bank[i]["id"]
            for i in range(self._index + 1, len(self.question_bank))
        ] if not self._closed else []

        # Choose which follow-up seed to suggest (rotate through unused ones)
        suggested_probe: Optional[str] = None
        if topic and next_action == "probe" and topic.follow_up_seeds:
            seed_index = min(topic.follow_up_count, len(topic.follow_up_seeds) - 1)
            suggested_probe = topic.follow_up_seeds[seed_index]

        return {
            "next_action": next_action,
            "current_question_id": topic.question_id if topic else None,
            "current_question_type": topic.question_type if topic else None,
            "follow_up_count": topic.follow_up_count if topic else 0,
            "got_concrete_example": topic.got_concrete_example if topic else False,
            "got_metric": topic.got_metric if topic else False,
            "signal_quality": topic.signal_quality if topic else "unknown",
            "topics_done": list(self._topics_done),
            "topics_remaining": remaining_ids,
            "suggested_probe": suggested_probe,
            "interview_complete": self._closed,
        }