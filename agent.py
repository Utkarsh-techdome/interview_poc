# =============================================================================
# agent.py -- Interview Session core (shared by CLI and FastAPI)
# =============================================================================

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
import threading
import time
import random
from pathlib import Path
from typing import Optional
from interview_state import InterviewStateTracker

# -- Force UTF-8 on Windows console so log lines with non-ASCII never crash --
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

from deepgram import DeepgramClient
from deepgram.core.events import EventType

from constants import SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, CHUNK_FRAMES
from prompts import build_system_prompt, DEFAULT_CONFIG, FAREWELL_PHRASES, GREETING_TEMPLATES
from utils import save_wav, save_report

# ---------------------------------------------------------------------------
# Core Session Manager
# ---------------------------------------------------------------------------

class InterviewSession:
    """
    Manages one interview session. Supports two modes:
      * run_async(websocket)  ->  FastAPI / browser (no server microphone needed)
      * run()                 ->  CLI with local PyAudio microphone
    """

    def __init__(self, session_id: str, cfg: dict):
        self.session_id  = session_id
        self.cfg         = {**DEFAULT_CONFIG, **cfg}
        self.status      = "created"
        self.created_at  = datetime.datetime.now()
        self.end_time: Optional[datetime.datetime] = None

        self.conversation   : list[dict] = []
        self.audio_buffer   = bytearray()
        self.response_count = 0
        

        # Threading / async primitives
        self._stop_event     = threading.Event()
        self._interview_done = threading.Event()

        # Set by run_async -- must be accessed only from that coroutine's loop
        self._async_done : Optional[asyncio.Event]                  = None
        self._loop       : Optional[asyncio.AbstractEventLoop]      = None
        self._ws                                                    = None
        self._dg_connection                                          = None

        # Per-session output directory
        self._data_dir = Path(f"data/{session_id}")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "responses").mkdir(exist_ok=True)

        self._log_path = self._data_dir / "interview_log.txt"
        self._log_path.write_text("", encoding="utf-8")

    def _log(self, text: str) -> None:
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        print(line)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _push_event(self, payload: dict) -> None:
        if self._ws is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._ws.send_json(payload), self._loop
        )

    def _push_audio(self, data: bytes) -> None:
        if self._ws is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._ws.send_bytes(data), self._loop
        )

    def _finish(self):
        if self.status == "ended":
            return
        self.status   = "ended"
        self.end_time = datetime.datetime.now()
        self._stop_event.set()

        report_path = str(self._data_dir / "interview_report.md")
        # Generate report content
        duration = self.end_time - self.created_at
        mins, secs = divmod(int(duration.total_seconds()), 60)
        report_lines = [
            "# Interview Report", "",
            f"**Role:** {self.cfg['role']}",
            f"**Candidate:** {self.cfg['candidate_name']}",
            f"**Date:** {self.created_at.strftime('%Y-%m-%d')}",
            f"**Duration:** {mins}m {secs}s",
            "", "---", "", "## Conversation Transcript", "",
        ]
        for entry in self.conversation:
            label = "**Interviewer**" if entry["role"] == "agent" else "**Candidate**"
            report_lines.append(f"{label}: {entry['text']}")
            report_lines.append("")
        
        save_report(self.session_id, "\n".join(report_lines))
        self._log(f"Report saved for session {self.session_id}")

        self._push_event({
            "type": "interview_complete",
            "turns": len(self.conversation),
            "responses": self.response_count,
            "report_url": f"/sessions/{self.session_id}/report",
        })

        self._interview_done.set()
        if self._async_done is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._async_done.set)

    async def run_async(self, websocket: str, dg_api_key: str):
        """Native async mode for FastAPI WebSockets."""
        from websockets.client import connect as ws_connect

        self._ws = websocket
        self._loop = asyncio.get_running_loop()
        self._async_done = asyncio.Event()

        dg_url = "wss://agent.deepgram.com/v1/agent/converse"
        headers = {"Authorization": f"Token {dg_api_key}"}

        # Select a dynamic greeting
        greeting_tmpl = random.choice(GREETING_TEMPLATES)
        greeting = greeting_tmpl.format(
            name=self.cfg.get('candidate_name', 'Candidate'),
            role=self.cfg.get('role', 'Software Engineer')
        )

        settings = {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
                "output": {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
            },
            "agent": {
                "think": {
                    "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
                    "prompt": build_system_prompt(self.cfg),
                },
                "speak": {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
                "greeting": greeting,
            },
        }

        try:
            async with ws_connect(dg_url, extra_headers=headers) as dg_ws:
                self._log("[OK] Deepgram WebSocket opened")
                await dg_ws.send(json.dumps(settings))
                
                # Proactively mute mic
                try:
                    await websocket.send_text(json.dumps({"ctrl": "agent_speaking"}))
                except Exception: pass
                
                self._push_event({"type": "connected"})

                # Task A: DG -> browser
                async def _dg_to_browser():
                    _agent_speaking = False
                    try:
                        async for raw in dg_ws:
                            if isinstance(raw, bytes):
                                if not _agent_speaking:
                                    _agent_speaking = True
                                    try: await websocket.send_text(json.dumps({"ctrl": "agent_speaking"}))
                                    except Exception: pass
                                
                                self.audio_buffer.extend(raw)
                                await websocket.send_bytes(raw)
                            else:
                                msg = json.loads(raw)
                                evt_type = msg.get("type")
                                if evt_type == "AgentAudioDone":
                                    _agent_speaking = False
                                    if self.audio_buffer:
                                        wav_path = self._data_dir / "responses" / f"response-{self.response_count}.wav"
                                        save_wav(str(wav_path), self.audio_buffer, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH)
                                        self.audio_buffer = bytearray()
                                        self.response_count += 1
                                    try: await websocket.send_text(json.dumps({"ctrl": "agent_done"}))
                                    except Exception: pass
                                
                                await self._handle_dg_event(evt_type, msg)
                    except Exception as e:
                        self._log(f"DG->Browser task error: {e}")

                # Task B: Browser -> DG
                async def _browser_to_dg():
                    try:
                        while not self._stop_event.is_set():
                            raw = await websocket.receive()
                            if "bytes" in raw:
                                await dg_ws.send(raw["bytes"])
                            elif "text" in raw:
                                # handle client-side end signal
                                if "end_interview" in raw["text"]:
                                    self._log("Client requested end")
                                    self._finish()
                                    break
                    except Exception as e:
                        self._log(f"Browser->DG task error: {e}")

                await asyncio.gather(_dg_to_browser(), _browser_to_dg())

        except Exception as e:
            self._log(f"Session failed: {e}")
            self._push_event({"type": "error", "message": str(e)})
        finally:
            self._finish()

    async def _handle_dg_event(self, evt_type: str, evt: dict) -> None:
        if evt_type == "Welcome":
            self._push_event({"type": "welcome"})
        elif evt_type == "SettingsApplied":
            self.status = "active"; self._push_event({"type": "settings_applied"})
        elif evt_type == "UserStartedSpeaking":
            self._push_event({"type": "user_started_speaking"})
        elif evt_type == "AgentThinking":
            self._push_event({"type": "agent_thinking"})
        elif evt_type == "AgentStartedSpeaking":
            self.audio_buffer = bytearray(); self._push_event({"type": "agent_started_speaking"})
        elif evt_type == "ConversationText":
            role = evt.get("role", "unknown")
            text = evt.get("content", str(evt))
            label = "agent" if role == "assistant" else "user"
            self.conversation.append({"role": label, "text": text, "ts": datetime.datetime.now().isoformat()})
            self._push_event({"type": "conversation_text", "role": label, "text": text})
            if label == "agent" and any(p in text.lower() for p in FAREWELL_PHRASES):
                threading.Timer(4.0, self._finish).start()
        elif evt_type in ("Error", "Warning"):
            self._push_event({"type": evt_type.lower(), "message": str(evt)})

    def run(self):
        """Legacy CLI mode using PyAudio can be re-implemented here if needed."""
        self._log("CLI mode not implemented in this version. Use main.py.")