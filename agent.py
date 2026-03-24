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
import wave
from pathlib import Path
from typing import Optional

# -- Force UTF-8 on Windows console so log lines with non-ASCII never crash --
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback -- just continue

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.agent.v1.types import (
    AgentV1Settings,
    AgentV1SettingsAgent,
    AgentV1SettingsAudio,
    AgentV1SettingsAudioInput,
    AgentV1SettingsAudioOutput,
    AgentV1SettingsAgentListen,
    AgentV1SettingsAgentListenProvider_V1,
)
from deepgram.types.think_settings_v1 import ThinkSettingsV1
from deepgram.types.think_settings_v1provider import ThinkSettingsV1Provider_OpenAi
from deepgram.types.speak_settings_v1 import SpeakSettingsV1
from deepgram.types.speak_settings_v1provider import SpeakSettingsV1Provider_Deepgram

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE  = 16000   # 16 kHz -- universally supported by browsers & Deepgram STT
CHANNELS     = 1
SAMPLE_WIDTH = 2       # 16-bit PCM = 2 bytes per sample
CHUNK_FRAMES = 4096    # larger buffer = more stable streaming

DEFAULT_CONFIG: dict = {
    "role": "Software Engineer",
    "candidate_name": "Candidate",
    "questions": [
        "Can you start by telling me a little about yourself and your background?",
        "What programming languages or technologies are you most comfortable with, and why?",
        "Describe a challenging technical problem you solved recently. What was your approach?",
        "How do you handle working under pressure or tight deadlines?",
        "Where do you see yourself professionally in the next three to five years?",
        "Do you have any questions for us about the role or the team?",
    ],
}

FAREWELL_PHRASES = [
    "goodbye",
    "best of luck with",
    "interview is now complete",
    "that concludes our interview",
    "we'll be in touch",
    "this concludes",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_system_prompt(cfg: dict) -> str:
    questions_block = "\n".join(f"  Q{i+1}: {q}" for i, q in enumerate(cfg["questions"]))
    return (
        f"You are a professional, friendly, and rigorous interviewer conducting a job interview "
        f"for the role of {cfg['role']}.\n"
        f"The candidate's name is {cfg['candidate_name']}.\n\n"
        "CORE RULES:\n"
        "1. STAY IN ROLE: ONLY trigger the hiring team response if the candidate "
"explicitly asks about company culture, salary, team structure, or role specifics. "
"If the candidate gives a short social response like 'thank you' or 'glad to be here', "
"simply acknowledge briefly and re-ask the pending question. "
"NEVER trigger the hiring team response for social pleasantries.\n\n"
        "2. NO GENERIC PRAISE: Do not use filler affirmations like \"That's great!\", \"Impressive!\", "
        "or \"Fantastic!\". Instead, briefly reflect back what was said in ONE sentence that shows you "
        "processed the answer, then proceed.\n\n"
        "--- INTERVIEW FLOW CONTROL ---\n\n"
        "## Topic Management\n"
        "- Cover all questions in the list, one at a time.\n"
        "- Track `topics_covered` and `current_topic_followup_count` mentally.\n"
        "- For non-technical questions (background, career goals, pressure handling): 0-1 follow-ups.\n"
        "- For technical questions (system design, problem solving, tools): ALWAYS ask 1 follow-up, max 2.\n"
        "- Never ask for the same dimension (e.g., metrics, tools, challenges) twice within the same topic.\n\n"
        "## Metrics Fishing Rule\n"
        "- If the candidate has already provided a quantitative result, mark `metrics_collected = True` for that topic.\n"
        "- Do NOT ask for metrics again once collected for that topic.\n\n"
        "## Handling Struggling Candidates\n"
        "- Only trigger 'No worries' if the candidate explicitly says they don't know, "
        "goes silent after being asked twice, or gives two consecutive answers that are "
        "clearly off-topic or incoherent.\n"
        "- Do NOT treat short or fragmented speech as struggling — candidates often pause "
        "mid-thought. Wait for a complete response before evaluating.\n"
        "- A response split across multiple short utterances is still ONE response.\n\n"
        "## Time Awareness\n"
        "- Target 10-14 total exchanges before wrapping up.\n"
        "- Always end with \"Do you have any questions for us?\" before closing.\n\n"
        "## Depth vs Breadth Balance\n"
        "- Ask all questions first, with mandatory follow-ups on technical ones.\n"
        "- A good interview covers all topics with at least 1 follow-up on each technical answer.\n"
        "- Max 2 follow-ups per topic. Never repeat a dimension already answered.\n"
        "------------------------------\n\n"
        "## Closing Rule\n"
        "- Once you have asked 'Do you have any questions for us?' and the candidate "
        "has responded (even with 'no'), the interview is OVER.\n"
        "- Do NOT ask any more questions after this point.\n"
        "- Close with: 'That concludes our interview. Best of luck with the next steps, Alex.'\n"
        "- Never loop back to previous questions after the closing question.\n\n"
        "Interview questions to cover (ask one at a time):\n"
        f"{questions_block}\n\n"
        "Important: Keep responses conversational but concise. Follow the Flow Control rules strictly."
    )


def save_wav(audio_buffer: bytearray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(audio_buffer))


def save_report(
    cfg: dict,
    conversation: list[dict],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    report_path: str,
) -> None:
    duration = end_time - start_time
    minutes, seconds = divmod(int(duration.total_seconds()), 60)
    lines = [
        "# Interview Report", "",
        f"**Role:** {cfg['role']}",
        f"**Candidate:** {cfg['candidate_name']}",
        f"**Date:** {start_time.strftime('%Y-%m-%d')}",
        f"**Start time:** {start_time.strftime('%H:%M:%S')}",
        f"**Duration:** {minutes}m {seconds}s",
        "", "---", "", "## Conversation Transcript", "",
    ]
    for entry in conversation:
        label = "**Interviewer**" if entry["role"] == "agent" else "**Candidate**"
        lines.append(f"{label}: {entry['text']}")
        lines.append("")
    lines += ["---", "", f"*Report generated at {end_time.strftime('%Y-%m-%d %H:%M:%S')}*"]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# InterviewSession
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
        self._ws                                                     = None
        self._dg_connection                                          = None
        self._first_audio_logged                                     = False

        # Per-session output directory
        self._data_dir = Path(f"data/{session_id}")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "responses").mkdir(exist_ok=True)

        self._log_path = self._data_dir / "interview_log.txt"
        self._log_path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------

    def _log(self, text: str) -> None:
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        print(line)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------
    # Thread-safe push helpers (called from Deepgram SDK threads)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Deepgram event handlers  (called from Deepgram SDK thread)
    # ------------------------------------------------------------------

    def _on_open(self, event):
        self._log("[OK] Deepgram WebSocket opened")
        self._push_event({"type": "connected"})

    def _on_message(self, message):
        # Binary audio chunk from agent -> forward straight to browser
        if isinstance(message, bytes):
            self.audio_buffer.extend(message)
            self._push_audio(message)
            return

        msg_type = getattr(message, "type", "Unknown")
        self._log(f"[DG] event: {msg_type}")

        if msg_type == "Welcome":
            self._push_event({"type": "welcome"})

        elif msg_type == "SettingsApplied":
            self.status = "active"
            self._log("Agent ready -- interview active")
            self._push_event({"type": "settings_applied"})

        elif msg_type == "UserStartedSpeaking":
            self._log("[VAD] UserStartedSpeaking detected!")
            self._push_event({"type": "user_started_speaking"})

        elif msg_type == "AgentThinking":
            self._push_event({"type": "agent_thinking"})

        elif msg_type == "AgentStartedSpeaking":
            self.audio_buffer = bytearray()  # fresh buffer per response
            self._push_event({"type": "agent_started_speaking"})

        elif msg_type == "AgentAudioDone":
            if self.audio_buffer:
                wav_path = (
                    self._data_dir / "responses" / f"response-{self.response_count}.wav"
                )
                save_wav(self.audio_buffer, str(wav_path))
                self._log(f"Agent audio saved -> {wav_path.name}")
                self.audio_buffer = bytearray()
                self.response_count += 1
            self._push_event({"type": "agent_audio_done"})

        elif msg_type == "ConversationText":
            role = getattr(message, "role", "unknown")
            text = getattr(message, "content", str(message))
            label = "agent" if role == "assistant" else "user"
            self._log(f"{'Interviewer' if label == 'agent' else 'Candidate'}: {text}")
            self.conversation.append({
                "role": label, "text": text,
                "ts": datetime.datetime.now().isoformat(),
            })
            self._push_event({"type": "conversation_text", "role": label, "text": text})

            # Detect farewell -> schedule graceful shutdown
            if label == "agent" and any(p in text.lower() for p in FAREWELL_PHRASES):
                self._log("Farewell detected -- finishing in 4 s")
                threading.Timer(4.0, self._finish).start()

        elif msg_type in ("Error", "Warning"):
            self._log(f"{msg_type}: {message}")
            self._push_event({"type": msg_type.lower(), "message": str(message)})

        else:
            # Log any unrecognised event so nothing is silently swallowed
            self._log(f"[DG] UNKNOWN event type '{msg_type}': {repr(message)[:300]}")

    def _on_error(self, error):
        self._log(f"Deepgram error: {error}")
        self._push_event({"type": "error", "message": str(error)})

    def _on_close(self, event):
        self._log("Deepgram WebSocket closed")
        self._finish()

    # ------------------------------------------------------------------
    # Finish / teardown
    # ------------------------------------------------------------------

    def _finish(self):
        if self.status == "ended":
            return
        self.status   = "ended"
        self.end_time = datetime.datetime.now()
        self._stop_event.set()

        report_path = str(self._data_dir / "interview_report.md")
        save_report(self.cfg, self.conversation, self.created_at, self.end_time, report_path)
        self._log(f"Report saved -> {report_path}")

        self._push_event({
            "type": "interview_complete",
            "turns": len(self.conversation),
            "responses": self.response_count,
            "report_url": f"/sessions/{self.session_id}/report",
        })

        self._interview_done.set()
        # Signal the asyncio event from whatever thread we're on
        if self._async_done is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._async_done.set)

    # ------------------------------------------------------------------
    # Deepgram settings object
    # ------------------------------------------------------------------

    def _build_dg_settings_dict(self) -> dict:
        """
        Build the Deepgram settings as a plain dict so we can include fields
        that the SDK's typed model does not expose.
        Sent via connection._send() which serialises dicts to JSON directly.
        """
        return {
            "type": "Settings",
            "audio": {
                "input": {
                    "encoding": "linear16",
                    "sample_rate": SAMPLE_RATE,
                },
                "output": {
                    "encoding": "linear16",
                    "sample_rate": SAMPLE_RATE,
                    # No 'container' → defaults to 'none' (raw PCM).
                },
            },
            "agent": {
                "language": "en",
                "listen": {
                    "provider": {
                        "type": "deepgram",
                        # nova-2 is more battle-tested for browser-mic streaming
                        # than nova-3.  Swap back once VAD is confirmed working.
                        "model": "nova-2",
                    }
                },
                "think": {
                    "provider": {
                        "type": "open_ai",
                        "model": "gpt-4o-mini",
                    },
                    "prompt": build_system_prompt(self.cfg),
                },
                "speak": {
                    "provider": {
                        "type": "deepgram",
                        "model": "aura-2-thalia-en",
                    }
                },
                "greeting": (
                    f"Hello {self.cfg['candidate_name']}! "
                    f"Welcome to your interview for the {self.cfg['role']} position. "
                    "I'm your AI interviewer. Please speak clearly and take your time. "
                    "Whenever you're ready, let's begin!"
                ),
            },
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Async event dispatcher (used by run_async native-WS path)
    # ------------------------------------------------------------------

    async def _handle_dg_event(self, evt_type: str, evt: dict) -> None:
        """Dispatch a Deepgram JSON event received over the native async WS."""
        if evt_type == "Welcome":
            self._push_event({"type": "welcome"})

        elif evt_type == "SettingsApplied":
            self.status = "active"
            self._log("Agent ready -- interview active")
            self._push_event({"type": "settings_applied"})

        elif evt_type == "UserStartedSpeaking":
            self._log("[VAD] UserStartedSpeaking detected!")
            self._push_event({"type": "user_started_speaking"})

        elif evt_type == "AgentThinking":
            self._push_event({"type": "agent_thinking"})

        elif evt_type == "AgentStartedSpeaking":
            self.audio_buffer = bytearray()  # fresh buffer per response
            self._push_event({"type": "agent_started_speaking"})
            # Tell the browser to STOP sending mic audio while we speak.
            # Without this, the mic picks up our TTS from the speakers and
            # Deepgram transcribes it as candidate speech (echo loop).
            try:
                if self._ws:
                    await self._ws.send_text(json.dumps({"ctrl": "agent_speaking"}))
            except Exception:
                pass

        elif evt_type == "AgentAudioDone":
            if self.audio_buffer:
                wav_path = (
                    self._data_dir / "responses" / f"response-{self.response_count}.wav"
                )
                save_wav(self.audio_buffer, str(wav_path))
                self._log(f"Agent audio saved -> {wav_path.name}")
                self.audio_buffer = bytearray()
                self.response_count += 1
            self._push_event({"type": "agent_audio_done"})
            # Agent is done speaking — unmute mic (with grace delay in browser).
            # Also reset AgentStartedSpeaking-equivalent so the NEXT utterance
            # triggers the mute again.
            try:
                if self._ws:
                    await self._ws.send_text(json.dumps({"ctrl": "agent_done"}))
            except Exception:
                pass

        elif evt_type == "ConversationText":
            role = evt.get("role", "unknown")
            text = evt.get("content", str(evt))
            label = "agent" if role == "assistant" else "user"
            self._log(f"{'Interviewer' if label == 'agent' else 'Candidate'}: {text}")
            self.conversation.append({
                "role": label, "text": text,
                "ts": datetime.datetime.now().isoformat(),
            })
            self._push_event({"type": "conversation_text", "role": label, "text": text})
            if label == "agent" and any(p in text.lower() for p in FAREWELL_PHRASES):
                self._log("Farewell detected -- finishing in 4 s")
                threading.Timer(4.0, self._finish).start()

        elif evt_type in ("Error", "Warning"):
            self._log(f"{evt_type}: {evt}")
            self._push_event({"type": evt_type.lower(), "message": str(evt)})

        else:
            self._log(f"[DG] UNKNOWN event type '{evt_type}': {str(evt)[:200]}")

    # ------------------------------------------------------------------
    # FastAPI / async mode
    # ------------------------------------------------------------------

    async def run_async(self, websocket):
        """
        Bridge mode: browser sends raw PCM bytes over `websocket`,
        server forwards them live to Deepgram and streams agent audio back.

        Uses websockets.asyncio.client directly — no SDK sync wrapper, no threads —
        so there is zero impedance mismatch with the FastAPI asyncio event loop.
        """
        from websockets.asyncio.client import connect as ws_connect

        self._ws         = websocket
        self._loop       = asyncio.get_running_loop()
        self._async_done = asyncio.Event()

        api_key = os.getenv("DEEPGRAM_API_KEY")
        dg_url  = "wss://agent.deepgram.com/v1/agent/converse"
        headers = {"Authorization": f"Token {api_key}"}

        settings = self._build_dg_settings_dict()
        self._log(f"[SETTINGS] {json.dumps(settings, indent=2)[:600]}")

        async with ws_connect(dg_url, additional_headers=headers) as dg_ws:
            self._log("[OK] Deepgram WebSocket opened")
            await dg_ws.send(json.dumps(settings))
            self._log("Settings sent to Deepgram")
            self._push_event({"type": "connected"})

            # ── Task A: Deepgram → browser (events + TTS audio) ───────────────
            async def _dg_to_browser():
                _agent_speaking = False   # local flag; avoids sending ctrl per-chunk
                try:
                    async for raw in dg_ws:
                        if self._async_done.is_set():
                            break
                        if isinstance(raw, bytes):
                            # First audio frame of a new agent utterance → mute mic
                            if not _agent_speaking:
                                _agent_speaking = True
                                try:
                                    await websocket.send_text(json.dumps({"ctrl": "agent_speaking"}))
                                except Exception:
                                    pass
                            # TTS audio chunk → buffer locally + forward to browser
                            self.audio_buffer.extend(raw)
                            await websocket.send_bytes(raw)
                        else:
                            try:
                                evt = json.loads(raw)
                            except Exception:
                                continue
                            evt_type = evt.get("type", "Unknown")
                            self._log(f"[DG] event: {evt_type}")
                            # Reset mute-guard so the NEXT TTS utterance triggers
                            # the mute signal again. Without this, only the first
                            # agent turn mutes the mic; all subsequent ones don't.
                            if evt_type == "AgentAudioDone":
                                _agent_speaking = False
                            await self._handle_dg_event(evt_type, evt)
                except Exception as e:
                    if not self._async_done.is_set():
                        self._log(f"[DG\u2192browser] error: {e}")
                finally:
                    self._log("Deepgram WebSocket closed")
                    self._finish()

            # ── Task B: browser → Deepgram (mic PCM + control commands) ───────
            async def _browser_to_dg():
                bytes_sent = 0
                last_ka    = self._loop.time()
                try:
                    while not self._async_done.is_set():
                        try:
                            msg = await asyncio.wait_for(
                                websocket.receive(), timeout=0.5
                            )
                        except asyncio.TimeoutError:
                            now = self._loop.time()
                            if now - last_ka >= 5.0:
                                await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                                last_ka = now
                            continue

                        msg_type = msg.get("type", "")
                        if msg_type == "websocket.disconnect":
                            self._log("Browser disconnected")
                            break

                        raw_bytes = msg.get("bytes")
                        if raw_bytes:
                            if not self._first_audio_logged:
                                self._log(f"First audio bytes ({len(raw_bytes)} B) -- forwarding to Deepgram")
                                self._first_audio_logged = True
                            await dg_ws.send(raw_bytes)   # async binary frame → DG STT
                            bytes_sent += len(raw_bytes)
                            continue

                        text_data = msg.get("text")
                        if text_data:
                            try:
                                cmd = json.loads(text_data)
                                if cmd.get("action") == "end":
                                    self._log("End command received from browser")
                                    self._finish()
                            except (json.JSONDecodeError, TypeError):
                                pass

                except Exception as e:
                    self._log(f"[browser\u2192DG] error: {e}")
                finally:
                    self._log(f"[browser\u2192DG] session total: {bytes_sent} bytes sent")
                    self._finish()

            self._log("Ready -- pumping browser audio to Deepgram")

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_dg_to_browser(),  name="dg_to_browser"),
                    asyncio.create_task(_browser_to_dg(), name="browser_to_dg"),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._dg_connection = None
        self._ws            = None


    # ------------------------------------------------------------------
    # CLI / PyAudio mode (optional, no server needed)
    # ------------------------------------------------------------------

    def run(self):
        """Standalone CLI mode -- captures from local microphone via PyAudio."""
        try:
            import pyaudio
        except ImportError:
            raise RuntimeError("pyaudio required for CLI mode: pip install pyaudio")

        api_key = os.getenv("DEEPGRAM_API_KEY")
        client  = DeepgramClient(api_key=api_key)

        with client.agent.v1.connect() as connection:
            self._dg_connection = connection

            connection.on(EventType.OPEN,    self._on_open)
            connection.on(EventType.MESSAGE, self._on_message)
            connection.on(EventType.ERROR,   self._on_error)
            connection.on(EventType.CLOSE,   self._on_close)

            connection._send(self._build_dg_settings_dict())

            threading.Thread(
                target=connection.start_listening, daemon=True
            ).start()

            def _keepalive():
                while not self._stop_event.is_set():
                    time.sleep(5)
                    try:
                        connection.keep_alive()
                    except Exception:
                        pass
            threading.Thread(target=_keepalive, daemon=True).start()

            time.sleep(1)

            def _mic():
                pa     = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_FRAMES,
                )
                self._log("Microphone open -- streaming to Deepgram")
                try:
                    while not self._stop_event.is_set():
                        data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                        connection.send_media(data)
                except Exception as e:
                    self._log(f"Mic error: {e}")
                finally:
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()
                    self._log("Microphone closed")
            threading.Thread(target=_mic, daemon=True).start()

            print(f"\n{'='*55}")
            print(f"  INTERVIEW  |  {self.cfg['role']}")
            print(f"  Candidate  |  {self.cfg['candidate_name']}")
            print(f"  Questions  |  {len(self.cfg['questions'])}")
            print(f"  Ctrl+C to abort")
            print(f"{'='*55}\n")

            try:
                self._interview_done.wait(timeout=30 * 60)
            except KeyboardInterrupt:
                self._log("Interrupted by user")
                self._finish()