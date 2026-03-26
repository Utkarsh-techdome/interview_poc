# AI Interview Agent

A real-time, voice-first AI interviewing system that dynamically assesses candidates based on their resume and the specific job description. This system conducts structured, linear, and empathetic technical interviews by leveraging open-source LLMs for screening and question generation, and the Deepgram Voice Agent API for ultra-low latency conversational AI.

---

## 🏗️ Architecture Overview

The system architecture is broken down into two main phases: **Asynchronous Preparation** (generating the interview plan based on the resume) and **Synchronous Execution** (running the live voice conversational AI via WebSocket).

```mermaid
graph TD
    subgraph Client [Frontend UI]
        A[Browser Mic/Speaker]
    end

    subgraph Preparation [Pipeline (Ollama)]
        C[Resume Screening] -->|JSON Extraction| D[Question Generator]
        D -->|Unique Project Rule| E[(Structured Question Bank)]
    end

    subgraph Execution [FastAPI Server]
        B[main.py WebSocket Endpoint]
        F[agent.py - Deepgram Bridge]
        G[interview_state.py - Tracker]
    end

    subgraph VoiceAPI [Deepgram]
        H[STT / LLM / TTS Pipeline]
    end

    %% Preparation Flow
    C -.->|Evaluates against JD| D
    
    %% Live Flow
    A <-->|Raw Audio| B
    B <-->|ws bytes| F
    F <-->|_handle_dg_event| G
    G -.->|State-Auth UpdateThink| F
    F <-->|WS Text/Audio| H
```

### 1. Preparation Pipeline (`resume_screening.py` & `question_generator.py`)
Before the interview begins, the candidate's resume and target Job Description (JD) are fed into a local **Ollama** LLM (e.g., Llama 3.2). 
- **Screening**: Extracts skills, past roles, calculates a fit score, and evaluates the *strength* of the candidate's experience.
- **Generation**: Generates a set of 6 structured interview questions (behavioural, technical, motivational). Each question is securely anchored to a **unique** project or skill from the candidate's resume to prevent overlap. Each question includes "depth gates" (e.g., requiring a specific metric or concrete example).

### 2. Live Conversational Hub (`agent.py`)
The FastAPI server acts as a bridge between the candidate's browser (collecting and rendering raw audio) and the Deepgram Voice Agent LLM.
- **WebSocket Bridge**: The server receives raw audio bytes from the user and streams them to Deepgram in real-time.
- **Deepgram Voice Agent**: Handles the Speech-to-Text (STT), the core conversational LLM (GPT-4o / Llama 3), and Text-to-Speech (TTS) back to the user.

### 3. The Interview State Machine (`interview_state.py`)
Because LLMs natively drift or hallucinate over long conversations, the system uses a **deterministic state machine** (`interview_state.py`) that strictly controls what the Deepgram LLM thinks is the "current" active question.
- **Just-in-Time (JIT) Injection**: As the candidate answers, their transcript is evaluated by the State Tracker. The Tracker injects an `UpdateThink` message via Deepgram's WebSocket, telling the LLM its "State Authority".
- **Strict Linear Gates**: The rules are strictly linear:
  1. **ADVANCE**: If the answer is satisfactory, it provides the LLM instructions to pivot to the next question.
  2. **PROBE**: If the answer is lacking specifics, it instructs the LLM to dig deeper.
  3. **CLARIFY**: If the answer was fully garbled or heavily tokenized noise.

---

## 🛠 Features
- **Strict Linear Progression**: The LLM cannot stall or loop on old questions. It is forced to progress cleanly.
- **Resume Anchoring**: Prevents the LLM from asking generic interview questions. "Walk me through the pipeline you built at _Company X_".
- **Dynamic Probing**: If the technical question has a _depth gate_ requiring a metric ("lifted CTR by 18%"), the state machine will force the agent to probe the candidate if the candidate's answer was generic.
- **Empathetic Closures**: Uses heuristical detection to safely conclude the interview securely if the candidate signals they are unwell or unwilling to continue.

---

## 🚀 Getting Started

### Prerequisites
1. Python 3.10+
2. A local instance of [Ollama](https://ollama.com/) (running `llama3.2` or your preferred model)
3. A [Deepgram](https://deepgram.com/) API Key

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file:
   ```env
   DEEPGRAM_API_KEY="your_api_key_here"
   ```
4. Start the server
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Running the App
- Go to `http://localhost:8000/static/index.html` to access the client interface.
- Provide a JD and Candidate Resume when prompted by the API endpoints or UI flow to bootstrap your first session.
