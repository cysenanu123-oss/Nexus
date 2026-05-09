# NEXUS

### Personal General AI Assistant & Autonomous Intelligence System

> **Project Rule:**
> Every feature, module, experiment, or subsystem being worked on under NEXUS must be written in this file.
>
> * Completed tasks must be marked as `[DONE]`
> * Tasks currently being worked on must be marked as `[IN PROGRESS]`
> * Tasks not started must be marked as `[NOT STARTED]`
> * Newly discovered ideas, requirements, or subtasks must immediately be added under their appropriate section
> * No unfinished task should be marked as complete unless tested and confirmed working
> * Anyone continuing development on NEXUS should update this file before and after working on any module

---

# PROJECT VISION

NEXUS is a modular, self-extending AI assistant designed to function as:

* A coding assistant
* A cybersecurity assistant
* A voice-controlled system
* A local/offline AI
* A multi-model intelligence system
* A research assistant
* A screen-aware AI
* A terminal-based AI operating system assistant
* A self-learning and expandable AI framework

The long-term goal is to create a personalized General AI ecosystem capable of:

* Understanding voice commands
* Recognizing speakers
* Monitoring screens when permitted
* Executing tasks autonomously
* Coordinating multiple AI models together
* Learning from interactions and data
* Assisting in cybersecurity analysis and threat detection
* Running locally without heavy dependency on external APIs

---

# CORE DEVELOPMENT PHILOSOPHY

* Local-first architecture
* Modular system design
* Offline functionality whenever possible
* Expandable plugin-based structure
* Privacy-focused
* Linux-first development
* Terminal-first interaction
* Open-source integrations where useful
* Custom-built orchestration pipeline

---

# PRIMARY TECH STACK

## Main Language

* Python

## Development Environment

* Linux
* VS Code / VSCodium

## AI & ML Frameworks

* PyTorch
* Transformers
* Ollama (optional local model serving)
* ONNX Runtime
* TensorFlow (optional)

## Voice Processing

* Faster-Whisper
* Vosk
* sounddevice (PortAudio)
* SpeechBrain
* Coqui TTS

## Vision & Screen Processing

* OpenCV
* MSS
* OCR (Tesseract)

## Backend / System

* FastAPI
* SQLite / PostgreSQL
* Redis (optional)

## Interface

* Terminal UI
* Future Desktop Interface
* Future Web Dashboard

---

# MASTER ROADMAP

---

# 1. FOUNDATION SETUP

## Goal

Build the core environment and architecture of NEXUS.

### Tasks

* [DONE] Install Python environment
* [DONE] Install VS Code extensions
* [NOT STARTED] Create GitHub repository
* [DONE] Create project folder structure
* [DONE] Setup virtual environment (`nexus_env`)
* [NOT STARTED] Install PyTorch
* [DONE] Install basic dependencies (sounddevice, numpy, scipy)
* [DONE] Create startup CLI (`main.py` — REPL launcher with command parser)
* [DONE] Create terminal ASCII intro banner (animated NEXUS banner with version bar)
* [DONE] Create logging system (Python `logging` module, configurable via `core/config.py`)
* [DONE] Create configuration file system (`core/config.py` — schema-validated JSON config with dot-path API)
* [NOT STARTED] Create modular plugin architecture

---

# 2. TERMINAL INTERFACE SYSTEM

## Goal

Create the main command interface for interacting with NEXUS.

### Tasks

* [DONE] Build CLI launcher (`main.py` — full REPL loop with prompt)
* [DONE] Add animated ASCII startup screen (streaming banner with delay effect)
* [DONE] Create command parser (`parse_and_dispatch()` with dispatch table)
* [DONE] Build Natural Language Intent Engine (`core/intent_engine.py` — rule-based & LLM-ready)
* [DONE] Build Command Dispatcher (`core/dispatcher.py` — maps intents to system actions)
* [DONE] Add colorized terminal output (`Color` class + `Printer` helper with ANSI support detection)
* [DONE] Add system status display (`status`, `sysinfo`, `modules` commands)
* [DONE] Add command history (in-session `history` command)
* [NOT STARTED] Add background daemon mode
* [NOT STARTED] Add hotword wake system ("Hey Nexus")

---

# 3. VOICE RECOGNITION SYSTEM

## Goal

Allow NEXUS to hear, transcribe, and understand spoken commands.

### Tasks

* [DONE] Setup microphone access (`voice/listener.py` — sounddevice InputStream with auto-detection of native sample rate)
* [DONE] Integrate speech-to-text engine (`voice/speech_to_text.py` via faster-whisper)
* [DONE] Add text-to-speech engine (`voice/tts.py` — using espeak and sounddevice)
* [NOT STARTED] Add real-time transcription
* [DONE] Add wake-word detection (`voice/wakeword.py` — openWakeWord + fallback VAD backend)
* [DONE] Reduce background noise (Butterworth high-pass filter at 80 Hz to remove HVAC/fan rumble)
* [NOT STARTED] Add command interruption support
* [NOT STARTED] Add multilingual support

### Completed Voice Sub-Components

* [DONE] Microphone listener with blocking-read mode (`MicrophoneListener` — no callback, overflow-free)
* [DONE] Voice Activity Detection (VAD) based on RMS energy with configurable threshold
* [DONE] `capture_phrase()` — blocks until a complete spoken phrase is captured (onset → silence padding → return)
* [DONE] `stream_chunks()` — generator yielding raw audio chunks continuously
* [DONE] `test_microphone()` — live terminal RMS meter for mic level calibration
* [DONE] `list_devices()` / `print_devices()` — enumerate available audio input devices
* [DONE] Automatic hardware sample rate detection with resampling to 16 kHz (Whisper-ready)
* [IN PROGRESS] `WakeWordDetector` class (`voice/wakeword.py`)
  * [DONE] openWakeWord backend — feeds 80 ms frames to pre-trained `hey_jarvis` model (closest to "Hey Nexus")
  * [DONE] Fallback VAD backend — energy burst duration pattern, zero extra dependencies
  * [DONE] `wait_for_wake_word()` — blocking API for synchronous usage
  * [DONE] Callback API — async/event-driven usage
  * [DONE] Cooldown guard to prevent double-fire
  * [DONE] Context manager (`with WakeWordDetector() as det:`)
  * [NOT STARTED] Custom "Hey Nexus" model training (Google Colab notebook)
  * [NOT STARTED] Verifier model for speaker-specific activation
* [DONE] `Transcriber` class (`voice/speech_to_text.py`)
  * [DONE] `faster-whisper` backend with VAD filtering
  * [DONE] Audio normalization & confidence scoring
  * [DONE] CLI testing modes (`--file`, `--listen`, `--bench`)
* [DONE] Voice Engine loop (`voice/engine.py` wiring listener + wakeword + stt together)

---

# 4. SPEAKER IDENTIFICATION SYSTEM

## Goal

Allow NEXUS to recognize who is speaking.

### Tasks

* [NOT STARTED] Research speaker embedding models
* [NOT STARTED] Build voice dataset structure
* [NOT STARTED] Record owner voice samples
* [NOT STARTED] Build speaker profile database
* [NOT STARTED] Train speaker recognition module
* [NOT STARTED] Add confidence scoring
* [NOT STARTED] Add unknown speaker detection
* [NOT STARTED] Add distance estimation using audio levels

---

# 5. SCREEN & VISION SYSTEM

## Goal

Allow NEXUS to view and understand screen activity.

### Tasks

* [NOT STARTED] Setup screen capture system
* [NOT STARTED] Build OCR pipeline
* [NOT STARTED] Add live screen monitoring
* [NOT STARTED] Add coding-window detection
* [NOT STARTED] Add object detection system
* [NOT STARTED] Add UI understanding
* [NOT STARTED] Add screenshot memory system

---

# 6. MULTI-MODEL ORCHESTRATION SYSTEM

## Goal

Allow NEXUS to coordinate multiple AI models together.

### Tasks

* [NOT STARTED] Design model routing architecture
* [NOT STARTED] Build task classifier
* [NOT STARTED] Create model selection pipeline
* [NOT STARTED] Add fallback model system
* [NOT STARTED] Add multi-model collaboration
* [NOT STARTED] Add reasoning pipeline
* [NOT STARTED] Add context memory management

---

# 7. LOCAL LLM INTEGRATION

## Goal

Run powerful language models locally.

### Tasks

* [NOT STARTED] Research lightweight local LLMs
* [NOT STARTED] Setup local inference environment
* [NOT STARTED] Integrate GGUF models
* [NOT STARTED] Add streaming responses
* [NOT STARTED] Add local memory storage
* [NOT STARTED] Optimize GPU/CPU usage

---

# 8. SELF-LEARNING & RESEARCH SYSTEM

## Goal

Allow NEXUS to improve over time.

### Tasks

* [NOT STARTED] Create research collection system
* [NOT STARTED] Add document ingestion pipeline
* [NOT STARTED] Build vector memory database
* [NOT STARTED] Add knowledge indexing
* [NOT STARTED] Add autonomous summarization
* [NOT STARTED] Add learning feedback loop
* [NOT STARTED] Add long-term memory architecture

---

# 9. CYBERSECURITY MODULE

## Goal

Turn NEXUS into a cybersecurity assistant.

### Tasks

* [NOT STARTED] Add network monitoring tools
* [NOT STARTED] Add log analysis system
* [NOT STARTED] Add suspicious activity detection
* [NOT STARTED] Add malware behavior analysis
* [NOT STARTED] Add packet inspection tools
* [NOT STARTED] Add vulnerability scanning support
* [NOT STARTED] Add threat intelligence integration

---

# 10. CODING ASSISTANT MODULE

## Goal

Allow NEXUS to assist with development work.

### Tasks

* [NOT STARTED] Add code understanding
* [NOT STARTED] Add terminal command assistance
* [NOT STARTED] Add debugging assistant
* [NOT STARTED] Add project structure analysis
* [NOT STARTED] Add Git integration
* [NOT STARTED] Add code generation support
* [NOT STARTED] Add auto-documentation system

---

# 11. MEMORY SYSTEM

## Goal

Allow NEXUS to remember conversations and preferences.

### Tasks

* [NOT STARTED] Design memory architecture
* [NOT STARTED] Add short-term memory
* [NOT STARTED] Add long-term memory
* [NOT STARTED] Add conversation indexing
* [NOT STARTED] Add memory search system
* [NOT STARTED] Add user preference storage

---

# 12. SECURITY & PERMISSIONS SYSTEM

## Goal

Prevent unauthorized access and unsafe actions.

### Tasks

* [NOT STARTED] Add permission management
* [NOT STARTED] Add voice authorization
* [NOT STARTED] Add encrypted storage
* [NOT STARTED] Add activity logging
* [NOT STARTED] Add restricted command execution
* [NOT STARTED] Add sandbox environment

---

# 13. FUTURE EXPANSION IDEAS

### Ideas

* [NOT STARTED] Smart home integration
* [NOT STARTED] Mobile companion app
* [NOT STARTED] Desktop graphical interface
* [NOT STARTED] AI-generated automation workflows
* [NOT STARTED] Autonomous task execution
* [NOT STARTED] Real-time meeting assistant
* [NOT STARTED] AI-driven scheduling assistant
* [NOT STARTED] Offline knowledge engine
* [NOT STARTED] Multi-user support
* [NOT STARTED] Personalized voice generation

---

# CURRENT DEVELOPMENT PRIORITY

## Phase 1 — Core Foundation

### Immediate Tasks

1. ~~Install Linux development environment~~ [DONE]
2. ~~Setup Python and virtual environment~~ [DONE]
3. ~~Create NEXUS project structure~~ [DONE]
4. ~~Build terminal launcher~~ [DONE]
5. ~~Create ASCII startup screen~~ [DONE]
6. ~~Setup voice input system~~ [DONE]
7. ~~Test basic command execution~~ [DONE]
8. Setup GitHub repository [NOT STARTED]
9. ~~Create module architecture~~ [DONE]
10. ~~Build first wake-word prototype~~ [DONE]

---

# INITIAL PROJECT STRUCTURE

```bash
NEXUS/
│
├── core/
│   └── config.py           # Schema-validated JSON config system
├── voice/
│   └── listener.py          # Microphone capture layer with VAD
├── vision/
├── models/
├── memory/
├── cyber/
├── automation/
├── interface/
├── plugins/
├── logs/
├── data/
├── config/
│   └── settings.json        # Runtime configuration file
├── tests/
├── docs/
│
├── main.py                  # CLI launcher, banner, command parser, REPL
├── requirements.txt         # Python dependencies
├── README.md                # This file — master project tracker
└── roadmap.md
```

---

# FIRST SUCCESS TARGET

The first working version of NEXUS should be able to:

* ~~Launch from terminal~~ ✅
* ~~Display NEXUS ASCII startup screen~~ ✅
* ~~Listen for "Hey Nexus"~~ ✅
* ~~Convert speech to text~~ ✅
* ~~Respond in terminal / TTS~~ ✅
* ~~Execute basic local commands~~ ✅ (shell command support)
* ~~Store simple memory logs~~ ✅ (flat file memory implemented)

---

# DEVELOPMENT STATUS

## Current Status

* [DONE] Planning architecture
* [DONE] Development environment setup (Python venv, Linux, VS Code)
* [DONE] Initial coding phase (main.py, core/config.py, voice/listener.py)
* [DONE] Voice pipeline implementation (microphone ✅, wake word ✅, STT ✅, engine ✅)
* [NOT STARTED] Model integration

---

# FINAL NOTE

NEXUS is intended to evolve continuously.

The system architecture, modules, models, and workflows are expected to expand over time.

Every contributor or future developer working on NEXUS must maintain documentation discipline and update this roadmap accordingly.
# Nexus
