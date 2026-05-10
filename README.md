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
* [DONE] Create GitHub repository
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
* [DONE] Add wake-word detection (`voice/wakeword.py` — custom ONNX model trained on owner's voice)
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
* [DONE] `WakeWordDetector` class (`voice/wakeword.py`)
  * [DONE] Custom "Hey Nexus" ONNX model — small CNN trained locally on 100 positive + 100 negative samples
  * [DONE] Recording tool (`tools/record_wakeword.py`) — interactive, RMS quality gating, 2-sec WAV clips
  * [DONE] Training script (`tools/train_wakeword.py`) — torchaudio mel-spectrogram CNN, scipy WAV loader, self-contained ONNX export
  * [DONE] 44100 Hz capture → 16000 Hz resample pipeline (scipy `resample_poly`)
  * [DONE] `wait_for_wake_word(listener=)` — shared mic listener support
  * [DONE] Rolling 1-second window inference with cooldown guard
  * [DONE] Confirmed detection: scores 0.869–1.000 in live testing
* [DONE] `Transcriber` class (`voice/speech_to_text.py`)
  * [DONE] `faster-whisper` backend with VAD filtering
  * [DONE] Audio normalization & confidence scoring
  * [DONE] CLI testing modes (`--file`, `--listen`, `--bench`)
* [DONE] Voice Engine loop (`voice/engine.py` — wires listener + wakeword + STT + speaker ID + TTS)

---

# 4. SPEAKER IDENTIFICATION SYSTEM

## Goal

Allow NEXUS to recognize who is speaking.

### Tasks

* [DONE] Research speaker embedding models (SpeechBrain ECAPA-TDNN selected)
* [DONE] Build voice dataset structure (`data/speaker_profiles/`)
* [DONE] Record owner voice samples (via `MicrophoneListener.capture_phrase()`)
* [DONE] Build speaker profile database (`SpeakerProfile` class — saves mean embedding as `.npy`)
* [DONE] Train speaker recognition module (ECAPA-TDNN from `speechbrain/spkrec-ecapa-voxceleb` — downloaded & cached)
* [DONE] Add confidence scoring (cosine similarity score returned with every `VerificationResult`)
* [DONE] Add unknown speaker detection (`SpeakerIdentifier.verify()` — rejects below threshold)
* [NOT STARTED] Add distance estimation using audio levels

### Completed Speaker ID Sub-Components

* [DONE] `SpeakerEmbedder` — wraps ECAPA-TDNN, extracts 192-dim L2-normalized embeddings
* [DONE] `SpeakerProfile` — saves/loads owner embedding, `is_enrolled()` check
* [DONE] `SpeakerIdentifier` — full enroll/verify flow with threshold control
* [DONE] `VerificationResult` — structured result object with `.accepted`, `.score`, `.reason`
* [DONE] Interactive enrollment CLI (`--enroll`) — records N mic samples, averages embeddings
* [DONE] File verification CLI (`--verify audio.wav`)
* [DONE] Live mic test CLI (`--test`)
* [DONE] Wired into `VoiceEngine._main_loop()` — speaker verified after capture, before transcription
* [DONE] Graceful fallback — if speechbrain missing, engine still runs with `speaker_id = None`

---

# 5. SCREEN & VISION SYSTEM

## Goal

Allow NEXUS to view and understand screen activity.

### Tasks

* [DONE] Setup screen capture system (`vision/capture.py` using `mss`)
* [DONE] Build OCR pipeline (`vision/ocr.py` using `pytesseract`)
* [DONE] Add live screen monitoring (`vision/monitor.py` for change detection)
* [NOT STARTED] Add coding-window detection
* [NOT STARTED] Add object detection system
* [NOT STARTED] Add UI understanding
* [NOT STARTED] Add screenshot memory system

### Completed Vision Sub-Components

* [DONE] `ScreenCapturer` — fast, multi-monitor screenshot capture
* [DONE] `OCREngine` — extracts text from screenshots with confidence scoring
* [DONE] Image pre-processing (contrast/sharpness/thresholding) for better OCR
* [DONE] `ScreenMonitor` — background thread detecting screen changes & specific text
* [DONE] `ScreenEvent` — fired when screen changes or text is found/lost

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
│   ├── logger.py            # Centralized logging — color terminal + rotating file
│   ├── config.py            # Schema-validated JSON config with dot-path API
│   ├── brain.py             # Brain orchestrator (memory + intent + reasoning)
│   ├── intent_engine.py     # Rule-based + LLM intent classifier
│   ├── dispatcher.py        # Maps intents to system actions
│   ├── planner.py           # Task planning layer
│   ├── reasoning.py         # Reasoning pipeline
│   ├── memory.py            # Short + long-term memory manager
│   └── conversation.py      # Conversation engine (Ollama/Mistral)
│
├── voice/
│   ├── listener.py          # Microphone capture, VAD, phrase detection
│   ├── wakeword.py          # Custom ONNX wake word detector (hey_nexus.onnx)
│   ├── speech_to_text.py    # faster-whisper transcription
│   ├── tts.py               # Text-to-speech (espeak backend)
│   ├── engine.py            # Voice orchestration loop
│   └── speaker_id.py        # Speaker identification (SpeechBrain ECAPA-TDNN)
│
├── models/
│   ├── wakeword/
│   │   └── hey_nexus.onnx       # Custom trained wake word model (26.7 KB, self-contained)
│   └── speaker_id/          # Cached SpeechBrain ECAPA-TDNN weights
│
├── data/
│   ├── wakeword/
│   │   ├── hey_nexus/           # 100 positive WAV samples
│   │   └── negative/            # 100 negative WAV samples
│   ├── speaker_profiles/
│   │   └── owner.npy            # Owner speaker embedding (192-dim)
│   └── memory.log           # Flat-file memory log
│
├── tools/
│   ├── record_wakeword.py   # Interactive wake word sample recorder
│   └── train_wakeword.py    # CNN trainer → ONNX exporter
│
├── config/
│   └── settings.json        # Runtime configuration file
│
├── logs/
│   └── nexus.log            # Rotating log file (DEBUG level)
│
├── vision/
│   ├── capture.py           # Fast multi-monitor screen capture (MSS)
│   ├── ocr.py               # Text extraction from screenshots (Tesseract)
│   └── monitor.py           # Live screen change & text detection
├── cyber/                   # [NOT STARTED]
├── memory/                  # [NOT STARTED]
├── automation/              # [NOT STARTED]
├── plugins/                 # [NOT STARTED]
├── tests/
├── docs/
│
├── main.py                  # CLI launcher, banner, REPL, command parser
├── requirements.txt
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
* [DONE] Initial coding phase (main.py, core/logger.py, core/config.py, voice/listener.py)
* [DONE] Voice pipeline implementation (microphone ✅, wake word ✅, STT ✅, TTS ✅, engine ✅)
* [DONE] Custom wake word model — trained locally, 100% val accuracy, live detection confirmed
* [DONE] Speaker identification system — ECAPA-TDNN enrolled, wired into voice engine
* [IN PROGRESS] Brain / reasoning / memory integration (Ollama/Mistral via `core/brain.py`)

---

# FINAL NOTE

NEXUS is intended to evolve continuously.

The system architecture, modules, models, and workflows are expected to expand over time.

Every contributor or future developer working on NEXUS must maintain documentation discipline and update this roadmap accordingly.
