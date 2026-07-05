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

# HARDENING & RELIABILITY STATUS

> **Honesty rule:** `[DONE]` in the roadmap below means *the code exists and
> imports*. It does **not** by itself mean the feature is verified end-to-end.
> A feature is only "reliable" once it has a test in `tests/`. Prefer depth
> (a few subsystems that are hardened and tested) over breadth (many `[DONE]`
> boxes that have never been exercised). **Do not mark a capability reliable
> without a test proving it.**

### Verified / hardened

* **Shell execution safety** — every shell call routes through
  `core/shell_safety.py`, which refuses destructive commands (`rm -rf /`,
  `mkfs`, `dd`, fork bombs, `curl | sh`, …) and blocks command-injection via
  shell operators. Covered by `tests/test_shell_safety.py`. This replaced
  several `subprocess.run(..., shell=True)` call sites that executed
  LLM-generated commands unchecked.
* **Command routing** — the exact-match command groups in `Brain._route` are
  now a testable registry (`core/command_router.py`), covered by
  `tests/test_command_router.py`.
* **Subsystem health** — `Brain.status_report()` reports which optional
  subsystems actually initialized instead of silently degrading to `None`.
  Surfaced by the `status` command.
* **Configuration** — Ollama host (`llm.host`, or `$NEXUS_OLLAMA_HOST`) and
  owner name (`identity.owner_name`) come from `config/settings.json` /
  `core/config.py` rather than hardcoded constants.
* **Adaptive model management** — `core/hardware.py` detects device capability
  (RAM/GPU/VRAM/accelerator) cross-platform, and `core/model_manager.py` keeps a
  catalog of models, shows which fit the device, and downloads them **only with
  the user's consent** (`models` CLI command). Covered by
  `tests/test_hardware.py` and `tests/test_model_manager.py`.
* **Tiered brain router** — `core/brain_router.py` routes each task to the
  cheapest capable brain (reflex → local → cloud), escalating on uncertainty and
  gating cloud behind explicit consent. `ask` / `router` commands, and (behind
  `llm.tiered_routing`) the main conversation path. Covered by
  `tests/test_brain_router.py`.
* **Autonomous web-research agent** — `core/web_agent.py` runs a
  search→read→refine→cite loop; `core/web_safety.py` blocks SSRF/internal-host
  fetches and gates web *actions* behind consent. `research` command. Covered by
  `tests/test_web_agent.py` and `tests/test_web_safety.py`.
* **Perception** — `vision/place_recognition.py` (enroll places by photo, then
  recognize/announce location) and `vision/scene_describer.py` (describe a frame
  with a device-appropriate VLM, moondream→LLaVA/Llama-Vision). `look` / `place`
  commands. Covered by `tests/test_place_recognition.py` and
  `tests/test_scene_describer.py`.
* **Awareness / fusion loop** — `core/fusion_loop.py` + `core/world_state.py`:
  fuses sensors into a live world-state and speaks up proactively (rate-limited).
  `awareness` command. Covered by `tests/test_fusion_loop.py`.
* **Prompt-engineer pre-stage** — `core/prompt_engineer.py`: a router pre-stage
  that builds a domain-appropriate system prompt (role/constraints/format) and,
  opt-in, rewrites the user's prompt for the target model — always preserving the
  original, gated to complex tasks. Behind `llm.prompt_engineering` /
  `llm.prompt_rewrite`. Covered by `tests/test_prompt_engineer.py`.

### Known limitations / next hardening steps

* `Brain.__init__` still eagerly constructs many subsystems; migrate the
  heavy ones to lazy initialization.
* `Brain._route` still contains prefix/predicate branches that have not been
  migrated to the router yet.
* Broad `except Exception` handlers remain widespread; narrow them so failures
  are visible, not swallowed.
* Personal data (`data/*.db`, `data/*.jsonl`) is now untracked, but earlier
  commits still contain it — rewrite history with `git filter-repo` before
  making the repo public.

### Running the tests

```bash
python -m pytest tests/ -q
```

---

# ADAPTIVE INTELLIGENCE ROADMAP ("towards Jarvis")

The direction: Nexus becomes a **broker of intelligence** — it knows which
brains are available (tiny local → bigger local → cloud → other AIs + the open
web) and routes each task to the cheapest one that can handle it, growing more
capable as hardware and network allow.

Decisions locked:
* **Cloud AIs:** hybrid — official APIs (Anthropic/OpenAI/Google) primary,
  real-browser automation as a fallback for services without an API.
* **Local backend:** support both — Ollama as the default, GGUF/llama.cpp as an
  advanced option.
* **Downloads:** always consent-gated; never pull a model without asking.

Build order:
1. `[DONE]` **Capability detection + model manager** — `core/hardware.py`,
   `core/model_manager.py`, `models` CLI command, tests. Foundation for the rest.
2. `[DONE]` **Tiered brain router** — `core/brain_router.py`: reflex (local tiny)
   → local reasoning → cloud, escalating only when the local model is uncertain.
   Cloud is consent-gated (`llm.allow_cloud` off by default; per-call confirm).
   `ask <question>` and `router` CLI commands; `tests/test_brain_router.py`.
3. `[DONE]` **Autonomous web-research agent** — `core/web_agent.py`:
   search → filter → read → summarize → assess → refine loop over the
   `research/` modules. `core/web_safety.py` gates it: reads freely but blocks
   internal/metadata hosts (SSRF) and non-web schemes, and requires consent for
   any state-changing action (mirrors `core/shell_safety.py`). `research
   <question>` CLI command; `tests/test_web_agent.py`, `tests/test_web_safety.py`.
4. `[DONE]` **Perception** — `vision/place_recognition.py` (CLIP-embedding place
   recognition, same enroll/verify pattern as `voice/speaker_id.py`) and
   `vision/scene_describer.py` (scene description via a vision-language model that
   scales from moondream on a light laptop up to LLaVA/Llama-Vision on a GPU,
   chosen by the model manager and downloaded on consent). `look`, `place` CLI
   commands; `tests/test_place_recognition.py`, `tests/test_scene_describer.py`.
5. `[DONE]` **Always-on fusion loop + proactivity** — `core/world_state.py` +
   `core/fusion_loop.py`: a background service that polls sensors (place, scene,
   …) on their own cadences into a live `WorldState`, with rate-limited,
   de-duplicated proactive triggers ("you've moved from the office to the
   kitchen"). `awareness` CLI command (built but off by default). Covered by
   `tests/test_fusion_loop.py`.

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

* Linux / WSL2
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
* pyautogui (GUI automation)

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
* [DONE] Create modular plugin architecture (`core/plugins.py` — auto-loading dynamic plugin manager)

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
* [DONE] Add fast-path input classifier (`core/normalizer.py` + `core/entry_model.py` — spell correction + intent triage before brain)
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
* [DONE] Streamed TTS output (`voice/engine.py` — word-by-word voice synthesis with overlap prevention)

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

* [DONE] Design model routing architecture (`core/brain.py` — central orchestrator)
* [DONE] Build fast-path input classifier (`core/entry_model.py` — rule + keyword triage; bypasses LLM for obvious commands)
* [DONE] Build input normalizer (`core/normalizer.py` — spell correction + lowercasing + punctuation cleanup before routing)
* [NOT STARTED] Add fallback model system (graceful degradation when Ollama is offline is partially handled)
* [NOT STARTED] Add multi-model collaboration
* [NOT STARTED] Add reasoning pipeline
* [DONE] Add context memory management (`core/memory.py` — SQLite-backed conversation and fact storage)

---

# 7. LOCAL LLM INTEGRATION

## Goal

Run powerful language models locally.

### Tasks

* [DONE] Research lightweight local LLMs (Ollama + Mistral selected)
* [DONE] Setup local inference environment (Ollama server via `core/conversation.py`)
* [NOT STARTED] Integrate GGUF models directly
* [DONE] Add streaming responses (`core/stream_output.py` — token-by-token terminal streaming with color support)
* [DONE] Add local memory storage (`core/knowledge.py` — knowledge base with ChromaDB-style vector search)
* [NOT STARTED] Optimize GPU/CPU usage

---

# 8. SELF-LEARNING & RESEARCH SYSTEM

## Goal

Allow NEXUS to improve over time.

### Tasks

* [DONE] Create research collection system (`research/searcher.py` and `research/researcher.py`)
* [DONE] Add document ingestion pipeline (`research/fetcher.py`)
* [DONE] Build vector memory database (`research/memory.py` using ChromaDB)
* [DONE] Add knowledge indexing (`core/knowledge.py`)
* [DONE] Add autonomous summarization (`research/summarizer.py` using local LLMs)
* [DONE] Add learning feedback loop (`core/task_planner.py` — learns procedures to `data/task_procedures.jsonl`)
* [NOT STARTED] Add long-term memory architecture

---

# 9. CYBERSECURITY MODULE

## Goal

Turn NEXUS into a cybersecurity assistant (authorized pentesting / CTF / educational use only).

### Tasks

* [DONE] Add network monitoring tools (`cyber/network.py`)
* [DONE] Add log analysis system (`cyber/analyzer.py`)
* [DONE] Add suspicious activity detection (`cyber/analyzer.py`)
* [DONE] Add vulnerability scanning support (`cyber/scanner.py` — nmap integration)
* [DONE] Add threat intelligence integration (`cyber/intel.py` — OSINT gathering)
* [DONE] Add reconnaissance module (`cyber/recon.py` — passive recon pipeline)
* [DONE] Add sandboxed execution environment (`cyber/sandbox.py` — safe subprocess isolation)
* [DONE] Wire cyber commands into brain (`core/brain.py` — natural language cyber interface)
* [DONE] Register 9 cyber skills in skill registry (port scan, OSINT, recon, exploit search, CVE lookup, etc.)
* [NOT STARTED] Add malware behavior analysis
* [NOT STARTED] Add packet inspection tools

---

# 10. CODING ASSISTANT MODULE

## Goal

Allow NEXUS to assist with development work.

### Tasks

* [DONE] Add code understanding (`core/code_engine.py` — narrated coding + code planning system)
* [DONE] Add streaming code output (`core/stream_output.py` — live token-by-token display)
* [DONE] Add knowledge base for code context (`core/knowledge.py` — persistent local knowledge store)
* [DONE] Add terminal command assistance (brain routes shell commands via ShellAgent)
* [NOT STARTED] Add debugging assistant
* [NOT STARTED] Add project structure analysis
* [NOT STARTED] Add Git integration
* [DONE] Add code generation support (`core/code_engine.py` + `core/task_planner.py` create_skill action)
* [NOT STARTED] Add auto-documentation system

### Completed Coding Sub-Components

* [DONE] `StreamOutput` (`core/stream_output.py`) — word-by-word streaming to terminal with ANSI color; also used by voice engine for TTS overlap prevention
* [DONE] `KnowledgeBase` (`core/knowledge.py`) — persistent SQLite + in-memory knowledge store; `store()`, `recall()`, `search()`, fuzzy keyword matching
* [DONE] `CodeEngine` (`core/code_engine.py`) — narrated coding and planning system; wired into `brain._route()` and `main.py` CLI via `code` command

---

# 11. MEMORY SYSTEM

## Goal

Allow NEXUS to remember conversations and preferences.

### Tasks

* [DONE] Design memory architecture (`core/memory.py` — SQLite-backed, multiple memory types)
* [DONE] Add short-term memory (conversation history, in-session context)
* [DONE] Add long-term memory (persistent fact and preference storage)
* [DONE] Add conversation indexing (stored and searchable via `core/memory.py`)
* [DONE] Add memory search system (`memory.search()` — keyword + semantic lookup)
* [DONE] Add user preference storage (preference facts stored in memory DB)
* [DONE] Add knowledge base layer (`core/knowledge.py` — supplementary knowledge outside conversation history)

---

# 12. SECURITY & PERMISSIONS SYSTEM

## Goal

Prevent unauthorized access and unsafe actions.

### Tasks

* [DONE] Add restricted command execution (brain whitelist/blacklist; high-risk confirmation in Executor)
* [NOT STARTED] Add permission management
* [NOT STARTED] Add voice authorization
* [NOT STARTED] Add encrypted storage
* [NOT STARTED] Add activity logging
* [DONE] Add sandbox environment (`cyber/sandbox.py` — subprocess isolation for untrusted command execution)

---

# 13. SKILL SYSTEM & LOGICAL TASK PLANNER

## Goal

Allow NEXUS to figure out HOW to do arbitrary tasks, acquire new capabilities from GitHub/URLs,
and create its own skills when none exist.

### Tasks

* [DONE] Build skill registry (`core/skill_registry.py` — SQLite catalog of named capabilities)
* [DONE] Seed built-in skills (24 skills across cyber, calendar, code, memory, research, system categories)
* [DONE] Build skill acquirer (`core/skill_acquirer.py` — clone GitHub repos, parse Python files, extract & register skills)
* [DONE] Build logical task planner (`core/task_planner.py` — 5-stage planning: memory → skills → web research → LLM → heuristic)
* [DONE] Wire task planner into brain (`core/brain.py` — skill commands + task planning commands)
* [DONE] Add skill creation on demand (LLM writes Python function → saved to `data/skills/created/` → registered)
* [DONE] Add task procedure learning (successful plans saved to `data/task_procedures.jsonl` for reuse)
* [DONE] Add skill CLI commands in `main.py` (`skills list`, `skills search`, `skills acquire`, `skills create`)
* [DONE] Add `task` CLI command in `main.py` (`task <natural language goal>`)

### Completed Skill System Sub-Components

* [DONE] `SkillRegistry` (`core/skill_registry.py`)
  * SQLite database at `data/skills/registry.db`
  * `Skill` dataclass: name, description, category, source, tags, code_path, invoke_fn, parameters, usage tracking
  * Methods: `register()`, `get()`, `search()`, `list_by_category()`, `update_usage()`, `count()`
  * `get_registry()` singleton
* [DONE] `SkillAcquirer` (`core/skill_acquirer.py`)
  * `acquire(url)` — auto-detects GitHub repo vs Python file vs webpage
  * GitHub: `git clone --depth=1`, scans up to 40 `.py` files, skips tests/venv
  * AST-based function extraction (public functions with docstrings ≥15 chars)
  * LLM grouping into high-level skills (max 10 per repo)
  * Acquired repos saved to `data/skills/acquired/<repo-name>/`
  * `get_acquirer()` singleton
* [DONE] `TaskPlanner` (`core/task_planner.py`)
  * 5-stage planning pipeline: memory recall → skill search → web research → LLM plan → heuristic fallback
  * `_recall_procedure()` — 60% word-overlap match against saved procedures
  * `_has_enough_skills()` — 35% word-overlap match against skill registry
  * `_research_how_to()` — live web search + stores findings to knowledge base
  * `_llm_plan()` — JSON-structured plan with `procedure` field
  * `_create_skill()` — LLM writes Python function, saves and registers it
  * `_learn()` — appends procedure to `data/task_procedures.jsonl` + memory
  * Action types: `shell`, `memory`, `research`, `notify`, `llm_respond`, `acquire_skill`, `create_skill`, `cyber`, `calendar`, any registry skill name
  * `get_task_planner()` singleton

---

# 14. UNIFIED AUTOMATION SYSTEM

## Goal

Allow NEXUS to execute any task — physical (mouse, keyboard, apps) or logical (AI reasoning, calendar, cyber) — through a single planning and execution pipeline.

### Tasks

* [DONE] Build rule-based task planner (`automation/planner.py` — 40+ rule patterns for shell/GUI/web/wait steps)
* [DONE] Build step executor (`automation/executor.py` — dependency resolution, retry, timeout, dry-run)
* [DONE] Build shell agent (`automation/shell_agent.py` — launch/kill apps, run commands, file ops; 60+ app aliases)
* [DONE] Build GUI agent (`automation/gui_agent.py` — mouse, keyboard, window management, OCR text finding)
* [DONE] Build progress reporter (`automation/reporter.py` — real-time step progress display)
* [DONE] Build unified entry point (`automation/automation.py` — `Automation.run(instruction)` pipeline)
* [DONE] Build AI/physical unified planner (`automation/autonomous_planner.py` — merged AI reasoning + physical execution)
* [DONE] Add physical task detection (regex routing: physical triggers vs AI override patterns)
* [DONE] Add calendar / reminder system (`.ics` file creation, system notifications, `at` scheduling)
* [DONE] Add WSL2 path fallback (calendar files saved to `data/calendar/` when Desktop doesn't exist)
* [DONE] Wire automation into brain and main.py
* [DONE] Backward-compatibility shim (`core/autonomous_planner.py` re-exports from `automation/autonomous_planner.py`)

### Completed Automation Sub-Components

* [DONE] `ShellAgent` (`automation/shell_agent.py`)
  * `launch_app(name)` — 60+ app aliases (browsers, editors, terminals, security tools)
  * `run_command(cmd)` — subprocess with timeout, cwd support
  * `create_file()`, `write_file()`, `delete_file()` — safe file operations
  * `check()` — check_file_exists, check_process_running, check_python_import, check_screen_text
* [DONE] `GUIAgent` (`automation/gui_agent.py`)
  * Mouse: `click()`, `double_click()`, `right_click()`, `drag_drop()`, `smooth_move_and_click()`
  * Keyboard: `type_text()`, `press_key()`, `hotkey()`
  * Window: `focus_window()`, `wait_window()`, `maximize_window()`, `minimize_window()`, `close_window()`
  * Navigation: `navigate_url()` — opens URL in default browser
  * Screen: `screenshot()`, `find_text_on_screen()` (Tesseract OCR)
  * Backend: pyautogui + xdotool + wmctrl (graceful fallback if tools missing)
* [DONE] `Automation` (`automation/automation.py`)
  * `run(instruction, dry_run, on_progress)` — plan → execute → report
  * `get_automation()` singleton
* [DONE] `AutonomousPlanner` (`automation/autonomous_planner.py`)
  * Physical task detection via `_PHYSICAL_TRIGGERS` / `_AI_OVERRIDE` regex
  * `execute(goal)` — routes to `_run_physical()` or AI planning based on goal type
  * Extended `_run_step()` handlers: `shell`, `shell_cmd`, `gui`, `automation`, `launch_app`, `click`, `type_text`, `press_key`, `hotkey`, `scroll`, `navigate_url`, `screenshot`
  * Calendar actions: `.ics` creation, desktop notification, `at`-command scheduling
  * Cyber actions: OSINT, recon, port scan, CVE lookup, exploit search
  * Lazy-loaded agents via `_get_gui_agent()`, `_get_shell_agent()`, `_get_automation()`

---

# 15. FUTURE EXPANSION IDEAS

### Ideas

* [NOT STARTED] Smart home integration
* [NOT STARTED] Mobile companion app
* [NOT STARTED] Desktop graphical interface
* [DONE] AI-generated automation workflows
* [DONE] Autonomous task execution
* [NOT STARTED] Real-time meeting assistant
* [DONE] AI-driven scheduling assistant (calendar + reminder system via AutonomousPlanner)
* [NOT STARTED] Offline knowledge engine
* [NOT STARTED] Multi-user support
* [NOT STARTED] Personalized voice generation

---

# CURRENT DEVELOPMENT PRIORITY

## Phase 1 — Core Foundation ✅ COMPLETE

1. ~~Install Linux development environment~~ [DONE]
2. ~~Setup Python and virtual environment~~ [DONE]
3. ~~Create NEXUS project structure~~ [DONE]
4. ~~Build terminal launcher~~ [DONE]
5. ~~Create ASCII startup screen~~ [DONE]
6. ~~Setup voice input system~~ [DONE]
7. ~~Test basic command execution~~ [DONE]
8. ~~Create module architecture~~ [DONE]
9. ~~Build first wake-word prototype~~ [DONE]

## Phase 2 — Intelligence & Automation ✅ COMPLETE

1. ~~Brain orchestrator with Ollama/Mistral~~ [DONE]
2. ~~Cybersecurity module (scan, recon, OSINT, intel)~~ [DONE]
3. ~~Automation system (shell, GUI, planning, execution)~~ [DONE]
4. ~~Unified autonomous planner (AI + physical tasks)~~ [DONE]
5. ~~Streaming output + knowledge base~~ [DONE]
6. ~~Code engine (narrated coding + planning)~~ [DONE]

## Phase 3 — Self-Extension [IN PROGRESS]

1. ~~Skill registry (catalog of named capabilities)~~ [DONE]
2. ~~Skill acquirer (clone GitHub repos, learn from URLs)~~ [DONE]
3. ~~Logical task planner (5-stage planning + learning)~~ [DONE]
4. ~~Skill creation on demand (LLM writes Python functions)~~ [DONE]
5. [IN PROGRESS] Long-term memory architecture
6. [NOT STARTED] Multi-model collaboration
7. [NOT STARTED] Background daemon mode

---

# PROJECT STRUCTURE

```
NEXUS/
│
├── core/
│   ├── logger.py               # Centralized logging — color terminal + rotating file
│   ├── config.py               # Schema-validated JSON config with dot-path API
│   ├── brain.py                # Brain orchestrator — routes input to all subsystems
│   ├── intent_engine.py        # Rule-based + LLM intent classifier
│   ├── dispatcher.py           # Maps intents to system actions
│   ├── normalizer.py           # Spell correction + input normalization (fast-path)
│   ├── entry_model.py          # Fast-path intent triage before brain (keyword routing)
│   ├── planner.py              # Legacy task planning layer
│   ├── reasoning.py            # Reasoning pipeline
│   ├── memory.py               # SQLite-backed short + long-term memory manager
│   ├── conversation.py         # Conversation engine (Ollama/Mistral)
│   ├── stream_output.py        # Token-by-token streaming terminal output with color
│   ├── knowledge.py            # Persistent knowledge base (SQLite + fuzzy search)
│   ├── code_engine.py          # Narrated coding + code planning system
│   ├── skill_registry.py       # SQLite skill catalog (24+ built-in skills)
│   ├── skill_acquirer.py       # GitHub/URL skill acquisition via AST parsing
│   ├── task_planner.py         # 5-stage logical task planner with learning
│   ├── autonomous_planner.py   # Backward-compat shim → automation/autonomous_planner.py
│   └── plugins.py              # Plugin architecture and manager
│
├── automation/
│   ├── __init__.py             # Module exports
│   ├── autonomous_planner.py   # Unified AI+physical planner (calendar, cyber, GUI, shell)
│   ├── automation.py           # Unified automation entry point (plan→execute→report)
│   ├── planner.py              # Rule-based task planner (40+ patterns)
│   ├── executor.py             # Step executor (retry, timeout, dependency resolution)
│   ├── shell_agent.py          # Shell commands, app launch/kill, file ops (60+ aliases)
│   ├── gui_agent.py            # Mouse, keyboard, window management, OCR (pyautogui+xdotool)
│   └── reporter.py             # Real-time step progress display
│
├── voice/
│   ├── listener.py             # Microphone capture, VAD, phrase detection
│   ├── wakeword.py             # Custom ONNX wake word detector (hey_nexus.onnx)
│   ├── speech_to_text.py       # faster-whisper transcription
│   ├── tts.py                  # Text-to-speech (espeak backend)
│   ├── engine.py               # Voice orchestration loop
│   └── speaker_id.py           # Speaker identification (SpeechBrain ECAPA-TDNN)
│
├── cyber/
│   ├── analyzer.py             # Log analyzer & threat detection
│   ├── cyber.py                # Natural language cyber interface
│   ├── network.py              # Network intelligence (interfaces, ARP, etc.)
│   ├── scanner.py              # Port scanner (nmap integration)
│   ├── toolkit.py              # Tool management & installation
│   ├── intel.py                # Threat intelligence & OSINT gathering
│   ├── recon.py                # Passive reconnaissance pipeline
│   └── sandbox.py              # Sandboxed subprocess execution environment
│
├── research/
│   ├── researcher.py           # Autonomous research pipeline coordinator
│   ├── searcher.py             # Web search module
│   ├── fetcher.py              # Web page downloading and HTML parsing
│   ├── summarizer.py           # Page summarization using local LLM
│   └── memory.py               # Vector database (ChromaDB) for research retention
│
├── vision/
│   ├── capture.py              # Fast multi-monitor screen capture (MSS)
│   ├── ocr.py                  # Text extraction from screenshots (Tesseract)
│   └── monitor.py              # Live screen change & text detection
│
├── models/
│   ├── wakeword/
│   │   └── hey_nexus.onnx      # Custom trained wake word model (26.7 KB)
│   └── speaker_id/             # Cached SpeechBrain ECAPA-TDNN weights
│
├── data/
│   ├── wakeword/
│   │   ├── hey_nexus/          # 100 positive WAV samples
│   │   └── negative/           # 100 negative WAV samples
│   ├── speaker_profiles/
│   │   └── owner.npy           # Owner speaker embedding (192-dim)
│   ├── skills/
│   │   ├── registry.db         # SQLite skill catalog
│   │   ├── acquired/           # Cloned GitHub repos
│   │   └── created/            # LLM-generated skill functions
│   ├── calendar/               # .ics calendar event files (WSL2 fallback)
│   ├── task_procedures.jsonl   # Learned task procedures (auto-appended on success)
│   ├── nexus_memory.db         # Main SQLite memory database
│   └── training_pairs.jsonl    # LLM fine-tuning data pairs
│
├── tools/
│   ├── record_wakeword.py      # Interactive wake word sample recorder
│   └── train_wakeword.py       # CNN trainer → ONNX exporter
│
├── config/
│   └── settings.json           # Runtime configuration file
│
├── logs/
│   └── nexus.log               # Rotating log file (DEBUG level)
│
├── plugins/                    # Modular plugin directory for extensions
├── tests/
├── docs/
│
├── main.py                     # CLI launcher, banner, REPL, full command parser
├── requirements.txt
└── README.md                   # This file — master project tracker
```

---

# CLI COMMAND REFERENCE

```
NEXUS> <natural language>         # Send any message to the brain

BASIC COMMANDS
  help                            # Show this help
  status                          # System status overview
  sysinfo                         # Hardware / OS info
  modules                         # List loaded modules
  history                         # In-session command history
  clear                           # Clear terminal
  exit / quit                     # Exit NEXUS

VOICE COMMANDS
  voice                           # Start voice engine loop
  voice test                      # Microphone RMS meter test
  voice enroll                    # Enroll speaker profile
  voice verify <file>             # Verify speaker from audio file

AUTOMATION COMMANDS
  run <instruction>               # Execute physical/GUI task (Automation pipeline)
  run --dry <instruction>         # Dry-run (show plan, don't execute)
  plan <instruction>              # Show automation plan without executing
  autoplan <goal>                 # Run AutonomousPlanner on a goal

SKILL COMMANDS
  skills list                     # List all registered skills
  skills search <query>           # Search skills by keyword
  skills acquire <url>            # Acquire skills from GitHub repo or URL
  skills create <description>     # Create a new skill via LLM

TASK PLANNING COMMANDS
  task <goal>                     # Logically plan and execute a goal
  code <request>                  # Coding assistant (narrated + streamed)

MEMORY COMMANDS
  remember <fact>                 # Store a fact in memory
  recall <query>                  # Search memory
  forget <query>                  # Remove matching memory entries

CYBERSECURITY COMMANDS (authorized use only)
  cyber scan <target>             # Port scan a host
  cyber recon <target>            # Passive reconnaissance
  cyber osint <target>            # OSINT / threat intelligence lookup
  cyber analyze <log>             # Log analysis & threat detection
  cyber cve <keyword>             # CVE lookup
  cyber exploit <service>         # Exploit search
  cyber network                   # Show local network info
```

---

# FIRST SUCCESS TARGET ✅ ACHIEVED

* ~~Launch from terminal~~ ✅
* ~~Display NEXUS ASCII startup screen~~ ✅
* ~~Listen for "Hey Nexus"~~ ✅
* ~~Convert speech to text~~ ✅
* ~~Respond in terminal / TTS~~ ✅
* ~~Execute basic local commands~~ ✅
* ~~Store simple memory logs~~ ✅
* ~~Autonomous task planning and execution~~ ✅
* ~~Self-extending skill system~~ ✅
* ~~Logical planning from natural language~~ ✅

---

# DEVELOPMENT STATUS

## Current Status

* [DONE] Foundation — environment, CLI, config, logging, plugins
* [DONE] Voice pipeline — microphone, wake word, STT, TTS, speaker ID, engine
* [DONE] Custom wake word model — trained locally, live detection confirmed
* [DONE] Speaker identification — ECAPA-TDNN enrolled, wired into voice engine
* [DONE] Screen & vision — capture, OCR, live monitoring
* [DONE] Research system — web search, fetcher, summarizer, vector memory
* [DONE] Cybersecurity module — network, scanner, log analysis, OSINT, recon, sandbox
* [DONE] Brain / reasoning / memory — Ollama/Mistral, SQLite memory, knowledge base
* [DONE] Streaming output & code engine — narrated coding, streamed LLM responses
* [DONE] Input fast-path — normalizer + entry model triage before brain
* [DONE] Automation system — shell agent, GUI agent, planner, executor, reporter
* [DONE] Unified autonomous planner — AI reasoning + physical task execution merged
* [DONE] Skill system — registry (24 skills), acquirer (GitHub/URL), creator (LLM)
* [DONE] Logical task planner — 5-stage planning with web research and learning
* [DONE] Long-term memory architecture — SQLite + ChromaDB vector semantic search (`core/vector_memory.py`)
* [DONE] Conversation session manager — auto-transcripts, LLM summaries, action item extraction (`core/conversation_session.py`)
* [DONE] Action item & goal extractor — LLM + rule-based, SQLite persistence (`core/action_extractor.py`)
* [DONE] Knowledge graph — entity/relationship tracking from conversations (`core/knowledge_graph.py`)
* [DONE] Trends & focus sessions — usage analytics, peak hours, productivity tracking (`core/trends.py`)
* [DONE] Reflexion verbal RL — self-improvement from failures, LLM reflection + vector storage (`core/reflexion.py`)
* [DONE] Sleep-time compute — background memory consolidation, Human/Persona/Task blocks (`core/sleep_compute.py`)
* [DONE] Self-refine loop — generate→critique→refine with code execution feedback, CRITIC pattern (`core/self_refine.py`)
* [DONE] Supervisor/Orchestrator — parallel worker agents for complex tasks (+90% on research evals) (`core/orchestrator.py`)
* [DONE] Durable execution — checkpoint/replay for automation tasks, resume from last step (`automation/checkpoint.py`)
* [DONE] Voyager skill retrieval — semantic vector search for skills, replaces keyword-only matching (`core/skill_registry.py`)
* [DONE] Voice barge-in — frame-based DOWNSTREAM/UPSTREAM pipeline, user can interrupt TTS (`voice/engine.py`)
* [NOT STARTED] Background daemon mode / hotword wake system

---

# FINAL NOTE

NEXUS is intended to evolve continuously.

The system architecture, modules, models, and workflows are expected to expand over time.

Every contributor or future developer working on NEXUS must maintain documentation discipline and update this roadmap accordingly.
