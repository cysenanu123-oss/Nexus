#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  NEXUS — one-shot installer (Linux / macOS)
#
#  Usage:
#    ./install.sh              # interactive: asks about heavy/optional deps
#    ./install.sh --full       # install everything (voice + vision + HUD + ML)
#    ./install.sh --minimal    # core only (brain + CLI), skip heavy deps
#    ./install.sh --yes        # assume "yes" to prompts
#
#  Safe to re-run: it skips what's already installed.
# ─────────────────────────────────────────────────────────────

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/nexus_venv"
PYTHON="${PYTHON:-python3}"

MODE="interactive"; ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --full) MODE="full" ;;
    --minimal) MODE="minimal" ;;
    --yes|-y) ASSUME_YES=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

c_blue="\033[1;34m"; c_grn="\033[1;32m"; c_yel="\033[1;33m"; c_red="\033[1;31m"; c_off="\033[0m"
say()  { echo -e "${c_blue}▸${c_off} $*"; }
ok()   { echo -e "${c_grn}✓${c_off} $*"; }
warn() { echo -e "${c_yel}!${c_off} $*"; }
err()  { echo -e "${c_red}✗${c_off} $*"; }

ask() {   # ask "question"  → returns 0 for yes
  [ "$ASSUME_YES" = "1" ] && return 0
  [ "$MODE" = "full" ] && return 0
  [ "$MODE" = "minimal" ] && return 1
  read -r -p "  $1 [y/N] " ans
  [[ "$ans" =~ ^[Yy] ]]
}

# ── 0. Detect platform + package manager ─────────────────────
OS="$(uname -s)"
PKG=""
if [ "$OS" = "Darwin" ]; then
  PKG="brew"
elif command -v apt-get >/dev/null 2>&1; then PKG="apt"
elif command -v dnf     >/dev/null 2>&1; then PKG="dnf"
elif command -v pacman  >/dev/null 2>&1; then PKG="pacman"
fi
say "Platform: $OS   Package manager: ${PKG:-none detected}"

sys_install() {   # sys_install pkg1 pkg2 ...
  [ -z "$PKG" ] && { warn "No package manager — install manually: $*"; return; }
  case "$PKG" in
    apt)    sudo apt-get install -y "$@" ;;
    dnf)    sudo dnf install -y "$@" ;;
    pacman) sudo pacman -S --noconfirm "$@" ;;
    brew)   brew install "$@" ;;
  esac
}

# ── 1. System libraries (audio / TTS / OCR / ffmpeg) ─────────
if [ "$MODE" != "minimal" ] && ask "Install system libs (portaudio, espeak, ffmpeg, tesseract)?"; then
  case "$PKG" in
    apt)    sudo apt-get update -qq
            sys_install libportaudio2 portaudio19-dev ffmpeg espeak-ng tesseract-ocr ;;
    dnf)    sys_install portaudio portaudio-devel ffmpeg espeak-ng tesseract ;;
    pacman) sys_install portaudio ffmpeg espeak-ng tesseract ;;
    brew)   sys_install portaudio ffmpeg espeak tesseract ;;
    *)      warn "Install manually: portaudio, ffmpeg, espeak, tesseract" ;;
  esac
  ok "System libraries step done."
fi

# ── 2. Python virtualenv ─────────────────────────────────────
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  err "python3 not found. Install Python 3.10+ first."; exit 1
fi
if [ ! -d "$VENV" ]; then
  say "Creating virtualenv at $VENV"
  "$PYTHON" -m venv "$VENV" || { err "venv creation failed"; exit 1; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
ok "Virtualenv active: $(python --version 2>&1)"

python -m pip install --upgrade pip -q

# ── 3. Core Python dependencies ──────────────────────────────
say "Installing core requirements…"
if pip install -r "$ROOT/requirements.txt" -q; then
  ok "Core requirements installed."
else
  warn "Some core requirements failed — NEXUS will run with reduced features."
fi

# ── 4. Optional heavy deps ───────────────────────────────────
if ask "Install the Jarvis HUD (PyQt5)?"; then
  pip install -q PyQt5 && ok "PyQt5 installed (HUD available)."
fi
if ask "Install vision/perception (opencv + CLIP + torch — large)?"; then
  pip install -q opencv-python open_clip_torch torch && ok "Vision stack installed."
fi
if ask "Install speaker-ID (speechbrain + torchaudio — large)?"; then
  pip install -q speechbrain torchaudio && ok "Speaker-ID installed."
fi

# ── 5. Ollama + a device-appropriate model ───────────────────
if command -v ollama >/dev/null 2>&1; then
  ok "Ollama found."
  say "Sizing up this machine…"
  python -m core.hardware || true
  if ask "Download a recommended local model now (via Ollama)?"; then
    REC="$(python - <<'PY'
try:
    from core.model_manager import get_model_manager
    m = get_model_manager().recommend("chat")
    print(m.name if m else "qwen2.5:1.5b")
except Exception:
    print("qwen2.5:1.5b")
PY
)"
    say "Pulling $REC …"
    ollama pull "$REC" && ok "Model $REC ready."
  fi
else
  warn "Ollama not found — local models won't run. Install from https://ollama.com/download"
fi

# ── 6. Done ──────────────────────────────────────────────────
echo
ok "NEXUS install complete."
echo -e "  Run the assistant:"
echo -e "    ${c_grn}source nexus_venv/bin/activate${c_off}"
echo -e "    ${c_grn}python run.py${c_off}          # Jarvis HUD if available, else CLI"
echo -e "    ${c_grn}python run.py --cli${c_off}    # terminal interface"
echo -e "    ${c_grn}python run.py --hud${c_off}    # force the HUD"
echo
