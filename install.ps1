<#
  NEXUS — one-shot installer (Windows / PowerShell)

  Usage (from the repo folder):
    powershell -ExecutionPolicy Bypass -File .\install.ps1
    .\install.ps1 -Full        # install everything
    .\install.ps1 -Minimal     # core only
    .\install.ps1 -Yes         # assume "yes" to prompts

  Re-runnable: skips what's already present.
#>
param(
  [switch]$Full,
  [switch]$Minimal,
  [switch]$Yes
)

$ErrorActionPreference = "Continue"
$Root   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Venv   = Join-Path $Root "nexus_venv"
$Python = "python"

function Say($m)  { Write-Host "▸ $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "✓ $m" -ForegroundColor Green }
function Warn($m) { Write-Host "! $m" -ForegroundColor Yellow }
function Err($m)  { Write-Host "✗ $m" -ForegroundColor Red }

function Ask($q) {
  if ($Yes -or $Full) { return $true }
  if ($Minimal)       { return $false }
  $a = Read-Host "  $q [y/N]"
  return ($a -match '^[Yy]')
}

# ── 0. Python check ──────────────────────────────────────────
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
  Err "Python not found. Install Python 3.10+ (winget install Python.Python.3.12)."
  exit 1
}
Say "Python: $(& $Python --version)"

# ── 1. System tools (best-effort via winget) ─────────────────
if (-not $Minimal -and (Ask "Install system tools (ffmpeg, tesseract) via winget?")) {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements
    Ok "System tools step done."
  } else {
    Warn "winget not available — install ffmpeg + tesseract manually."
  }
}

# ── 2. Virtualenv ────────────────────────────────────────────
if (-not (Test-Path $Venv)) {
  Say "Creating virtualenv at $Venv"
  & $Python -m venv $Venv
}
$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate
Ok "Virtualenv active: $(python --version)"
python -m pip install --upgrade pip -q

# ── 3. Core requirements ─────────────────────────────────────
Say "Installing core requirements…"
pip install -r (Join-Path $Root "requirements.txt") -q
Ok "Core requirements installed."

# ── 4. Optional heavy deps ───────────────────────────────────
if (Ask "Install the Jarvis HUD (PyQt5)?")                          { pip install -q PyQt5;  Ok "PyQt5 installed." }
if (Ask "Install vision/perception (opencv + CLIP + torch)?")       { pip install -q opencv-python open_clip_torch torch; Ok "Vision stack installed." }
if (Ask "Install speaker-ID (speechbrain + torchaudio)?")           { pip install -q speechbrain torchaudio; Ok "Speaker-ID installed." }

# ── 5. Ollama + a device-appropriate model ───────────────────
if (Get-Command ollama -ErrorAction SilentlyContinue) {
  Ok "Ollama found."
  python -m core.hardware
  if (Ask "Download a recommended local model now?") {
    $rec = python -c "from core.model_manager import get_model_manager as g; m=g().recommend('chat'); print(m.name if m else 'qwen2.5:1.5b')"
    Say "Pulling $rec …"
    ollama pull $rec
    Ok "Model $rec ready."
  }
} else {
  Warn "Ollama not found — install from https://ollama.com/download"
}

# ── 6. Done ──────────────────────────────────────────────────
Write-Host ""
Ok "NEXUS install complete."
Write-Host "  Run the assistant:"
Write-Host "    nexus_venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "    python run.py          # Jarvis HUD if available, else CLI" -ForegroundColor Green
Write-Host "    python run.py --cli    # terminal interface" -ForegroundColor Green
Write-Host ""
