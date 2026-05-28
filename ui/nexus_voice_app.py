"""
ui/nexus_voice_app.py
NEXUS Voice Interface — native desktop application.

Cross-platform: Windows / Linux (incl. WSL2 w/ WSLg) / macOS.
Built with PyQt5 — no browser, no Electron.

Visual states:
  idle       → slow breathing orb, dim blue
  wake       → bright white flash + greeting
  listening  → fast cyan pulse + sonar rings
  processing → purple orb + spinning arc
  speaking   → green orb + waveform bars
  error      → red orb

Voice modes:
  Full mode  → wake-word ('Hey Nexus') → mic → STT → LLM → TTS
  Demo mode  → ACTIVATE button replaces wake-word (no mic/model needed)

Launch:
  python ui/nexus_voice_app.py
  python ui/nexus_voice_app.py --demo     # no mic / model needed
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal,
    QPoint, QPointF, QRectF,
)
from PyQt5.QtGui import (
    QPainter, QColor, QRadialGradient, QConicalGradient,
    QPen, QBrush, QFont, QPalette,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit,
    QFrame, QSizePolicy,
)

# Ensure project root is importable
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────
#  Constants & Palettes
# ─────────────────────────────────────────────────────────────

WIN_W, WIN_H = 460, 600
ORB_SIZE     = 240

GREETINGS = [
    "Hey boss, what can I do for you?",
    "Hey — I'm listening. Go ahead.",
    "NEXUS online. What do you need?",
    "What's up? I'm all ears.",
    "Yes boss?",
    "At your service.",
]

STATUS_TEXT = {
    "idle":       "Say  'Hey Nexus'  to wake me",
    "demo":       "Click  ACTIVATE  to start",
    "wake":       "Hey!",
    "listening":  "Listening...",
    "processing": "Processing...",
    "speaking":   "Speaking...",
    "error":      "Error — check the logs",
}

# Per-state color palettes  (orb_c=center, orb_m=mid, orb_e=edge, glow, text, ring)
PAL = {
    "idle": dict(
        orb_c="#2466a8", orb_m="#154878", orb_e="#07192b",
        glow="#0d3a6e",  text="#5a9fd4",  ring="#1a5080",
        dot="#1a5a3a",   label="#3a7ab8",
    ),
    "wake": dict(
        orb_c="#ffffff", orb_m="#b3e5fc", orb_e="#0288d1",
        glow="#4fc3f7",  text="#0288d1",  ring="#80d8ff",
        dot="#80d0ff",   label="#80d0ff",
    ),
    "listening": dict(
        orb_c="#40d4ff", orb_m="#0096c7", orb_e="#023e8a",
        glow="#0096c7",  text="#caf0f8",  ring="#00b4d8",
        dot="#00b4d8",   label="#00b4d8",
    ),
    "processing": dict(
        orb_c="#c4a8ff", orb_m="#7e57c2", orb_e="#311b92",
        glow="#7e57c2",  text="#ede7f6",  ring="#9c27b0",
        dot="#9c27b0",   label="#b39ddb",
    ),
    "speaking": dict(
        orb_c="#80ffca", orb_m="#26a69a", orb_e="#004d40",
        glow="#26a69a",  text="#e0f2f1",  ring="#4db6ac",
        dot="#26a69a",   label="#4db6ac",
    ),
    "error": dict(
        orb_c="#ff8a80", orb_m="#e53935", orb_e="#7f0000",
        glow="#e53935",  text="#ffcdd2",  ring="#ef9a9a",
        dot="#e53935",   label="#ef9a9a",
    ),
}


# ─────────────────────────────────────────────────────────────
#  Orb Widget  —  the animated centrepiece
# ─────────────────────────────────────────────────────────────

class OrbWidget(QWidget):
    """Animated orb that reflects voice state via color, pulse, and effects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(ORB_SIZE, ORB_SIZE)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._state       = "idle"
        self._phase       = 0.0
        self._arc_angle   = 0        # processing spinner
        self._sonar       : list     = []   # [[radius_delta, alpha], ...]
        self._wave_bars   = [0.4] * 32
        self._wave_phase  = 0.0
        self._wake_flash  = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(22)        # ~45 fps

    def set_state(self, state: str):
        prev = self._state
        self._state = state
        if state == "wake":
            self._wake_flash = 1.0
        if state == "listening" and prev != "listening":
            self._sonar = []

    # ── Animation tick ────────────────────────────────────────

    def _tick(self):
        s = self._state

        if s == "idle":
            self._phase += 0.022
        elif s in ("wake", "listening"):
            self._phase += 0.065
            self._wake_flash = max(0.0, self._wake_flash - 0.045)
        elif s == "processing":
            self._phase      += 0.030
            self._arc_angle   = (self._arc_angle + 7) % 360
        elif s == "speaking":
            self._phase      += 0.040
            self._wave_phase += 0.18
            self._wave_bars   = [
                0.25 + 0.75 * abs(math.sin(self._wave_phase + i * 0.42))
                for i in range(32)
            ]

        # Sonar ring lifecycle (listening only)
        if s == "listening":
            if not self._sonar or self._sonar[-1][0] > 18:
                self._sonar.append([0.0, 230])
            kept = []
            for ring in self._sonar:
                ring[0] += 1.4
                ring[1]  = max(0, ring[1] - 5)
                if ring[1] > 0:
                    kept.append(ring)
            self._sonar = kept

        self.update()

    # ── Drawing ───────────────────────────────────────────────

    def paintEvent(self, _event):
        p    = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        w, h   = self.width(), self.height()
        cx, cy = w * 0.5, h * 0.5
        base_r = min(w, h) * 0.34
        pal    = PAL.get(self._state, PAL["idle"])
        s      = math.sin(self._phase)

        # Radius & glow intensity per state
        if self._state == "idle":
            r, gi = base_r * (1.0 + 0.05 * s), 0.22 + 0.08 * s
        elif self._state == "listening":
            r, gi = base_r * (1.0 + 0.10 * abs(s)), 0.65 + 0.25 * abs(s)
        elif self._state in ("wake",):
            r, gi = base_r * (1.0 + 0.15 * abs(s)), 0.90 + 0.10 * abs(s)
        elif self._state == "processing":
            r, gi = base_r * (1.0 + 0.03 * s), 0.40 + 0.12 * s
        elif self._state == "speaking":
            r, gi = base_r * (1.0 + 0.06 * s), 0.52 + 0.16 * abs(s)
        else:
            r, gi = base_r, 0.30

        if self._wake_flash > 0:
            r  += base_r * 0.22 * self._wake_flash
            gi  = max(gi, self._wake_flash)

        # 1 · Sonar rings (behind everything)
        if self._state == "listening":
            ring_c = QColor(pal["ring"])
            for dr, alpha in self._sonar:
                rr = r + dr
                ring_c.setAlpha(int(alpha))
                pen = QPen(ring_c, 1.4)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QRectF(cx - rr, cy - rr, rr * 2, rr * 2))

        # 2 · Glow halos (4 layers, outer→inner)
        gc = QColor(pal["glow"])
        for i in range(4):
            gr    = r * (1.55 - i * 0.15)
            alpha = int(110 * gi * (0.18 + i * 0.27))
            gc.setAlpha(max(0, min(255, alpha)))
            p.setBrush(QBrush(gc))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(cx - gr, cy - gr, gr * 2, gr * 2))

        # 3 · Main orb (radial gradient)
        grad = QRadialGradient(cx, cy, r)
        grad.setColorAt(0.0, QColor(pal["orb_c"]))
        grad.setColorAt(0.55, QColor(pal["orb_m"]))
        grad.setColorAt(1.0, QColor(pal["orb_e"]))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # 4 · Wake flash
        if self._wake_flash > 0:
            fg = QRadialGradient(cx, cy, r)
            fg.setColorAt(0.0, QColor(255, 255, 255, int(200 * self._wake_flash)))
            fg.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.setBrush(QBrush(fg))
            p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # 5 · Inner highlight (specular)
        hx = cx - r * 0.18
        hy = cy - r * 0.18
        hr = r * 0.52
        hg = QRadialGradient(hx, hy, hr)
        hg.setColorAt(0.0, QColor(255, 255, 255, 52))
        hg.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(hg))
        p.drawEllipse(QRectF(hx - hr, hy - hr, hr * 2, hr * 2))

        # 6 · Processing spinner
        if self._state == "processing":
            self._draw_spinner(p, cx, cy, r, pal)

        # 7 · Speaking waveform
        if self._state == "speaking":
            self._draw_waveform(p, cx, cy, r, pal)

        # 8 · NEXUS label
        ta  = 200 if self._state != "idle" else int(130 + 65 * abs(s))
        tc  = QColor(pal["text"])
        tc.setAlpha(ta)
        p.setPen(tc)
        fs  = max(9, int(r * 0.28))
        fnt = QFont("Arial", fs, QFont.Bold)
        fnt.setLetterSpacing(QFont.AbsoluteSpacing, 3)
        p.setFont(fnt)
        p.drawText(
            QRectF(cx - r, cy - r * 0.35, r * 2, r * 0.7),
            Qt.AlignCenter, "NEXUS",
        )

    def _draw_spinner(self, p: QPainter, cx, cy, r, pal):
        ar   = r + 15
        rect = QRectF(cx - ar, cy - ar, ar * 2, ar * 2)

        # Dim track ring
        tc = QColor(pal["ring"]); tc.setAlpha(30)
        p.setPen(QPen(tc, 2.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(rect)

        # Spinning bright arc (120°)
        ac = QColor(pal["orb_c"]); ac.setAlpha(220)
        p.setPen(QPen(ac, 3, Qt.SolidLine, Qt.RoundCap))
        start = (90 - self._arc_angle) * 16
        p.drawArc(rect, start, -120 * 16)

    def _draw_waveform(self, p: QPainter, cx, cy, r, pal):
        bc = QColor(pal["orb_c"])
        n  = len(self._wave_bars)
        for i, amp in enumerate(self._wave_bars):
            angle  = 2 * math.pi * i / n - math.pi / 2
            inner  = r + 5
            outer  = inner + r * 0.12 + r * 0.22 * amp
            ix = cx + inner * math.cos(angle)
            iy = cy + inner * math.sin(angle)
            ox = cx + outer * math.cos(angle)
            oy = cy + outer * math.sin(angle)
            bc.setAlpha(int(170 * amp))
            p.setPen(QPen(bc, 2.0, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(ix, iy), QPointF(ox, oy))


# ─────────────────────────────────────────────────────────────
#  Rounded Container  —  the window background panel
# ─────────────────────────────────────────────────────────────

class RoundPanel(QWidget):
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#182840"), 1))
        p.setBrush(QColor("#05090f"))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 16, 16)


# ─────────────────────────────────────────────────────────────
#  Voice Worker  —  background thread for all voice I/O
# ─────────────────────────────────────────────────────────────

class VoiceWorker(QThread):
    state_changed = pyqtSignal(str)   # idle / wake / listening / processing / speaking / error
    nexus_spoke   = pyqtSignal(str)   # NEXUS response text
    user_spoke    = pyqtSignal(str)   # user transcript
    error_msg     = pyqtSignal(str)

    def __init__(self, demo: bool = False):
        super().__init__()
        self.demo_mode = demo
        self._stop     = False
        self._trigger  = threading.Event()

        self._detector    = None
        self._listener    = None
        self._transcriber = None
        self._llm         = None
        self._speaker     = None
        self._load()

    def _load(self):
        try:
            from voice.wakeword import WakeWordDetector
            self._detector = WakeWordDetector()
        except Exception:
            self.demo_mode = True

        try:
            from voice.listener import MicrophoneListener
            self._listener = MicrophoneListener()
        except Exception:
            pass

        try:
            from voice.speech_to_text import Transcriber
            self._transcriber = Transcriber(model_size="tiny")
        except Exception:
            pass

        try:
            from core.llm import get_llm
            llm = get_llm()
            if llm.is_ready:
                self._llm = llm
        except Exception:
            pass

        try:
            from voice.tts import Speaker
            self._speaker = Speaker()
        except Exception:
            pass

    # ── Control ────────────────────────────────────────────

    def trigger(self):
        self._trigger.set()

    def stop(self):
        self._stop = True
        self._trigger.set()

    # ── Main loop ──────────────────────────────────────────

    def run(self):
        if self._listener:
            try:
                self._listener.start()
            except Exception:
                pass

        while not self._stop:
            self.state_changed.emit("idle")

            # ── Wait for wake word ─────────────────────────
            if self.demo_mode or self._detector is None:
                self._trigger.clear()
                self._trigger.wait()
                if self._stop:
                    break
            else:
                try:
                    self._detector.wait_for_wake_word(listener=self._listener)
                except Exception as e:
                    self.error_msg.emit(str(e))
                    self.demo_mode = True
                    time.sleep(0.5)
                    continue

            if self._stop:
                break

            # ── Wake greeting ──────────────────────────────
            greeting = random.choice(GREETINGS)
            self.state_changed.emit("wake")
            self.nexus_spoke.emit(greeting)
            if self._speaker:
                self._speaker.say(greeting)
            time.sleep(0.7)

            # ── Listen ─────────────────────────────────────
            self.state_changed.emit("listening")

            if not self._listener or not self._transcriber:
                self.nexus_spoke.emit(
                    "[Mic / STT unavailable — type your command below]"
                )
                time.sleep(1.5)
                self.state_changed.emit("idle")
                continue

            audio = self._listener.capture_phrase(verbose=False)
            if audio is None or len(audio) == 0:
                self.state_changed.emit("idle")
                continue

            # ── Transcribe ─────────────────────────────────
            self.state_changed.emit("processing")
            try:
                result = self._transcriber.transcribe(audio)
                if not result.is_speech:
                    self.state_changed.emit("idle")
                    continue
                user_text = result.text.strip()
            except Exception as e:
                self.error_msg.emit(f"STT failed: {e}")
                self.state_changed.emit("idle")
                continue

            self.user_spoke.emit(user_text)

            # ── Respond ────────────────────────────────────
            response = self.respond(user_text)

            self.state_changed.emit("speaking")
            self.nexus_spoke.emit(response)
            if self._speaker:
                self._speaker.say(response, block=True)
            else:
                time.sleep(max(1.0, len(response) * 0.045))

            self.state_changed.emit("idle")

        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
        if self._speaker:
            try:
                self._speaker.shutdown()
            except Exception:
                pass

    def respond(self, text: str) -> str:
        """Generate a response — LLM if online, keyword fallback otherwise."""
        if self._llm:
            try:
                return self._llm.chat(text)
            except Exception:
                pass

        t = text.lower()
        if any(w in t for w in ("hello", "hi ", "hey")):
            return "Hey! Good to hear from you, boss."
        if "time" in t:
            from datetime import datetime
            return f"It's {datetime.now().strftime('%I:%M %p')}."
        if "how are you" in t:
            return "All systems nominal. What do you need?"
        if any(w in t for w in ("who are you", "what are you")):
            return "I'm NEXUS — your personal AI. Start Ollama for full responses."
        if "thank" in t:
            return "Anytime, boss."
        if any(w in t for w in ("stop", "quit", "bye", "exit")):
            return "Signing off. Call me when you need me."
        return "Got it. Ollama is offline right now — start it for full AI responses."


# ─────────────────────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────────────────────

class NexusVoiceApp(QMainWindow):

    def __init__(self, demo: bool = False):
        super().__init__()
        self._drag_pos: Optional[QPoint] = None

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(WIN_W, WIN_H)
        self._center_on_screen()

        self._build_ui()
        self._start_worker(demo)

    def _center_on_screen(self):
        geo = QApplication.primaryScreen().availableGeometry()
        self.move(
            (geo.width()  - WIN_W) // 2,
            (geo.height() - WIN_H) // 2,
        )

    # ── UI Construction ───────────────────────────────────────

    def _build_ui(self):
        panel = RoundPanel(self)
        panel.setGeometry(0, 0, WIN_W, WIN_H)

        root = QVBoxLayout(panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_titlebar())
        root.addWidget(self._make_orb_section(), 1)
        root.addWidget(self._make_divider())
        root.addWidget(self._make_transcript())
        root.addWidget(self._make_divider())
        root.addWidget(self._make_bottom_bar())

    def _make_titlebar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(46)
        bar.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)

        # MacOS-style traffic lights
        for color, hover, slot in (
            ("#ff5f57", "#ff3b30", self.close),
            ("#febc2e", "#f5a000", self.showMinimized),
        ):
            btn = QPushButton()
            btn.setFixedSize(13, 13)
            btn.setStyleSheet(
                f"QPushButton{{background:{color};border-radius:6px;border:none;}}"
                f"QPushButton:hover{{background:{hover};}}"
            )
            btn.clicked.connect(slot)
            lay.addWidget(btn)
            lay.addSpacing(6)

        lay.addStretch()

        title = QLabel("N E X U S")
        title.setStyleSheet(
            "color:#3a6a9a; font-size:11px; font-weight:bold;"
            "font-family:'Courier New',monospace; letter-spacing:4px;"
        )
        lay.addWidget(title)

        lay.addStretch()

        self._dot = QLabel("●")
        self._dot.setStyleSheet("color:#1a5a3a; font-size:9px;")
        lay.addWidget(self._dot)

        return bar

    def _make_orb_section(self) -> QWidget:
        sec = QWidget()
        sec.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(sec)
        lay.setContentsMargins(0, 12, 0, 8)
        lay.setSpacing(10)

        self._orb = OrbWidget()
        lay.addWidget(self._orb, 0, Qt.AlignCenter)

        self._status = QLabel(STATUS_TEXT["idle"])
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(
            "color:#3a7ab8; font-size:12px;"
            "font-family:'Courier New',monospace; letter-spacing:1px;"
        )
        lay.addWidget(self._status)

        return sec

    def _make_transcript(self) -> QWidget:
        box = QTextEdit()
        box.setReadOnly(True)
        box.setFixedHeight(168)
        box.setStyleSheet("""
            QTextEdit {
                background: #020710;
                color: #5a9fd4;
                font-size: 12px;
                font-family: 'Courier New', monospace;
                border: none;
                padding: 10px 18px;
            }
            QScrollBar:vertical {
                width: 4px;
                background: #0a1a2e;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #1a4878;
                border-radius: 2px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0; }
        """)
        box.setPlaceholderText("Transcript will appear here...")
        self._transcript = box
        return box

    def _make_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(62)
        bar.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(10)

        inp = QLineEdit()
        inp.setPlaceholderText("Type a command and press Enter...")
        inp.setStyleSheet("""
            QLineEdit {
                background: #080e1c;
                color: #7ab3d4;
                border: 1px solid #182840;
                border-radius: 7px;
                padding: 7px 12px;
                font-size: 12px;
                font-family: 'Courier New', monospace;
            }
            QLineEdit:focus { border-color: #2a5a90; }
        """)
        inp.returnPressed.connect(self._on_text_submit)
        self._input = inp
        lay.addWidget(inp, 1)

        btn = QPushButton("▶  ACTIVATE")
        btn.setFixedSize(120, 38)
        btn.setStyleSheet("""
            QPushButton {
                background: #091e3a;
                color: #4a9ad4;
                border: 1px solid #1a3a60;
                border-radius: 7px;
                font-size: 11px;
                font-weight: bold;
                font-family: 'Courier New', monospace;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: #112a50;
                color: #70c0f0;
                border-color: #2a5a90;
            }
            QPushButton:pressed { background: #071428; }
        """)
        btn.clicked.connect(self._on_activate)
        self._activate_btn = btn
        lay.addWidget(btn)

        return bar

    def _make_divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet("background: #0e2040; border: none;")
        return line

    # ── Worker ────────────────────────────────────────────────

    def _start_worker(self, demo: bool):
        self._worker = VoiceWorker(demo=demo)
        self._worker.state_changed.connect(self._on_state)
        self._worker.nexus_spoke.connect(self._on_nexus)
        self._worker.user_spoke.connect(self._on_user)
        self._worker.error_msg.connect(self._on_error)
        self._worker.start()

        if self._worker.demo_mode:
            self._status.setText(STATUS_TEXT["demo"])

    # ── Slots ─────────────────────────────────────────────────

    def _on_state(self, state: str):
        self._orb.set_state(state)

        pal = PAL.get(state, PAL["idle"])
        self._dot.setStyleSheet(f"color:{pal['dot']}; font-size:9px;")
        self._status.setText(
            STATUS_TEXT.get(
                "demo" if (state == "idle" and self._worker.demo_mode) else state,
                state,
            )
        )
        self._status.setStyleSheet(
            f"color:{pal['label']}; font-size:12px;"
            "font-family:'Courier New',monospace; letter-spacing:1px;"
        )

    def _on_nexus(self, text: str):
        self._append("NEXUS", text, "#26a69a")

    def _on_user(self, text: str):
        self._append("YOU", text, "#5a9fd4")

    def _on_error(self, msg: str):
        self._append("ERR", msg, "#e53935")

    def _on_activate(self):
        if self._worker.isRunning():
            self._worker.trigger()

    def _on_text_submit(self):
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._append("YOU", text, "#5a9fd4")
        self._orb.set_state("processing")

        response = self._worker.respond(text)

        self._orb.set_state("speaking")
        self._append("NEXUS", response, "#26a69a")
        if self._worker._speaker:
            self._worker._speaker.say(response)

        QTimer.singleShot(1800, lambda: self._on_state(
            "demo" if self._worker.demo_mode else "idle"
        ))
        QTimer.singleShot(1800, lambda: self._orb.set_state("idle"))

    def _append(self, who: str, text: str, color: str):
        self._transcript.append(
            f'<span style="color:{color};font-weight:bold">{who}:</span>'
            f'&nbsp;<span style="color:#8ab8d4">{text}</span>'
        )
        sb = self._transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Drag-to-move (frameless window) ──────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, _event):
        self._drag_pos = None

    def closeEvent(self, event):
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        event.accept()


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="NEXUS Voice Interface")
    ap.add_argument("--demo", action="store_true",
                    help="Demo mode — button-triggered, no wake-word model needed")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("NEXUS")
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,       QColor("#05090f"))
    pal.setColor(QPalette.WindowText,   QColor("#c9d1d9"))
    pal.setColor(QPalette.Base,         QColor("#020710"))
    pal.setColor(QPalette.AlternateBase,QColor("#0a1a2e"))
    pal.setColor(QPalette.Text,         QColor("#7ab3d4"))
    pal.setColor(QPalette.Button,       QColor("#091e3a"))
    pal.setColor(QPalette.ButtonText,   QColor("#4a9ad4"))
    app.setPalette(pal)

    win = NexusVoiceApp(demo=args.demo)
    win.setWindowTitle("NEXUS")
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
