"""
ui/nexus_voice_app.py  — NEXUS Molecular Voice Interface
3D particle network that converges into a rotating sphere on wake,
springs open when speaking, and drifts as a nebula when idle.

Launch:
  python ui/nexus_voice_app.py           # full voice mode
  python ui/nexus_voice_app.py --demo    # button-triggered, no mic needed
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

import numpy as np

from PyQt5.QtCore  import Qt, QTimer, QThread, pyqtSignal, QPoint, QPointF, QRectF
from PyQt5.QtGui   import (QPainter, QColor, QRadialGradient, QLinearGradient,
                            QPen, QBrush, QFont, QPainterPath, QRegion)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QTextEdit, QLineEdit, QFrame)

_WIN = sys.platform == "win32"

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

WIN_W   = 520
WIN_H   = 680
FIELD_H = 460        # height of the 3D animation area

N       = 110        # particles
SPH_R   = 1.05       # sphere radius (world units)
FOV     = 370        # perspective focal length
Z_CAM   = 3.6        # camera distance

CONNECT_D  = 0.72    # max 3D distance for drawing a connection line
SPRING_K   = 6.5
SPRING_D   = 3.8
DT         = 0.018

GREETINGS = [
    "Hey boss, what can I do for you?",
    "Hey — I'm listening. Go ahead.",
    "NEXUS online. What do you need?",
    "What's up? I'm all ears.",
    "Yes boss?",
    "At your service.",
]

STATUS = {
    "idle":       "Say  'Hey Nexus'  to activate",
    "demo":       "Press  ACTIVATE  to start",
    "wake":       "Hey!",
    "listening":  "Listening…",
    "processing": "Processing…",
    "speaking":   "Speaking…",
    "error":      "Error — check logs",
}

# per-state palette
STYLE = {
    "idle":       dict(pc=(28,  72, 160), pe=(10, 28,  70), cc=(14, 36,  90), gc=(12, 40, 100), lc="#2a5a9a"),
    "wake":       dict(pc=(220,240,255),  pe=(100,190,255), cc=(80, 170,255), gc=(80,180,255),  lc="#80d8ff"),
    "listening":  dict(pc=(0,  210,255),  pe=(0,  110,190), cc=(0,  140,200), gc=(0,  150,210), lc="#00c8f0"),
    "processing": dict(pc=(190,130,255),  pe=(80,  30,180), cc=(70,  20,160), gc=(100, 40,200), lc="#a070f0"),
    "speaking":   dict(pc=(80, 255,170),  pe=(0,  140, 90), cc=(0,  130, 80), gc=(20, 160,100), lc="#40e090"),
    "error":      dict(pc=(255,100,100),  pe=(160,  0,  0), cc=(120,  0,  0), gc=(180, 20, 20), lc="#e04040"),
}


# ═══════════════════════════════════════════════════════════════
#  3-D MATH HELPERS
# ═══════════════════════════════════════════════════════════════

def fibonacci_sphere(n: int) -> np.ndarray:
    """Evenly distribute n points on a unit sphere (Fibonacci lattice)."""
    i     = np.arange(n, dtype=float)
    phi   = math.pi * (3.0 - math.sqrt(5.0))
    y     = 1.0 - (i / (n - 1)) * 2.0
    r     = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


# ═══════════════════════════════════════════════════════════════
#  PARTICLE FIELD  —  the 3D animation widget
# ═══════════════════════════════════════════════════════════════

class ParticleField(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(FIELD_H)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # ── Particle data (all numpy arrays, shape (N, 3)) ─────
        sphere_pts  = fibonacci_sphere(N) * SPH_R

        # Scatter: random positions in a larger shell
        phi  = np.random.uniform(0, 2 * math.pi, N)
        th   = np.random.uniform(0, math.pi,     N)
        sr   = np.random.uniform(1.0, 2.8,        N)
        idle_pts = np.stack([
            sr * np.sin(th) * np.cos(phi),
            sr * np.sin(th) * np.sin(phi),
            sr * np.cos(th),
        ], axis=1)

        self._sphere  = sphere_pts             # (N,3) — sphere targets
        self._idle    = idle_pts               # (N,3) — scatter targets
        self._pos     = idle_pts.copy()        # current positions
        self._vel     = (np.random.rand(N, 3) - 0.5) * 0.04
        self._target  = idle_pts.copy()        # spring targets
        self._sizes   = np.random.uniform(1.8, 4.2, N)

        # Per-particle explosion multiplier (for speaking state)
        self._expl_mult = np.random.uniform(1.6, 2.8, N)

        # ── Rotation state ─────────────────────────────────────
        self._ry      = 0.0    # y-axis rotation angle
        self._rx      = 0.0    # x-axis tilt angle
        self._rx_tgt  = 0.0

        # ── Animation state ─────────────────────────────────────
        self._state       = "idle"
        self._speak_phase = 0.0   # oscillates during speaking
        self._wake_flash  = 0.0   # bright flash on wake

        # ── Pairwise distance cache (updated every 8 ticks) ────
        self._conn_pairs : np.ndarray = np.empty((0, 2), int)
        self._conn_alpha : np.ndarray = np.empty(0)
        self._dist_tick  = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(20)    # 50 fps

    # ── State control ─────────────────────────────────────────

    def set_state(self, state: str):
        prev          = self._state
        self._state   = state

        if state == "wake":
            self._wake_flash = 1.0

        # Recompute targets
        if state in ("wake", "listening", "processing"):
            self._target = self._sphere.copy()
        elif state == "idle":
            self._target = self._idle.copy()
        elif state == "speaking":
            self._speak_phase = 0.0
            self._target = self._sphere * self._expl_mult[:, None]
        elif state == "error":
            self._target = self._idle * 0.6

    # ── Physics tick ──────────────────────────────────────────

    def _tick(self):
        state = self._state

        # ── Speaking: oscillate explosion ──────────────────────
        if state == "speaking":
            self._speak_phase += 0.06
            burst = 1.0 + 0.55 * abs(math.sin(self._speak_phase * 0.7))
            self._target = self._sphere * (self._expl_mult * burst)[:, None]

        # ── Spring physics (vectorized) ─────────────────────────
        force      = SPRING_K * (self._target - self._pos) - SPRING_D * self._vel
        self._vel += force * DT
        self._pos += self._vel * DT

        # ── Rotation ────────────────────────────────────────────
        if state == "idle":
            self._ry      += 0.003
            self._rx_tgt   = 0.18 * math.sin(self._ry * 0.4)
        elif state == "listening":
            self._ry      += 0.012
            self._rx_tgt   = 0.30 * math.sin(self._ry * 0.3)
        elif state == "processing":
            self._ry      += 0.018
            self._rx_tgt   = 0.0
        elif state in ("wake", "speaking"):
            self._ry      += 0.008
            self._rx_tgt   = 0.12 * math.sin(self._ry * 0.5)

        self._rx += (self._rx_tgt - self._rx) * 0.04

        # ── Wake flash decay ────────────────────────────────────
        self._wake_flash = max(0.0, self._wake_flash - 0.035)

        # ── Rebuild connection list every 8 ticks ───────────────
        self._dist_tick += 1
        if self._dist_tick >= 8:
            self._dist_tick = 0
            self._rebuild_connections()

        self.update()

    def _rebuild_connections(self):
        p = self._pos
        # (N,N,3) difference — use broadcasting
        diff   = p[:, None, :] - p[None, :, :]
        dsq    = (diff * diff).sum(axis=2)
        mask   = np.triu(dsq < CONNECT_D * CONNECT_D, k=1)
        pairs  = np.argwhere(mask)
        if len(pairs):
            d    = np.sqrt(dsq[pairs[:, 0], pairs[:, 1]])
            self._conn_alpha = ((1.0 - d / CONNECT_D) ** 1.6)
        else:
            self._conn_alpha = np.empty(0)
        self._conn_pairs = pairs

    # ── Paint ──────────────────────────────────────────────────

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        W, H   = self.width(), self.height()
        cx, cy = W * 0.5, H * 0.5
        st     = STYLE.get(self._state, STYLE["idle"])

        # ── Background ──────────────────────────────────────────
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(2, 5, 14))
        painter.drawRect(0, 0, W, H)

        # ── Ambient glow behind sphere ──────────────────────────
        gr, gg, gb = st["gc"]
        glow_r     = min(W, H) * 0.55
        bg         = QRadialGradient(cx, cy, glow_r)
        bg.setColorAt(0.0, QColor(gr, gg, gb, 28))
        bg.setColorAt(0.5, QColor(gr, gg, gb, 10))
        bg.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(bg))
        painter.drawEllipse(QRectF(cx - glow_r, cy - glow_r, glow_r * 2, glow_r * 2))

        # ── Apply rotation ──────────────────────────────────────
        R     = rot_y(self._ry) @ rot_x(self._rx)
        rot   = self._pos @ R.T                # (N, 3)

        # ── Perspective projection ──────────────────────────────
        z_d   = rot[:, 2] + Z_CAM
        z_d   = np.maximum(z_d, 0.2)
        sc    = FOV / z_d
        sx    = cx + rot[:, 0] * sc
        sy    = cy - rot[:, 1] * sc             # flip y

        # Depth normalisation (0=far, 1=close)
        zmin, zmax = rot[:, 2].min(), rot[:, 2].max()
        depth  = (rot[:, 2] - zmin) / max(zmax - zmin, 1e-6)

        # ── Connection lines ────────────────────────────────────
        pr, pg, pb = st["cc"]
        for k, (i, j) in enumerate(self._conn_pairs):
            a   = int(self._conn_alpha[k] * 140)
            pen = QPen(QColor(pr, pg, pb, a), 0.9)
            painter.setPen(pen)
            painter.drawLine(QPointF(sx[i], sy[i]), QPointF(sx[j], sy[j]))

        # ── Particles (back to front) ───────────────────────────
        order  = np.argsort(rot[:, 2])          # ascending z → far first
        cr, cg, cb = st["pc"]
        er, eg, eb = st["pe"]

        for i in order:
            d     = float(depth[i])
            alpha = int(160 + 90 * d)
            r_    = int(cr * d + er * (1 - d))
            g_    = int(cg * d + eg * (1 - d))
            b_    = int(cb * d + eb * (1 - d))
            col   = QColor(r_, g_, b_, alpha)

            size  = self._sizes[i] * (0.55 + 0.75 * d) * (sc[i] / (FOV / Z_CAM))
            size  = max(0.8, size)
            half  = size * 0.5

            # Outer glow dot
            gd = QRadialGradient(sx[i], sy[i], size * 2.2)
            gd.setColorAt(0.0, QColor(r_, g_, b_, int(60 * d)))
            gd.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(gd))
            painter.setPen(Qt.NoPen)
            gs = size * 2.2
            painter.drawEllipse(QRectF(sx[i] - gs, sy[i] - gs, gs * 2, gs * 2))

            # Core dot
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QRectF(sx[i] - half, sy[i] - half, size, size))

        # ── Wake flash ──────────────────────────────────────────
        if self._wake_flash > 0:
            f   = self._wake_flash
            fgr = QRadialGradient(cx, cy, min(W, H) * 0.6 * f)
            fgr.setColorAt(0.0, QColor(200, 230, 255, int(180 * f)))
            fgr.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(fgr))
            painter.setPen(Qt.NoPen)
            rr = min(W, H) * 0.6 * f
            painter.drawEllipse(QRectF(cx - rr, cy - rr, rr * 2, rr * 2))

        # ── Status overlay text ──────────────────────────────────
        label_color = QColor(st["lc"])
        label_color.setAlpha(200)
        painter.setPen(label_color)
        f = QFont("Courier New", 11, QFont.Normal)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        painter.setFont(f)
        txt = STATUS.get(self._state, self._state)
        painter.drawText(QRectF(0, H - 44, W, 36), Qt.AlignCenter, txt)


# ═══════════════════════════════════════════════════════════════
#  ROUND PANEL  —  window background
# ═══════════════════════════════════════════════════════════════

class RoundPanel(QWidget):
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(20, 45, 90, 120), 1))
        p.setBrush(QColor(2, 5, 14))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 18, 18)


# ═══════════════════════════════════════════════════════════════
#  VOICE WORKER
# ═══════════════════════════════════════════════════════════════

class VoiceWorker(QThread):
    state_changed = pyqtSignal(str)
    nexus_spoke   = pyqtSignal(str)
    user_spoke    = pyqtSignal(str)
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

    def trigger(self):
        self._trigger.set()

    def stop(self):
        self._stop = True
        self._trigger.set()

    def run(self):
        if self._listener:
            try:
                self._listener.start()
            except Exception:
                pass

        while not self._stop:
            self.state_changed.emit("idle")

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

            greeting = random.choice(GREETINGS)
            self.state_changed.emit("wake")
            self.nexus_spoke.emit(greeting)
            if self._speaker:
                self._speaker.say(greeting)
            time.sleep(0.9)

            self.state_changed.emit("listening")

            if not self._listener or not self._transcriber:
                self.nexus_spoke.emit("[Mic / STT unavailable — type below]")
                time.sleep(1.5)
                self.state_changed.emit("idle")
                continue

            audio = self._listener.capture_phrase(verbose=False)
            if audio is None or len(audio) == 0:
                self.state_changed.emit("idle")
                continue

            self.state_changed.emit("processing")
            try:
                result = self._transcriber.transcribe(audio)
                if not result.is_speech:
                    self.state_changed.emit("idle")
                    continue
                user_text = result.text.strip()
            except Exception as e:
                self.error_msg.emit(f"STT: {e}")
                self.state_changed.emit("idle")
                continue

            self.user_spoke.emit(user_text)
            response = self.respond(user_text)

            self.state_changed.emit("speaking")
            self.nexus_spoke.emit(response)
            if self._speaker:
                self._speaker.say(response, block=True)
            else:
                time.sleep(max(1.2, len(response) * 0.042))

            self.state_changed.emit("idle")

        if self._listener:
            try: self._listener.stop()
            except Exception: pass
        if self._speaker:
            try: self._speaker.shutdown()
            except Exception: pass

    def respond(self, text: str) -> str:
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
        return "Got it. Start Ollama for full AI responses."


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════

class NexusApp(QMainWindow):

    def __init__(self, demo: bool = False):
        super().__init__()
        self._drag_pos: Optional[QPoint] = None

        print("[NEXUS] init: setting flags", flush=True)
        self.setWindowFlags(Qt.FramelessWindowHint)

        if _WIN:
            print("[NEXUS] init: Windows mode — solid background", flush=True)
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(2, 5, 14))
            self.setPalette(pal)
            self.setAutoFillBackground(True)
        else:
            self.setAttribute(Qt.WA_TranslucentBackground)

        print("[NEXUS] init: setting size", flush=True)
        self.setFixedSize(WIN_W, WIN_H)

        print("[NEXUS] init: setting position", flush=True)
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = max(0, (geo.width()  - WIN_W) // 2)
            y = max(0, (geo.height() - WIN_H) // 2)
            self.move(x, y)

        print("[NEXUS] init: building UI", flush=True)
        self._build_ui()

        print("[NEXUS] init: starting worker", flush=True)
        self._start_worker(demo)
        print("[NEXUS] init: done", flush=True)

    # ── Layout ────────────────────────────────────────────────

    def _build_ui(self):
        panel = RoundPanel(self)
        panel.setGeometry(0, 0, WIN_W, WIN_H)

        root = QVBoxLayout(panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_titlebar())
        self._field = ParticleField()
        root.addWidget(self._field, 1)
        root.addWidget(self._make_divider())
        root.addWidget(self._make_transcript())
        root.addWidget(self._make_divider())
        root.addWidget(self._make_bottom())

    def _make_titlebar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(42)
        bar.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)

        for color, hover, slot in (
            ("#ff5f57", "#ff3b30", self.close),
            ("#febc2e", "#f5a000", self.showMinimized),
        ):
            btn = QPushButton()
            btn.setFixedSize(13, 13)
            btn.setStyleSheet(
                f"QPushButton{{background:{color};border-radius:6px;border:none;}}"
                f"QPushButton:hover{{background:{hover};}}")
            btn.clicked.connect(slot)
            lay.addWidget(btn)
            lay.addSpacing(6)

        lay.addStretch()

        title = QLabel("N · E · X · U · S")
        title.setStyleSheet(
            "color:#1e4a80; font-size:11px; font-weight:bold;"
            "font-family:'Courier New',monospace; letter-spacing:3px;")
        lay.addWidget(title)

        lay.addStretch()

        self._dot = QLabel("◉")
        self._dot.setStyleSheet("color:#0a2a50; font-size:10px;")
        lay.addWidget(self._dot)
        return bar

    def _make_transcript(self) -> QTextEdit:
        box = QTextEdit()
        box.setReadOnly(True)
        box.setFixedHeight(108)
        box.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: #3a6a9a;
                font-size: 12px;
                font-family: 'Courier New', monospace;
                border: none;
                padding: 8px 20px;
            }
            QScrollBar:vertical {
                width: 3px; background: transparent; border: none;
            }
            QScrollBar::handle:vertical {
                background: #1a3a70; border-radius: 1px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0; }
        """)
        box.setPlaceholderText("Transcript…")
        self._transcript = box
        return box

    def _make_bottom(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(60)
        bar.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(10)

        inp = QLineEdit()
        inp.setPlaceholderText("Type a command…")
        inp.setStyleSheet("""
            QLineEdit {
                background: #030c1c;
                color: #4a8ab8;
                border: 1px solid #0e2a4a;
                border-radius: 8px;
                padding: 7px 14px;
                font-size: 12px;
                font-family: 'Courier New', monospace;
            }
            QLineEdit:focus { border-color: #1a4a80; }
        """)
        inp.returnPressed.connect(self._on_text)
        self._input = inp
        lay.addWidget(inp, 1)

        btn = QPushButton("▶  ACTIVATE")
        btn.setFixedSize(118, 38)
        btn.setStyleSheet("""
            QPushButton {
                background: #040e20;
                color: #2a6090;
                border: 1px solid #0e2a4a;
                border-radius: 8px;
                font-size: 10px; font-weight: bold;
                font-family: 'Courier New', monospace;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: #0a1e40;
                color: #4a90d0;
                border-color: #1a4070;
            }
            QPushButton:pressed { background: #020a18; }
        """)
        btn.clicked.connect(self._on_activate)
        lay.addWidget(btn)
        return bar

    def _make_divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet("background: #0a1e3a; border: none; margin: 0 0px;")
        return line

    # ── Worker ────────────────────────────────────────────────

    def _start_worker(self, demo: bool):
        self._worker = VoiceWorker(demo=demo)
        self._worker.state_changed.connect(self._on_state)
        self._worker.nexus_spoke.connect(self._on_nexus)
        self._worker.user_spoke.connect(self._on_user)
        self._worker.error_msg.connect(self._on_error)
        self._worker.start()

    # ── Slots ─────────────────────────────────────────────────

    def _on_state(self, state: str):
        self._field.set_state(state)
        st = STYLE.get(state, STYLE["idle"])
        self._dot.setStyleSheet(f"color:{st['lc']}; font-size:10px;")

    def _on_nexus(self, text: str):
        self._append("NEXUS", text, "#28a876")

    def _on_user(self, text: str):
        self._append("YOU",   text, "#2a6aaa")

    def _on_error(self, msg: str):
        self._append("ERR",   msg,  "#c03030")

    def _on_activate(self):
        if self._worker.isRunning():
            self._worker.trigger()

    def _on_text(self):
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._append("YOU", text, "#2a6aaa")
        self._field.set_state("processing")

        response = self._worker.respond(text)

        self._field.set_state("speaking")
        self._append("NEXUS", response, "#28a876")
        if self._worker._speaker:
            self._worker._speaker.say(response)

        QTimer.singleShot(2000, lambda: self._field.set_state("idle"))
        QTimer.singleShot(2000, lambda: self._on_state("idle"))

    def _append(self, who: str, msg: str, color: str):
        self._transcript.append(
            f'<span style="color:{color};font-weight:bold">{who}:</span>'
            f'&nbsp;<span style="color:#3a6a90">{msg}</span>'
        )
        sb = self._transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Drag ──────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, _):
        self._drag_pos = None

    def closeEvent(self, e):
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        e.accept()


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    try:
        # HiDPI support — must be set before QApplication
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

        app = QApplication(sys.argv)
        app.setApplicationName("NEXUS")
        app.setStyle("Fusion")

        from PyQt5.QtGui import QPalette
        pal = QPalette()
        pal.setColor(QPalette.Window,     QColor(2,  5, 14))
        pal.setColor(QPalette.WindowText, QColor(180, 200, 220))
        pal.setColor(QPalette.Base,       QColor(3,  8, 20))
        pal.setColor(QPalette.Text,       QColor(80, 140, 180))
        app.setPalette(pal)

        print("[NEXUS] Starting…")
        win = NexusApp(demo=args.demo)
        win.setWindowTitle("NEXUS")
        win.show()
        win.raise_()
        win.activateWindow()
        print("[NEXUS] Window shown — entering event loop")
        sys.exit(app.exec_())

    except Exception as e:
        import traceback
        print(f"\n[NEXUS] STARTUP ERROR: {e}")
        traceback.print_exc()
        if _WIN:
            input("\nPress Enter to exit…")
        sys.exit(1)


if __name__ == "__main__":
    main()
