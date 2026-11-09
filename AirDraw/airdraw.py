"""
AirDraw - draw on top of ANY Windows app with hand gestures.

Uses the current MediaPipe Tasks API (HandLandmarker). The old `mp.solutions`
API is gone from recent MediaPipe releases, so this file does not use it.

Install (use a fresh venv, 64-bit Python 3.10-3.12):
    python -m pip install -U pip
    pip install -U opencv-python mediapipe PyQt6 keyboard

Run:
    python airdraw.py              # add --preview for a camera debug window

The hand model (hand_landmarker.task, ~7 MB) is downloaded automatically
next to this script on first run.

Gestures (one hand, palm facing the camera):
    Index finger up only ........ draw
    Index + middle up (peace) ... move pointer without drawing
    Open palm (hold ~1 s) ....... clear the screen
    Fist / anything else ........ pen up

Global hotkeys:
    Ctrl+Alt+D  hand tracking on/off (off = camera released)
    Ctrl+Alt+C  clear drawing
    Ctrl+Alt+H  hide/show overlay
    Ctrl+Alt+K  next color
    Ctrl+Alt+Q  quit
(Run from an admin terminal if hotkeys don't fire while an admin app has focus.)
"""
import argparse
import ctypes
import math
import sys
import threading
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path

import cv2
import keyboard
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PyQt6.QtCore import QObject, QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QGuiApplication, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget

# ----------------------------- tuning knobs -----------------------------
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")

FRAME_W, FRAME_H = 640, 480
ACTIVE_X = (0.15, 0.85)   # part of the camera frame that maps to the full screen width
ACTIVE_Y = (0.10, 0.75)   # ...and height (hands rarely reach the bottom of the frame)
STABLE_FRAMES = 2         # frames a gesture must persist before it counts
LOST_FRAMES = 6           # frames without a hand before the pointer disappears
CLEAR_HOLD_S = 1.0        # hold open palm this long to clear
SMOOTH_MIN_CUTOFF = 1.2   # lower = smoother but laggier
SMOOTH_BETA = 10.0        # higher = more responsive on fast moves
COLORS = ["#ff2d55", "#30d158", "#0a84ff", "#ffd60a", "#ffffff", "#bf5af2"]

# Hand skeleton, used only for the --preview window (the Tasks API doesn't ship a drawer).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

user32 = ctypes.windll.user32 if sys.platform == "win32" else None


# ----------------------------- helpers -----------------------------
def clamp01(v):
    return max(0.0, min(1.0, v))


def ensure_model():
    """Download the HandLandmarker model once, next to this script."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return
    print(f"Downloading hand model to {MODEL_PATH} ...")
    tmp = MODEL_PATH.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"Could not download the model: {exc}\n"
                 f"Download it manually from:\n  {MODEL_URL}\n"
                 f"and save it as:\n  {MODEL_PATH}")


class OneEuro:
    """One Euro filter: removes jitter when slow, stays responsive when fast."""

    def __init__(self, min_cutoff=SMOOTH_MIN_CUTOFF, beta=SMOOTH_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def classify(lm, w, h):
    """Return 'draw', 'hover', 'clear' or 'idle' from 21 hand landmarks."""

    def pt(i):
        return lm[i].x * w, lm[i].y * h

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    wrist = pt(0)

    def extended(tip, pip):  # rotation-independent: tip farther from wrist than PIP joint
        return dist(wrist, pt(tip)) > dist(wrist, pt(pip)) * 1.1

    index, middle, ring, pinky = extended(8, 6), extended(12, 10), extended(16, 14), extended(20, 18)
    if index and not middle and not ring and not pinky:
        return "draw"
    if index and middle and not ring and not pinky:
        return "hover"
    if index and middle and ring and pinky:
        return "clear"
    return "idle"


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


# ----------------------------- camera / gesture thread -----------------------------
class HandWorker(QThread):
    moved = pyqtSignal(float, float, str)   # normalized screen x, y, stable gesture
    lost = pyqtSignal()
    clear_requested = pyqtSignal()
    camera_error = pyqtSignal(str)

    def __init__(self, cam_index, preview):
        super().__init__()
        self.cam_index = cam_index
        self.preview = preview
        self.enabled = threading.Event()
        self.enabled.set()
        self.running = True

    def _open_camera(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _make_landmarker(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.HandLandmarker.create_from_options(options)

    def run(self):
        try:
            landmarker = self._make_landmarker()
        except Exception as exc:
            self.camera_error.emit(f"Failed to load hand model: {exc}")
            return

        fx, fy = OneEuro(), OneEuro()
        cap = None
        cand, cand_n, stable = "idle", 0, "idle"
        missing = LOST_FRAMES
        clear_since, clear_fired = None, False
        last_ts = -1  # VIDEO mode requires strictly increasing timestamps (ms)

        while self.running:
            if not self.enabled.is_set():
                if cap is not None:
                    cap.release()
                    cap = None
                    self.lost.emit()
                    if self.preview:
                        cv2.destroyAllWindows()
                time.sleep(0.1)
                continue

            if cap is None:
                cap = self._open_camera()
                if cap is None:
                    self.camera_error.emit(f"Could not open camera {self.cam_index}")
                    self.enabled.clear()
                    continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # mirror so movement feels natural
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = int(time.monotonic() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            res = landmarker.detect_for_video(mp_image, ts)
            now = time.time()

            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
                missing = 0

                g = classify(lm, w, h)
                if g == cand:
                    cand_n += 1
                else:
                    cand, cand_n = g, 1
                if cand_n >= STABLE_FRAMES:
                    stable = cand

                tip = lm[8]
                nx = clamp01((tip.x - ACTIVE_X[0]) / (ACTIVE_X[1] - ACTIVE_X[0]))
                ny = clamp01((tip.y - ACTIVE_Y[0]) / (ACTIVE_Y[1] - ACTIVE_Y[0]))
                sx, sy = fx(nx, now), fy(ny, now)

                if stable == "clear":
                    if clear_since is None:
                        clear_since = now
                    elif not clear_fired and now - clear_since >= CLEAR_HOLD_S:
                        clear_fired = True
                        self.clear_requested.emit()
                else:
                    clear_since, clear_fired = None, False

                self.moved.emit(sx, sy, stable)

                if self.preview:
                    draw_hand(frame, lm)
                    cv2.putText(frame, stable, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                missing += 1
                if missing == LOST_FRAMES:
                    fx.reset()
                    fy.reset()
                    cand, cand_n, stable = "idle", 0, "idle"
                    clear_since, clear_fired = None, False
                    self.lost.emit()

            if self.preview:
                cv2.imshow("AirDraw preview", frame)
                cv2.waitKey(1)

        if cap is not None:
            cap.release()
        landmarker.close()
        if self.preview:
            cv2.destroyAllWindows()


# ----------------------------- overlay window -----------------------------
class Overlay(QWidget):
    def __init__(self, color, pen_width):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput   # click-through
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QGuiApplication.primaryScreen()
        # 1px shorter than the screen so Windows doesn't treat the overlay as a
        # fullscreen app (which can hide the taskbar / confuse other apps).
        geo = screen.geometry().adjusted(0, 0, 0, -1)
        self.setGeometry(geo)

        dpr = screen.devicePixelRatio()
        self.canvas = QImage(int(geo.width() * dpr), int(geo.height() * dpr),
                             QImage.Format.Format_ARGB32_Premultiplied)
        self.canvas.setDevicePixelRatio(dpr)
        self.canvas.fill(Qt.GlobalColor.transparent)

        self.color = QColor(color)
        self.pen_width = pen_width
        self.cursor_pos = None
        self.gesture = "idle"
        self.last = None

        # Keep the overlay above everything, including borderless fullscreen apps.
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self.force_topmost)
        self._topmost_timer.start(1000)

    def force_topmost(self):
        if user32 is None or not self.isVisible():
            return
        try:
            user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, wintypes.UINT]
            # HWND_TOPMOST=-1 ; NOSIZE|NOMOVE|NOACTIVATE|SHOWWINDOW
            user32.SetWindowPos(int(self.winId()), wintypes.HWND(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010 | 0x0040)
        except Exception:
            pass

    # ---- slots ----
    def on_hand(self, nx, ny, gesture):
        p = QPointF(nx * self.width(), ny * self.height())
        self.cursor_pos, self.gesture = p, gesture
        if gesture == "draw":
            self._stroke(self.last or p, p)
            self.last = p
        else:
            self.last = None
        self.update()

    def on_lost(self):
        self.cursor_pos, self.last = None, None
        self.update()

    def clear(self):
        self.canvas.fill(Qt.GlobalColor.transparent)
        self.last = None
        self.update()

    def set_color(self, color):
        self.color = QColor(color)

    # ---- drawing ----
    def _stroke(self, a, b):
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.color, self.pen_width, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(a, b)
        painter.end()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawImage(0, 0, self.canvas)
        c = self.cursor_pos
        if c is not None:
            r = self.pen_width / 2 + 3
            if self.gesture == "draw":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(self.color)
                p.drawEllipse(c, r, r)
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
                ring = QColor(255, 80, 80) if self.gesture == "clear" else QColor(255, 255, 255)
                if self.gesture == "idle":
                    ring.setAlpha(120)
                p.setPen(QPen(QColor(0, 0, 0, 160), 5))
                p.drawEllipse(c, 14, 14)
                p.setPen(QPen(ring, 3))
                p.drawEllipse(c, 14, 14)
        p.end()


# ----------------------------- global hotkeys bridge -----------------------------
class HotkeyBridge(QObject):
    toggle = pyqtSignal()
    clear = pyqtSignal()
    hide_show = pyqtSignal()
    next_color = pyqtSignal()
    quit = pyqtSignal()


def make_icon(color):
    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(QPen(QColor(30, 30, 30), 2))
    p.drawEllipse(4, 4, 24, 24)
    p.end()
    return QIcon(pm)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Draw over any Windows app with hand gestures")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--preview", action="store_true", help="show camera debug window")
    ap.add_argument("--width", type=int, default=6, help="pen width in px")
    args = ap.parse_args()

    ensure_model()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    color_idx = [0]
    overlay = Overlay(COLORS[0], args.width)
    overlay.show()
    overlay.force_topmost()

    worker = HandWorker(args.camera, args.preview)
    worker.moved.connect(overlay.on_hand)
    worker.lost.connect(overlay.on_lost)
    worker.clear_requested.connect(overlay.clear)

    tray = QSystemTrayIcon(make_icon(COLORS[0]), app)
    menu = QMenu()

    def toggle_tracking():
        if worker.enabled.is_set():
            worker.enabled.clear()
            tray.showMessage("AirDraw", "Hand tracking OFF (camera released)")
        else:
            worker.enabled.set()
            tray.showMessage("AirDraw", "Hand tracking ON")

    def toggle_overlay():
        overlay.setVisible(not overlay.isVisible())
        if overlay.isVisible():
            overlay.force_topmost()

    def cycle_color():
        color_idx[0] = (color_idx[0] + 1) % len(COLORS)
        overlay.set_color(COLORS[color_idx[0]])
        tray.setIcon(make_icon(COLORS[color_idx[0]]))

    def quit_app():
        worker.running = False
        worker.wait(3000)
        keyboard.unhook_all()
        tray.hide()
        app.quit()

    for label, fn in [("Toggle tracking  (Ctrl+Alt+D)", toggle_tracking),
                      ("Clear  (Ctrl+Alt+C)", overlay.clear),
                      ("Hide/show overlay  (Ctrl+Alt+H)", toggle_overlay),
                      ("Next color  (Ctrl+Alt+K)", cycle_color),
                      ("Quit  (Ctrl+Alt+Q)", quit_app)]:
        act = QAction(label, menu)
        act.triggered.connect(fn)
        menu.addAction(act)
    tray.setContextMenu(menu)
    tray.setToolTip("AirDraw")
    tray.show()

    bridge = HotkeyBridge()
    bridge.toggle.connect(toggle_tracking)
    bridge.clear.connect(overlay.clear)
    bridge.hide_show.connect(toggle_overlay)
    bridge.next_color.connect(cycle_color)
    bridge.quit.connect(quit_app)
    keyboard.add_hotkey("ctrl+alt+d", bridge.toggle.emit)
    keyboard.add_hotkey("ctrl+alt+c", bridge.clear.emit)
    keyboard.add_hotkey("ctrl+alt+h", bridge.hide_show.emit)
    keyboard.add_hotkey("ctrl+alt+k", bridge.next_color.emit)
    keyboard.add_hotkey("ctrl+alt+q", bridge.quit.emit)

    worker.camera_error.connect(lambda msg: tray.showMessage("AirDraw", msg))
    worker.start()
    tray.showMessage("AirDraw", "Running. Ctrl+Alt+D toggles tracking, Ctrl+Alt+Q quits.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()