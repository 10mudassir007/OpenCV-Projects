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
from PyQt6.QtGui import (
    QAction,
    QColor,
    QGuiApplication,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)


# ----------------------------- tuning knobs -----------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")

FRAME_W, FRAME_H = 360, 240
ACTIVE_X = (0.15, 0.85)  # camera area mapped to screen width
ACTIVE_Y = (0.10, 0.75)  # camera area mapped to screen height
STABLE_FRAMES = 2
LOST_FRAMES = 6
CLEAR_HOLD_S = 1.0
COLORS = ["#ff2d55", "#30d158", "#0a84ff", "#ffd60a", "#ffffff", "#bf5af2"]

# One Euro filter. Tune one at a time:
#   wobbles when hand is still -> lower MIN_CUTOFF (try 0.6-0.8)
#   lags on fast strokes       -> raise BETA (try 15-25)
SMOOTH_MIN_CUTOFF = 1.2
SMOOTH_BETA = 10.0

# Gesture classification: cosine of the joint bend angle above which a finger
# counts as straight. Lower = more forgiving.
STRAIGHT_COS = 0.5

# Frames of ink held back so the unstable tail before a gesture change can be
# discarded. ~30 ms per frame at 30 fps; set to 1 if the trailing ink bothers you.
LAG = 2

# MediaPipe confidence thresholds. Raise tracking to 0.6-0.7 if landmarks jitter
# or snap to wrong poses; lower to 0.4 if tracking drops too often.
MIN_DETECTION_CONF = 0.6
MIN_PRESENCE_CONF = 0.5
MIN_TRACKING_CONF = 0.5

# OpenCV uses: output = input * contrast + brightness.
DEFAULT_BRIGHTNESS = 25       # slider range: -50 to +100
DEFAULT_CONTRAST = 1.30       # slider range: 0.50x to 2.50x

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

user32 = ctypes.windll.user32 if sys.platform == "win32" else None


# -------------------------------- helpers --------------------------------
def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def ensure_model() -> None:
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
        sys.exit(
            f"Could not download the model: {exc}\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        )


class OneEuro:
    """One Euro filter: reduces jitter while remaining responsive to fast motion."""

    def __init__(
        self,
        min_cutoff: float = SMOOTH_MIN_CUTOFF,
        beta: float = SMOOTH_BETA,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, timestamp: float) -> float:
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, timestamp
            return x

        dt = max(timestamp - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, timestamp
        return x_hat


def classify(landmarks, width: int, height: int) -> str:
    """Return 'draw', 'hover', 'clear' or 'idle' from 21 hand landmarks.

    Uses the 3D bend angle at each finger's PIP joint instead of wrist
    distance, so tilting the hand toward the camera doesn't misfire.
    """

    def vec(a: int, b: int):  # 3D vector a->b, z scaled like x
        return (
            (landmarks[b].x - landmarks[a].x) * width,
            (landmarks[b].y - landmarks[a].y) * height,
            (landmarks[b].z - landmarks[a].z) * width,
        )

    def straight(mcp: int, pip: int, tip: int) -> bool:
        a, b = vec(mcp, pip), vec(pip, tip)
        dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        na = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        nb = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        return dot / (na * nb + 1e-6) > STRAIGHT_COS

    index = straight(5, 6, 8)
    middle = straight(9, 10, 12)
    ring = straight(13, 14, 16)
    pinky = straight(17, 18, 20)

    if index and not middle and not ring and not pinky:
        return "draw"
    if index and middle and not ring and not pinky:
        return "hover"
    if index and middle and ring and pinky:
        return "clear"
    return "idle"


def draw_hand(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 90), 2)
    for point in points:
        cv2.circle(frame, point, 3, (30, 30, 255), -1)


# ----------------------------- camera worker -----------------------------
class HandWorker(QThread):
    moved = pyqtSignal(float, float, str)
    lost = pyqtSignal()
    clear_requested = pyqtSignal()
    camera_error = pyqtSignal(str)
    camera_frame = pyqtSignal(QImage)
    camera_status = pyqtSignal(str)

    def __init__(self, cam_index: int, preview: bool) -> None:
        super().__init__()
        self.cam_index = cam_index
        self.preview = preview
        self.enabled = threading.Event()
        self.enabled.set()
        self.running = True

        self._adjustment_lock = threading.Lock()
        self._brightness = DEFAULT_BRIGHTNESS
        self._contrast = DEFAULT_CONTRAST

    def set_image_adjustments(self, brightness: int, contrast_percent: int) -> None:
        """Thread-safe update from the camera window's sliders."""
        with self._adjustment_lock:
            self._brightness = int(brightness)
            self._contrast = max(0.5, min(2.5, contrast_percent / 100.0))

    def _get_image_adjustments(self):
        with self._adjustment_lock:
            return self._brightness, self._contrast

    def _open_camera(self):
        if sys.platform == "win32":
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.cam_index)

        # MJPG avoids the slow YUY2 default that silently caps many webcams at
        # 15 fps under DSHOW. FOURCC must be set before the frame size.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always read the newest frame

        if not cap.isOpened():
            cap.release()
            return None
        return cap

    @staticmethod
    def _make_landmarker():
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=MIN_DETECTION_CONF,
            min_hand_presence_confidence=MIN_PRESENCE_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        )
        return vision.HandLandmarker.create_from_options(options)

    def run(self) -> None:
        try:
            landmarker = self._make_landmarker()
        except Exception as exc:
            self.camera_error.emit(f"Failed to load hand model: {exc}")
            self.camera_status.emit("Hand model failed to load")
            return

        filter_x, filter_y = OneEuro(), OneEuro()
        cap = None
        candidate, candidate_count, stable_gesture = "idle", 0, "idle"
        missing_frames = LOST_FRAMES
        clear_since, clear_fired = None, False
        last_timestamp_ms = -1

        try:
            while self.running:
                if not self.enabled.is_set():
                    if cap is not None:
                        cap.release()
                        cap = None
                        self.lost.emit()
                        self.camera_status.emit("Tracking paused — camera released")
                        if self.preview:
                            cv2.destroyWindow("AirDraw debug preview")
                    time.sleep(0.1)
                    continue

                if cap is None:
                    cap = self._open_camera()
                    if cap is None:
                        message = f"Could not open camera {self.cam_index}"
                        self.camera_error.emit(message)
                        self.camera_status.emit(message)
                        self.enabled.clear()
                        continue
                    self.camera_status.emit("Camera live — show one hand")

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                # Mirror first so the preview behaves like a mirror.
                frame = cv2.flip(frame, 1)

                # Apply the live brightness/contrast settings before inference,
                # so what the user sees is the same image the detector receives.
                brightness, contrast = self._get_image_adjustments()
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = int(time.monotonic() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                now = time.time()
                gesture_text = "Show one hand"

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    missing_frames = 0

                    gesture = classify(landmarks, width, height)
                    if gesture == candidate:
                        candidate_count += 1
                    else:
                        candidate, candidate_count = gesture, 1
                    if candidate_count >= STABLE_FRAMES:
                        stable_gesture = candidate

                    fingertip = landmarks[8]
                    normalized_x = clamp01(
                        (fingertip.x - ACTIVE_X[0]) / (ACTIVE_X[1] - ACTIVE_X[0])
                    )
                    normalized_y = clamp01(
                        (fingertip.y - ACTIVE_Y[0]) / (ACTIVE_Y[1] - ACTIVE_Y[0])
                    )
                    screen_x = filter_x(normalized_x, now)
                    screen_y = filter_y(normalized_y, now)

                    if stable_gesture == "clear":
                        if clear_since is None:
                            clear_since = now
                        elif not clear_fired and now - clear_since >= CLEAR_HOLD_S:
                            clear_fired = True
                            self.clear_requested.emit()
                    else:
                        clear_since, clear_fired = None, False

                    self.moved.emit(screen_x, screen_y, stable_gesture)
                    draw_hand(frame, landmarks)
                    gesture_text = f"Gesture: {stable_gesture.upper()}"
                else:
                    missing_frames += 1
                    if missing_frames == LOST_FRAMES:
                        filter_x.reset()
                        filter_y.reset()
                        candidate, candidate_count, stable_gesture = "idle", 0, "idle"
                        clear_since, clear_fired = None, False
                        self.lost.emit()

                cv2.putText(
                    frame,
                    gesture_text,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (80, 255, 120),
                    2,
                    cv2.LINE_AA,
                )

                # Copy detaches the QImage from the OpenCV frame's memory.
                camera_image = QImage(
                    frame.data,
                    width,
                    height,
                    int(frame.strides[0]),
                    QImage.Format.Format_BGR888,
                ).copy()
                self.camera_frame.emit(camera_image)

                if self.preview:
                    cv2.imshow("AirDraw debug preview", frame)
                    cv2.waitKey(1)
        finally:
            if cap is not None:
                cap.release()
            landmarker.close()
            if self.preview:
                cv2.destroyAllWindows()
            self.camera_status.emit("Camera stopped")


# ------------------------------ camera view ------------------------------
class CameraView(QWidget):
    """Separate live camera panel with brightness and contrast controls."""

    def __init__(self, worker: HandWorker) -> None:
        super().__init__()
        self.worker = worker
        self.setWindowTitle("AirDraw — Camera View")
        self.setMinimumWidth(680)
        self.setStyleSheet(
            "QWidget { background: #17191f; color: #f2f3f5; }"
            "QLabel#video { background: #08090c; border: 1px solid #343944; }"
            "QSlider::groove:horizontal { height: 6px; background: #3a3f4b; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #55aaff; width: 14px; margin: -5px 0; border-radius: 7px; }"
            "QPushButton { background: #303644; border: 1px solid #454d5c;"
            " padding: 7px 12px; border-radius: 5px; }"
            "QPushButton:hover { background: #3b4555; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        heading = QLabel("CAMERA FEED")
        heading.setStyleSheet("font-size: 15px; font-weight: 700; letter-spacing: 1px;")
        root.addWidget(heading)

        self.video_label = QLabel("Waiting for camera…")
        self.video_label.setObjectName("video")
        self.video_label.setFixedSize(FRAME_W, FRAME_H)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.status_label = QLabel("Starting camera…")
        self.status_label.setStyleSheet("color: #aeb8c8; font-size: 12px;")
        root.addWidget(self.status_label)

        root.addSpacing(4)
        controls_title = QLabel("IMAGE ADJUSTMENTS")
        controls_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #cbd5e1;")
        root.addWidget(controls_title)

        brightness_row = QHBoxLayout()
        brightness_title = QLabel("Brightness")
        brightness_title.setFixedWidth(90)
        self.brightness_value = QLabel(f"{DEFAULT_BRIGHTNESS:+d}")
        self.brightness_value.setFixedWidth(42)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-50, 100)
        self.brightness_slider.setValue(DEFAULT_BRIGHTNESS)
        brightness_row.addWidget(brightness_title)
        brightness_row.addWidget(self.brightness_slider, 1)
        brightness_row.addWidget(self.brightness_value)
        root.addLayout(brightness_row)

        contrast_row = QHBoxLayout()
        contrast_title = QLabel("Contrast")
        contrast_title.setFixedWidth(90)
        self.contrast_value = QLabel(f"{DEFAULT_CONTRAST:.2f}x")
        self.contrast_value.setFixedWidth(42)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(50, 250)
        self.contrast_slider.setValue(round(DEFAULT_CONTRAST * 100))
        contrast_row.addWidget(contrast_title)
        contrast_row.addWidget(self.contrast_slider, 1)
        contrast_row.addWidget(self.contrast_value)
        root.addLayout(contrast_row)

        footer = QHBoxLayout()
        self.reset_button = QPushButton("Reset image")
        self.reset_button.clicked.connect(self.reset_adjustments)
        footer.addWidget(self.reset_button)
        footer.addStretch(1)
        hint = QLabel("Adjustments affect both the preview and hand tracking")
        hint.setStyleSheet("color: #8993a3; font-size: 11px;")
        footer.addWidget(hint)
        root.addLayout(footer)

        self.brightness_slider.valueChanged.connect(self._brightness_changed)
        self.contrast_slider.valueChanged.connect(self._contrast_changed)
        self._apply_adjustments()

    def _brightness_changed(self, value: int) -> None:
        self.brightness_value.setText(f"{value:+d}")
        self._apply_adjustments()

    def _contrast_changed(self, value: int) -> None:
        self.contrast_value.setText(f"{value / 100.0:.2f}x")
        self._apply_adjustments()

    def _apply_adjustments(self) -> None:
        self.worker.set_image_adjustments(
            self.brightness_slider.value(),
            self.contrast_slider.value(),
        )

    def reset_adjustments(self) -> None:
        self.brightness_slider.setValue(DEFAULT_BRIGHTNESS)
        self.contrast_slider.setValue(round(DEFAULT_CONTRAST * 100))

    def update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)


# ----------------------------- drawing overlay ----------------------------
class Overlay(QWidget):
    def __init__(self, color: str, pen_width: int) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QGuiApplication.primaryScreen()
        geometry = screen.geometry().adjusted(0, 0, 0, -1)
        self.setGeometry(geometry)

        device_pixel_ratio = screen.devicePixelRatio()
        self.canvas = QImage(
            int(geometry.width() * device_pixel_ratio),
            int(geometry.height() * device_pixel_ratio),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        self.canvas.setDevicePixelRatio(device_pixel_ratio)
        self.canvas.fill(Qt.GlobalColor.transparent)

        self.color = QColor(color)
        self.pen_width = pen_width
        self.cursor_pos = None
        self.gesture = "idle"
        self.last = None
        self.pending = []  # points not yet inked (held back by LAG frames)

        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self.force_topmost)
        self._topmost_timer.start(1000)

    def force_topmost(self) -> None:
        if user32 is None or not self.isVisible():
            return
        try:
            user32.SetWindowPos.argtypes = [
                wintypes.HWND, wintypes.HWND,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                wintypes.UINT,
            ]
            # HWND_TOPMOST=-1; NOSIZE|NOMOVE|NOACTIVATE|SHOWWINDOW.
            user32.SetWindowPos(
                int(self.winId()), wintypes.HWND(-1), 0, 0, 0, 0,
                0x0001 | 0x0002 | 0x0010 | 0x0040,
            )
        except Exception:
            pass

    def on_hand(self, nx: float, ny: float, gesture: str) -> None:
        point = QPointF(nx * self.width(), ny * self.height())
        self.cursor_pos, self.gesture = point, gesture
        if gesture == "draw":
            self.pending.append(point)
            if len(self.pending) > LAG:
                q = self.pending.pop(0)
                self._stroke(self.last if self.last is not None else q, q)
                self.last = q
        else:
            # Throw away the unstable tail recorded while the fingers were
            # curling/extending, so strokes don't end in a hook.
            self.pending.clear()
            self.last = None
        self.update()

    def on_lost(self) -> None:
        self.cursor_pos, self.last = None, None
        self.pending.clear()
        self.update()

    def clear(self) -> None:
        self.canvas.fill(Qt.GlobalColor.transparent)
        self.last = None
        self.pending.clear()
        self.update()

    def set_color(self, color: str) -> None:
        self.color = QColor(color)

    def _stroke(self, start: QPointF, end: QPointF) -> None:
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QPen(
                self.color,
                self.pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawLine(start, end)
        painter.end()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, self.canvas)

        cursor = self.cursor_pos
        if cursor is not None:
            radius = self.pen_width / 2 + 3
            if self.gesture == "draw":
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self.color)
                painter.drawEllipse(cursor, radius, radius)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                ring = (
                    QColor(255, 80, 80)
                    if self.gesture == "clear"
                    else QColor(255, 255, 255)
                )
                if self.gesture == "idle":
                    ring.setAlpha(120)
                painter.setPen(QPen(QColor(0, 0, 0, 160), 5))
                painter.drawEllipse(cursor, 14, 14)
                painter.setPen(QPen(ring, 3))
                painter.drawEllipse(cursor, 14, 14)
        painter.end()


# ----------------------------- hotkey bridge ------------------------------
class HotkeyBridge(QObject):
    toggle = pyqtSignal()
    clear = pyqtSignal()
    hide_show = pyqtSignal()
    next_color = pyqtSignal()
    quit = pyqtSignal()


def make_icon(color: str) -> QIcon:
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(QPen(QColor(30, 30, 30), 2))
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()
    return QIcon(pixmap)


# ---------------------------------- main -----------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw over any Windows app with hand gestures"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="camera index (default: 0)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="also show the optional OpenCV debug preview",
    )
    parser.add_argument(
        "--width", type=int, default=6, help="drawing pen width in pixels"
    )
    args = parser.parse_args()

    ensure_model()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    color_index = [0]
    overlay = Overlay(COLORS[0], args.width)
    overlay.show()
    overlay.force_topmost()

    worker = HandWorker(args.camera, args.preview)
    camera_view = CameraView(worker)
    worker.moved.connect(overlay.on_hand)
    worker.lost.connect(overlay.on_lost)
    worker.clear_requested.connect(overlay.clear)
    worker.camera_frame.connect(camera_view.update_frame)
    worker.camera_status.connect(camera_view.set_status)

    # The camera window opens automatically on startup.
    camera_view.show()

    tray = QSystemTrayIcon(make_icon(COLORS[0]), app)
    menu = QMenu()

    def toggle_tracking() -> None:
        if worker.enabled.is_set():
            worker.enabled.clear()
            tray.showMessage("AirDraw", "Hand tracking OFF (camera released)")
            camera_view.set_status("Tracking paused — camera released")
        else:
            worker.enabled.set()
            tray.showMessage("AirDraw", "Hand tracking ON")
            camera_view.set_status("Reconnecting to camera…")

    def toggle_overlay() -> None:
        overlay.setVisible(not overlay.isVisible())
        if overlay.isVisible():
            overlay.force_topmost()

    def show_camera_view() -> None:
        camera_view.show()
        camera_view.raise_()
        camera_view.activateWindow()

    def toggle_camera_view() -> None:
        if camera_view.isVisible():
            camera_view.hide()
        else:
            show_camera_view()

    def cycle_color() -> None:
        color_index[0] = (color_index[0] + 1) % len(COLORS)
        overlay.set_color(COLORS[color_index[0]])
        tray.setIcon(make_icon(COLORS[color_index[0]]))

    def quit_app() -> None:
        worker.running = False
        worker.enabled.set()
        worker.wait(5000)
        keyboard.unhook_all()
        tray.hide()
        camera_view.close()
        overlay.close()
        app.quit()

    actions = [
        ("Toggle tracking  (Ctrl+Alt+D)", toggle_tracking),
        ("Clear drawing  (Ctrl+Alt+C)", overlay.clear),
        ("Hide/show overlay  (Ctrl+Alt+H)", toggle_overlay),
        ("Show/hide camera view", toggle_camera_view),
        ("Next drawing color  (Ctrl+Alt+K)", cycle_color),
        ("Quit  (Ctrl+Alt+Q)", quit_app),
    ]
    for label, callback in actions:
        action = QAction(label, menu)
        action.triggered.connect(callback)
        menu.addAction(action)

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

    worker.camera_error.connect(lambda message: tray.showMessage("AirDraw", message))
    worker.start()
    tray.showMessage(
        "AirDraw",
        "Running. Camera view is open. Ctrl+Alt+D toggles tracking; Ctrl+Alt+Q quits.",
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
