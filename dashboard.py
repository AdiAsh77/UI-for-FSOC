"""
dashboard.py

FSOC Vision Stabilization Console - UI only.
The real Unity + YOLO/Kalman tracking lives in backend.py;
this file wires backend.TrackingThread into the Sim View tab.

Run:      python dashboard.py
Requires: backend.py in the same folder (or on the Python path).
"""

# IMPORTANT: import backend BEFORE any PyQt import. backend.py
# loads cv2/ultralytics (torch) first; if PyQt has already
# touched the DLL search path by then, torch's native DLLs can
# fail to load on Windows. See the note at the top of backend.py.
from backend import (
    TrackingThread,
    TRACK_HOST, TRACK_PORT, MODEL_PATH, CONFIDENCE_THRESHOLD,
    YOLO_INTERVAL, INITIALIZATION_DETECTIONS,
    ACQUISITION_ERROR_X, ACQUISITION_ERROR_Y, TRACK_CAMERA_FPS,
    UNITY_EXE, ACCURACY_THRESHOLD, LOCK_ERROR_THRESHOLD,
)

# Excel + Word Benchmark Performance-1 report generator (kept in its own
# module — see benchmark_report.py). Only needed when Run benchmark is
# actually pressed, so a missing openpyxl/python-docx install doesn't stop
# the rest of the dashboard from starting.
try:
    import benchmark_report
    BENCH_REPORT_ERROR = None
except Exception as _bench_report_exc:
    benchmark_report = None
    BENCH_REPORT_ERROR = _bench_report_exc

# video_tracking (YOLO/torch) must also be imported before PyQt, for the same DLL reason.
import video_tracking
from video_tracking import BeaconTracker

import sys, os, math, random, tempfile, threading, time
import subprocess
from collections import deque

import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, QPoint, QSettings, QObject, QEvent, QUrl, QPointF, QRectF, QSize, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QImage, QRadialGradient, QPolygonF, QFont, QFontMetrics,
    QIcon, QPixmap, QPalette, QPainterPath, QLinearGradient, QAction, QDesktopServices)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QStackedWidget, QSlider, QComboBox, QLineEdit, QCheckBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QHeaderView, QAbstractItemView, QButtonGroup, QScrollArea, QSizePolicy, QProxyStyle, QStyle, QPlainTextEdit,
    QTabWidget, QMenu)

import win32gui  # type: ignore
import win32con  # type: ignore
import win32process  # type: ignore
import glob
from config import scener
# Single source of truth for the scenes folder — used by both the Scene-Parameters
# tab (build_scene_params/save_scene_settings write a .txt file here for every Save)
# and the Sim View tab (build_sim/refresh_scene_list reads the .txt file names here
# to populate the "Test case scene" dropdown). Change only this constant to move it.
SCENES_DIR = scener

# Where Run benchmark writes its Excel + Word deliverables — a folder
# next to this script, created on first use.
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_reports')

# --- FSOC Track Lab console theme ---------------------------------------
# Palette shape/roles mirror the fsoc-track-lab.vercel.app telemetry console
# (dark navy canvas/cards, always-dark sidebar rail, mono-leaning status
# pills in green/amber/red) but the signature accent is swapped from the
# site's cyan to violet, so links/highlights/buttons read as this app's own.
LIGHT = dict(bg='#f1f3f7', pn='#ffffff', ink='#10151f', mut='#5c6675', ln='#dde2ea', sd='#1f2b45', bc='#dc2626', ok='#0a8f66',
             wn='#b2560c', ac='#0f7ea8', ach='#0c6a8c', acd='#095671', onac='#ffffff', soft='#e7f1f7')
DARK = dict(bg='#090c12', pn='#101826', ink='#edf0f5', mut='#8791a3', ln='#202939', sd='#080b12', bc='#ef4444', ok='#1cb37e',
            wn='#e39a3d', ac='#2ea8e8', ach='#1c8ecf', acd='#136694', onac='#ffffff', soft='#132435')
T = dict(DARK)

THEME_MODES = ('light', 'dark', 'system')

def system_prefers_dark():
    """True if the operating system is currently set to dark mode. Uses Qt's own
    colour-scheme hint (Qt 6.5+); on older Qt falls back to the Windows registry
    value 'AppsUseLightTheme'. Defaults to light if neither is available."""
    try:
        cs = QApplication.styleHints().colorScheme()
        if cs == Qt.ColorScheme.Dark: return True
        if cs == Qt.ColorScheme.Light: return False
    except AttributeError:
        pass
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize') as k:
                return winreg.QueryValueEx(k, 'AppsUseLightTheme')[0] == 0
        except OSError:
            pass
    return False

def _pen(color, w):
    p = QPen(QColor(color), w); p.setCapStyle(Qt.PenCapStyle.RoundCap); p.setJoinStyle(Qt.PenJoinStyle.RoundJoin); return p

def _pix(size):
    pm = QPixmap(size * 2, size * 2); pm.setDevicePixelRatio(2); pm.fill(Qt.GlobalColor.transparent); return pm

def asset(name, draw, size):
    """Render a small PNG (checkbox tick, combo chevron) so QSS can reference it."""
    pm = _pix(size); q = QPainter(pm); q.setRenderHint(QPainter.RenderHint.Antialiasing); draw(q); q.end()
    p = os.path.join(tempfile.gettempdir(), f"fsoc_{name}.png"); pm.save(p); return p.replace('\\', '/')

def _tick(q):
    q.setPen(_pen(T['onac'], 2.2)); q.drawPolyline(QPolygonF([QPointF(3.5, 8.5), QPointF(6.8, 11.6), QPointF(12.6, 4.8)]))

def _chev(q):
    q.setPen(_pen(T['mut'], 1.8)); q.drawPolyline(QPolygonF([QPointF(3.5, 4.5), QPointF(7, 8), QPointF(10.5, 4.5)]))

def draw_icon(kind, color, size=26):
    pm = _pix(size); q = QPainter(pm); q.setRenderHint(QPainter.RenderHint.Antialiasing)
    q.setPen(_pen(color, 1.9)); q.setBrush(Qt.BrushStyle.NoBrush); c = size / 2; fill = QColor(color)
    if kind == 'sim':        # planet with an orbiting satellite
        q.drawEllipse(QPointF(c, c), 4.2, 4.2); q.save(); q.translate(c, c); q.rotate(-28)
        q.drawEllipse(QPointF(0, 0), 11, 4.6); q.setBrush(fill); q.drawEllipse(QPointF(11, 0), 1.9, 1.9); q.restore()
    elif kind == 'video':    # screen with play triangle
        q.drawRoundedRect(QRectF(3, 6, size - 6, size - 12), 3.5, 3.5); q.setBrush(fill)
        q.drawPolygon(QPolygonF([QPointF(c - 2.4, c - 3.6), QPointF(c - 2.4, c + 3.6), QPointF(c + 3.8, c)]))
    elif kind == 'bench':    # gauge
        q.drawArc(QRectF(3.5, 5, size - 7, size - 7), 200 * 16, -220 * 16)
        q.drawLine(QPointF(c, c + 3), QPointF(c + 4.6, c - 4.4)); q.setBrush(fill); q.drawEllipse(QPointF(c, c + 3), 1.7, 1.7)
    elif kind == 'data':     # clipboard with a check
        q.drawRoundedRect(QRectF(5.5, 5, size - 11, size - 8), 3, 3); q.drawRoundedRect(QRectF(c - 3.4, 3, 6.8, 4.2), 1.6, 1.6)
        q.drawPolyline(QPolygonF([QPointF(c - 3.6, c + 2), QPointF(c - .8, c + 4.8), QPointF(c + 4, c - .6)]))
    elif kind == 'info':     # open book / spec sheet
        q.drawEllipse(QPointF(c, c), 8.5, 8.5); q.setBrush(fill); q.drawEllipse(QPointF(c, c - 4.2), 1.15, 1.15)
        q.drawLine(QPointF(c, c - 1.2), QPointF(c, c + 4.6))
    elif kind == 'scene':    # parameter sliders (three horizontal tracks with knobs)
        for y, kx in ((c - 7, 9), (c, 17), (c + 7, 12)):
            q.drawLine(QPointF(4, y), QPointF(size - 4, y))
            q.setBrush(fill); q.drawEllipse(QPointF(kx, y), 2.6, 2.6); q.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == 'beacon':    # Scene-Parameters "Beacon Settings" heading: signal source + radiating waves
        q.setBrush(fill); q.drawEllipse(QPointF(c, c + c * .44), c * .24, c * .24); q.setBrush(Qt.BrushStyle.NoBrush)
        q.drawArc(QRectF(c - c * .6, c - c * .6, c * 1.2, c * 1.2), 35 * 16, 110 * 16)
        q.drawArc(QRectF(c - c * .94, c - c * .94, c * 1.88, c * 1.88), 35 * 16, 110 * 16)
    elif kind == 'satellite':  # Scene-Parameters "Satellite Settings" heading: body, two panels, antenna
        b = c * .29
        q.drawRect(QRectF(c - b, c - b, 2 * b, 2 * b))
        q.drawRect(QRectF(c - c * .95, c - b * .8, c * .5, b * 1.6)); q.drawRect(QRectF(c + c * .45, c - b * .8, c * .5, b * 1.6))
        q.drawLine(QPointF(c - c * .45, c), QPointF(c - b, c)); q.drawLine(QPointF(c + b, c), QPointF(c + c * .45, c))
        q.drawLine(QPointF(c + b * .6, c - b), QPointF(c + c * .5, c - c * .62))
        q.setBrush(fill); q.drawEllipse(QPointF(c + c * .5, c - c * .62), c * .11, c * .11)
    elif kind == 'disturbance':   # Scene-Parameters "Disturbances" heading: jagged noise wave
        q.drawPolyline(QPolygonF([QPointF(size * .14, c), QPointF(size * .33, c - c * .56), QPointF(size * .5, c + c * .56),
                                   QPointF(size * .67, c - c * .56), QPointF(size * .86, c)]))
    elif kind == 'camera':     # Scene-Parameters "Camera Settings" heading: body, viewfinder bump, lens
        q.drawRoundedRect(QRectF(size * .14, c - c * .44, size * .72, c * .88), c * .24, c * .24)
        q.drawRect(QRectF(c - c * .27, c - c * .71, c * .54, c * .29))
        q.setBrush(Qt.BrushStyle.NoBrush); q.drawEllipse(QPointF(c, c), c * .29, c * .29)
    else:                    # theme: half-filled circle
        q.drawEllipse(QPointF(c, c), 8, 8); q.setBrush(fill); q.drawPie(QRectF(c - 8, c - 8, 16, 16), 90 * 16, 180 * 16)
    q.end(); return pm

def make_icon(kind):
    ic = QIcon()
    ic.addPixmap(draw_icon(kind, '#7f8fac'), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(draw_icon(kind, '#ffffff'), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(draw_icon(kind, '#ffffff'), QIcon.Mode.Active, QIcon.State.Off)
    return ic

def beacon_logo():
    pm = _pix(30); q = QPainter(pm); q.setRenderHint(QPainter.RenderHint.Antialiasing)
    g = QRadialGradient(15, 15, 14); g.setColorAt(0, QColor('#ffffff')); g.setColorAt(.28, QColor('#ff5a5f')); g.setColorAt(1, QColor(255, 90, 95, 0))
    q.setPen(Qt.PenStyle.NoPen); q.setBrush(QBrush(g)); q.drawEllipse(QPointF(15, 15), 14, 14); q.end(); return pm

def qss():
    t = T
    return f"""
    QWidget {{ color:{t['ink']}; }}
    QMainWindow, QStackedWidget, QWidget#page {{ background:{t['bg']}; }}
    QFrame#panel {{ background:{t['pn']}; border:1px solid {t['ln']}; border-radius:10px; }}
    QLabel {{ background:transparent; }}
    QLabel#h1 {{ font-size:21px; font-weight:800; }} QLabel#h3 {{ font-size:13px; font-weight:700; }} QLabel#panelTitle {{ font-size:19px; font-weight:800; }}
    QLabel#mut {{ color:{t['mut']}; }}
    QLabel#soon {{ color:{t['mut']}; font-size:11px; font-style:italic; }}
    QFrame#nav {{ background:{t['sd']}; border-right:1px solid rgba(255,255,255,.06); }}
    QPushButton#nb {{ background:transparent; border:0; border-radius:9px; padding:0; }}
    QPushButton#nb:hover {{ background:rgba(255,255,255,.08); }}
    QPushButton#nb:checked {{ background:{t['ac']}; }}
    QMenu {{ background:{t['pn']}; border:1px solid {t['ln']}; border-radius:8px; padding:5px; }}
    QMenu::item {{ padding:7px 22px 7px 10px; border-radius:5px; color:{t['ink']}; }}
    QMenu::item:selected {{ background:{t['soft']}; }}
    QToolTip {{ background:{t['sd']}; color:#ffffff; border:1px solid {t['ln']}; border-radius:6px; padding:5px 9px; }}
    QPushButton {{ background:{t['ac']}; color:{t['onac']}; border:0; border-radius:7px; padding:8px 16px; font-weight:700; }}
    QPushButton#gaugeBtn {{ padding:0; font-size:17px; font-weight:800; }}
    QPushButton:hover {{ background:{t['ach']}; }}
    QPushButton:pressed {{ background:{t['acd']}; }}
    QPushButton:disabled {{ background:{t['ln']}; color:{t['mut']}; }}
    QPushButton[o="true"] {{ background:transparent; color:{t['ink']}; border:1px solid {t['ln']}; }}
    QPushButton[o="true"]:hover {{ background:{t['soft']}; border-color:{t['ac']}; color:{t['ac']}; }}
    QPushButton[o="true"]:pressed {{ background:{t['ln']}; }}
    QPushButton[o="true"]:disabled {{ background:transparent; color:{t['mut']}; border-color:{t['ln']}; }}
    QLineEdit, QComboBox {{ background:{t['soft']}; border:1px solid {t['ln']}; border-radius:7px; padding:6px 10px; min-height:20px; }}
    QLineEdit:hover, QComboBox:hover {{ border-color:{t['ac']}; }}
    QLineEdit:focus, QComboBox:focus {{ border:1.5px solid {t['ac']}; background:{t['pn']}; }}
    QLineEdit:disabled, QComboBox:disabled {{ color:{t['mut']}; background:transparent; }}
    QComboBox::drop-down {{ border:0; width:26px; }}
    QComboBox::down-arrow {{ image:url({t['chev']}); width:12px; height:12px; }}
    QComboBox QAbstractItemView {{ background:{t['pn']}; border:1px solid {t['ln']}; border-radius:8px; selection-background-color:{t['soft']}; selection-color:{t['ink']}; padding:4px; margin:0; outline:0; }}
    QComboBox QAbstractItemView::item {{ padding:6px 10px; margin:0; border:0; min-height:0; border-radius:5px; }}
    QComboBox QAbstractItemView::item:hover {{ background:{t['soft']}; color:{t['ink']}; }}
    QComboBox QAbstractItemView QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
    QComboBox QAbstractItemView QScrollBar::handle:vertical {{ background:{t['ln']}; border-radius:4px; min-height:24px; }}
    QComboBox QAbstractItemView QScrollBar::add-line:vertical, QComboBox QAbstractItemView QScrollBar::sub-line:vertical {{ height:0; }}
    QTableWidget {{ background:transparent; alternate-background-color:{t['soft']}; border:0; gridline-color:{t['ln']}; }}
    QTableWidget::item {{ padding:4px 8px; }}
    QTableWidget::item:selected {{ background:{t['soft']}; color:{t['ink']}; }}
    QHeaderView::section {{ background:transparent; color:{t['mut']}; font-weight:600; border:0; border-bottom:1px solid {t['ln']}; padding:6px 8px; }}
    QListWidget {{ background:transparent; border:0; outline:0; }}
    QListWidget::item {{ padding:10px; border-radius:8px; margin-bottom:2px; }}
    QListWidget::item:hover {{ background:{t['soft']}; }}
    QListWidget::item:selected {{ background:{t['soft']}; color:{t['ink']}; border:1px solid {t['ln']}; }}
    QTabWidget::pane {{ background:{t['pn']}; border:1px solid {t['ln']}; border-radius:10px; top:-1px; }}
    QTabWidget::tab-bar {{ alignment:center; }}
    QTabBar::tab {{ background:transparent; color:{t['mut']}; padding:9px 22px; margin:0 2px; font-weight:700;
                    border:none; border-bottom:2px solid transparent; }}
    QTabBar::tab:selected {{ color:{t['ac']}; border-bottom:2px solid {t['ac']}; }}
    QTabBar::tab:hover:!selected {{ color:{t['ink']}; background:{t['soft']}; border-top-left-radius:6px; border-top-right-radius:6px; }}
    QPlainTextEdit#evl {{ background:transparent; border:0; padding:0; color:{t['mut']}; }}
    QProgressBar {{ background:{t['soft']}; border:1px solid {t['ln']}; border-radius:7px; min-height:16px; text-align:center; font-weight:700; }}
    QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {t['ac']}, stop:1 {t['ach']}); border-radius:6px; }}
    QSlider::groove:horizontal {{ height:5px; background:{t['ln']}; border-radius:2px; }}
    QSlider::sub-page:horizontal {{ background:{t['ac']}; border-radius:2px; }}
    QSlider::handle:horizontal {{ background:{t['ac']}; width:15px; height:15px; margin:-5px 0; border-radius:8px; border:2px solid {t['pn']}; }}
    QSlider::handle:horizontal:hover {{ background:{t['ach']}; }}
    QCheckBox {{ spacing:10px; }}
    QCheckBox::indicator {{ width:16px; height:16px; border:1.5px solid {t['mut']}; border-radius:4px; background:{t['pn']}; }}
    QCheckBox::indicator:hover {{ border-color:{t['ac']}; }}
    QCheckBox::indicator:checked {{ background:{t['ac']}; border-color:{t['ac']}; image:url({t['tick']}); }}
    QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
    QScrollBar::handle:vertical {{ background:{t['ln']}; border-radius:4px; min-height:24px; }}
    QScrollBar::handle:vertical:hover {{ background:{t['mut']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
    QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
    QScrollBar::handle:horizontal {{ background:{t['ln']}; border-radius:4px; min-width:24px; }}
    QScrollBar::handle:horizontal:hover {{ background:{t['mut']}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
    QScrollArea {{ background:transparent; border:0; }}
    QScrollArea > QWidget > QWidget {{ background:transparent; }}
    """

def gauss():
    return (sum(random.random() for _ in range(4)) - 2) * 1.7

_r = random.Random(7)
STARS = [(_r.random() * 320, _r.random() * 240, .3 + _r.random() * .7) for _ in range(70)]

# --------------------------------------------------------------------------- simulation
class Sim:
    """Stand-in for the closed loop: disturbance -> detector -> Kalman -> pan/tilt controller."""
    def __init__(s):
        s.p = dict(noise=0, blur=0, atm=0, sd=0, jit=0, plat=0)
        s.type, s.running = 'Mixed', True
        s.reset()

    def reset(s):
        s.t = 0.; s.pan = -70.; s.tilt = 45.
        s.kx = s.ky = s.vx = s.vy = s.wx = s.wy = 0.
        s.ox = s.oy = s.mx = s.my = s.dy = 0.
        s.det, s.lock, s.started = True, False, False
        s.fr = s.lk = s.ac = s.loss = 0
        s.err, s.acq, s.events, s.lost_at = [], [], [], 0.

    def log(s, m):
        s.events.insert(0, f"{s.t:.1f}s   {m}"); del s.events[200:]

    def step(s):
        p = s.p; s.t += 1 / 30; t = s.t; a = 40 * p['plat'] + 4
        if s.type == 'Linear':
            dx, dy = a * ((t * .15 % 2) - 1), .2 * a * math.sin(.4 * t)
        elif s.type == 'Orbital':
            dx, dy = a * math.cos(.8 * t), .7 * a * math.sin(.8 * t)
        else:
            dx = .6 * a * math.cos(.8 * t) + .4 * a * math.sin(2.3 * t)
            dy = .5 * a * math.sin(.8 * t) + .3 * a * math.cos(1.7 * t)
        s.wx = s.wx * .97 + gauss() * p['atm'] * 1.5          # atmospheric turbulence
        s.wy = s.wy * .97 + gauss() * p['atm'] * 1.5
        s.dy = dy
        s.ox = dx + s.wx - s.pan + gauss() * p['jit']          # true beacon offset in camera frame
        s.oy = dy + s.wy - s.tilt + gauss() * p['jit']
        sig = p['noise'] * p['sd'] * 3 + .3
        s.det = (random.random() > .004 + p['atm'] * p['noise'] * .06) and abs(s.ox) < 150 and abs(s.oy) < 110
        s.mx, s.my = s.ox + gauss() * sig, s.oy + gauss() * sig  # detector output
        s.kx += s.vx; s.ky += s.vy                               # alpha-beta Kalman
        if s.det:
            rx, ry = s.mx - s.kx, s.my - s.ky
            s.kx += .55 * rx; s.ky += .55 * ry; s.vx += .15 * rx; s.vy += .15 * ry
        s.pan += .4 * s.kx; s.tilt += .4 * s.ky                  # pan/tilt controller
        s.kx *= .6; s.ky *= .6
        e = math.hypot(s.ox, s.oy); lk = e < 60 and s.det
        if lk and not s.lock:
            s.acq.append(t - s.lost_at); s.started = True
            s.log('Lock reacquired' if len(s.acq) > 1 else 'Beacon acquired'); s.lock = True
        elif not lk and s.lock:
            s.lock = False; s.lost_at = t; s.loss += 1; s.log('Lock lost')
        if s.started:
            s.fr += 1; s.lk += lk; s.ac += e < 12
        s.err.append(e); del s.err[:-300]

    retention = property(lambda s: s.lk / s.fr * 100 if s.fr else 0)
    loss_pct = property(lambda s: s.loss / s.fr * 100 if s.fr else 0)
    accuracy = property(lambda s: s.ac / s.fr * 100 if s.fr else 0)
    rmse = property(lambda s: math.sqrt(sum(e * e for e in s.err) / len(s.err)) if s.err else 0)
    acq_avg = property(lambda s: sum(s.acq) / len(s.acq) if s.acq else None)


class Video:
    """State for the imported MP4. The detection itself (YOLO + Kalman, video_tracking.BeaconTracker)
    runs in VideoWorker; this just holds the latest frame / metrics for the Video View widgets."""
    def __init__(s):
        s.worker = None; s.active = False; s.paused = False; s.reset()

    def reset(s):
        s.frame = s.raw = None; s.t = 0.; s.det = False; s.info = {}
        s.err, s.rows, s.n = [], [], 0

    def update(s, img, raw, info):
        s.frame, s.raw, s.info = img, raw, info
        s.t, s.det = info['video_time'], info['detected']
        s.n += 1
        if info['acquisition_complete']:
            s.err.append(info['error']); del s.err[:-300]
        if s.n % 6 == 0:
            bo = info['beacon_offset']
            if not info['acquisition_complete']: state = 'Acquiring' if s.det else 'Searching'
            else: state = 'Tracking' if s.det else 'Target lost'
            s.rows.insert(0, (f"{s.t:.1f}s", f"{bo[0]:.0f}" if bo else '-', f"{bo[1]:.0f}" if bo else '-', state))
            del s.rows[5:]


class VideoWorker(QThread):
    """Reads the MP4 and runs video_tracking.BeaconTracker on every frame, off the GUI thread.
    Pause clears the play event, which stops BOTH the video and the detection; Play resumes both."""
    result = pyqtSignal(object, object, object)   # (annotated QImage, raw QImage, info dict incl. the annotated cv2 frame)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    ended = pyqtSignal(str)   # message about the saved tracking CSV

    def __init__(s, path):
        super().__init__(); s.path = path; s.speed = 1.0
        s._play = threading.Event(); s._play.set()
        s._quit = False; s._seek = None

    def play(s): s._play.set()
    def pause(s): s._play.clear()
    def seek_to(s, frac): s._seek = frac
    def stop(s): s._quit = True; s._play.set()

    @staticmethod
    def _qimage(bgr, w, h):
        out = cv2.resize(bgr, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA) if w > 640 else bgr
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888).copy()

    def run(s):
        cap = cv2.VideoCapture(s.path)
        if not cap.isOpened():
            s.failed.emit('Could not open the video.'); return
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0: fps = video_tracking.TRACK_CAMERA_FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            s.status.emit('Loading YOLO model...')
            tracker = BeaconTracker(w, h, fps)
        except Exception as e:
            cap.release(); s.failed.emit(f'Could not load the detector: {e}'); return
        s.status.emit('Detection running on the imported video (YOLO + Kalman).')
        next_t = time.perf_counter()
        csv_rows = []
        # ---- Reacquisition timing ----
        # video_tracking.BeaconTracker counts target losses (target_loss_count) but
        # doesn't time how long each one takes to recover from, so that's tracked here
        # instead, using the same signal update_video_rail() already reads per frame:
        # info['target_loss_count'] rising means a loss just started; info['detected']
        # going back to True (after a loss was in progress) means it's reacquired.
        prev_loss_count = 0
        loss_start_time = None
        reacquisition_times = []
        while not s._quit:
            frac, s._seek = s._seek, None
            if frac is not None:
                # Seek: restart tracking from the new position and show one frame even if paused.
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * total) if total > 0 else 0)
                tracker.reset(); csv_rows = []; next_t = time.perf_counter()
                prev_loss_count = 0; loss_start_time = None; reacquisition_times = []
            elif not s._play.is_set():
                s._play.wait(0.05); next_t = time.perf_counter(); continue
            ok, frame = cap.read()
            if not ok:
                # End of video: pause, rewind to the start (tracking restarts on the next Play).
                # The tracking CSV is written when the video has played through.
                try: msg = f"Tracking data saved to {video_tracking.write_csv(csv_rows)}"
                except Exception as e: msg = f"Could not save tracking CSV: {e}"
                s._play.clear(); s._seek = 0.0; s.ended.emit(msg); continue
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            raw_img = s._qimage(frame, w, h)                 # untouched copy: process() draws on `frame`
            info = tracker.process(frame)
            # tracker.acquisition_time is an attribute on the tracker (set once, when
            # acquisition first completes), not one of the per-frame dict keys process()
            # returns, so it's copied across explicitly here.
            info['acquisition_time'] = tracker.acquisition_time
            if info['target_loss_count'] > prev_loss_count:
                loss_start_time = info['time']
                prev_loss_count = info['target_loss_count']
            elif loss_start_time is not None and info['detected']:
                reacquisition_times.append(info['time'] - loss_start_time)
                loss_start_time = None
            info['avg_reacquisition_time'] = (
                sum(reacquisition_times) / len(reacquisition_times) if reacquisition_times else None
            )
            csv_rows.append(video_tracking.csv_row(info))
            info.update(video_time=idx / fps, frame_index=idx, total=total, pos_frac=idx / total if total > 0 else 0.)
            s.result.emit(s._qimage(frame, w, h), raw_img, info)
            # pace to the video's own frame rate (x speed); if detection is slower, just run flat out
            next_t += 1 / (fps * s.speed); d = next_t - time.perf_counter()
            if d > 0: time.sleep(d)
            elif d < -1: next_t = time.perf_counter()
        cap.release()

# --------------------------------------------------------------------------- custom widgets
def _canvas_bg(q, w, h):
    path = QPainterPath(); path.addRoundedRect(QRectF(0, 0, w, h), 9, 9); q.setClipPath(path); q.fillRect(0, 0, w, h, QColor('#070c18'))

class CamView(QWidget):
    """Video View camera panel. mode 'py': the annotated frame from the video detector (YOLO box,
    Kalman estimate, etc.); mode 'raw': the plain imported mp4. Blank prompt when nothing is imported."""
    def __init__(s, video):
        super().__init__(); s.video = video; s.mode = 'py'; s.setMinimumSize(210, 150)

    def paintEvent(s, _):
        q = QPainter(s); q.setRenderHint(QPainter.RenderHint.Antialiasing); q.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = s.width(), s.height(); _canvas_bg(q, w, h); k = min(w / 320, h / 240)
        q.translate((w - 320 * k) / 2, (h - 240 * k) / 2); q.scale(k, k); q.setClipRect(QRectF(0, 0, 320, 240), Qt.ClipOperation.IntersectClip)
        v = s.video
        img = (v.raw if s.mode == 'raw' else v.frame) if v and v.active else None   # 'raw' = plain mp4, nothing drawn on it
        if img is not None:
            iw, ih = img.width(), img.height(); r = min(320 / iw, 240 / ih); dw, dh = iw * r, ih * r
            q.drawImage(QRectF((320 - dw) / 2, (240 - dh) / 2, dw, dh), img)
            return
        q.setPen(QColor('#8fa3c7')); q.drawText(QRectF(0, 0, 320, 240), Qt.AlignmentFlag.AlignCenter,
                                                 'Loading video...' if v and v.active else 'No video imported')


class SatView(QWidget):
    """Unity-style scene: sender camera cone tracking a receiver, laser beam between them."""
    def __init__(s, provider):
        super().__init__(); s.provider = provider; s.setMinimumSize(230, 140)

    def paintEvent(s, _):
        pan, tilt, dy, res = s.provider()
        q = QPainter(s); q.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = s.width(), s.height(); _canvas_bg(q, w, h); k = min(w / 480, h / 270)
        q.translate((w - 480 * k) / 2, (h - 270 * k) / 2); q.scale(k, k); q.setClipRect(QRectF(0, 0, 480, 270), Qt.ClipOperation.IntersectClip)
        sy = 130; ry = sy + dy * .8
        g = QRadialGradient(240, 440, 240); g.setColorAt(.6, QColor('#0e2a55')); g.setColorAt(1, QColor('#1d5fa8'))
        q.setPen(Qt.PenStyle.NoPen); q.setBrush(QBrush(g)); q.drawEllipse(QPointF(240, 440), 230, 230)
        q.setBrush(QColor('#9fb3d9'))
        for sx, sy2, _a in STARS: q.drawRect(QRectF(sx * 1.5, sy2 * 1.1, 1.3, 1.3))
        an = math.atan2(ry - sy, 348) - res * .0025
        cone = QPolygonF([QPointF(66, sy), QPointF(66 + 420 * math.cos(an - .13), sy + 420 * math.sin(an - .13)),
                          QPointF(66 + 420 * math.cos(an + .13), sy + 420 * math.sin(an + .13))])
        q.setBrush(QColor(245, 185, 66, 42)); q.drawPolygon(cone)
        q.setPen(QPen(QColor('#ff5a5f'), 1.5, Qt.PenStyle.DashLine)); q.drawLine(QPointF(414, ry), QPointF(70, sy))
        for cx, cy, c in ((57, sy, '#cfd8ea'), (405, ry, '#8ea0c4')):
            q.setPen(Qt.PenStyle.NoPen); q.setBrush(QColor(c)); q.drawRect(QRectF(cx - 9, cy - 8, 36, 16))
            q.setBrush(QColor('#2f5bea')); q.drawRect(QRectF(cx - 21, cy - 5, 12, 10)); q.drawRect(QRectF(cx + 27, cy - 5, 12, 10))
        q.setPen(QColor('#cfd8ea')); q.setFont(QFont('Sans', 8))
        q.drawText(QPointF(20, sy + 38), 'Sender camera'); q.drawText(QPointF(345, ry - 24), 'Receiver beacon')
        q.drawText(QPointF(14, 256), f"Pan {pan * .05:.1f}°   Tilt {tilt * .05:.1f}°")


class Chart(QWidget):
    def __init__(s, provider, ymax, thresh=0, auto=False):
        super().__init__(); s.provider, s.ymax, s.thresh, s.auto = provider, ymax, thresh, auto; s.setMinimumHeight(110)

    def _ymax(s, a):
        """Fixed ymax, or (auto) the largest value on screen rounded up to a tidy number."""
        if not s.auto or not a: return s.ymax
        m = max(max(a) * 1.15, 5)
        e = 10 ** math.floor(math.log10(m))
        return next(k * e for k in (1, 2, 2.5, 5, 10) if m <= k * e)

    def paintEvent(s, _):
        q = QPainter(s); q.setRenderHint(QPainter.RenderHint.Antialiasing); w, h = s.width(), s.height()
        path = QPainterPath(); path.addRoundedRect(QRectF(0, 0, w, h), 9, 9); q.setClipPath(path); q.fillRect(0, 0, w, h, QColor(T['soft']))
        f = q.font(); f.setPointSize(8); q.setFont(f)
        a = s.provider(); ymax = s._ymax(a); fmt = '.1f' if ymax < 10 else '.0f'
        for i in range(1, 4):
            y = h * i // 4; q.setPen(QPen(QColor(T['ln']), 1)); q.drawLine(0, y, w, y)
            q.setPen(QColor(T['mut'])); q.drawText(QPointF(8, y - 3), f"{ymax * (1 - i / 4):{fmt}}")
        if s.thresh:
            y = h - s.thresh / ymax * h; q.setPen(QPen(QColor(T['wn']), 1, Qt.PenStyle.DashLine)); q.drawLine(QPointF(0, y), QPointF(w, y))
            q.setPen(QColor(T['wn'])); q.drawText(QPointF(w - 64, y - 4), 'lock zone')
        if len(a) > 1:
            pts = [QPointF(i / 299 * w, h - min(v, ymax) / ymax * h) for i, v in enumerate(a)]
            line = QPainterPath(pts[0])
            for p in pts[1:]: line.lineTo(p)
            fill = QPainterPath(line); fill.lineTo(pts[-1].x(), h); fill.lineTo(pts[0].x(), h); fill.closeSubpath()
            g = QLinearGradient(0, 0, 0, h); c1 = QColor(T['ac']); c1.setAlpha(90); c2 = QColor(T['ac']); c2.setAlpha(0); g.setColorAt(0, c1); g.setColorAt(1, c2)
            q.fillPath(fill, QBrush(g)); q.setPen(_pen(T['ac'], 1.8)); q.setBrush(Qt.BrushStyle.NoBrush); q.drawPath(line)

class BarChart(QWidget):
    """Vertical bar chart for comparing a metric across scenarios/runs. Bars are colour-coded
    green/red against a spec threshold when one is supplied (SPEC_THRESH), else drawn in the
    accent colour. `provider` returns (labels, values)."""
    def __init__(s, provider, thresh=None, fmt='{:.1f}'):
        super().__init__(); s.provider, s.thresh, s.fmt = provider, thresh, fmt; s.setMinimumHeight(150)

    def paintEvent(s, _):
        q = QPainter(s); q.setRenderHint(QPainter.RenderHint.Antialiasing); w, h = s.width(), s.height()
        path = QPainterPath(); path.addRoundedRect(QRectF(0, 0, w, h), 9, 9); q.setClipPath(path); q.fillRect(0, 0, w, h, QColor(T['soft']))
        labels, values = s.provider()
        if not values: q.end(); return
        top, bot, lp = 10, 30, 8
        vmax = max(max(values) * 1.15, (s.thresh[0] if s.thresh else 0) * 1.15, 1e-6)
        n = len(values); bw = max((w - 2 * lp) / n * .58, 3); gap = (w - 2 * lp) / n
        f = q.font(); f.setPointSize(7); q.setFont(f)
        if s.thresh:
            lim, lower_better = s.thresh; y = h - bot - lim / vmax * (h - top - bot)
            q.setPen(QPen(QColor(T['wn']), 1, Qt.PenStyle.DashLine)); q.drawLine(QPointF(lp, y), QPointF(w - lp, y))
            q.setPen(QColor(T['wn'])); q.drawText(QPointF(w - lp - 74, y - 4), f"spec limit {lim:g}")
        for i, v in enumerate(values):
            x = lp + gap * i + (gap - bw) / 2; bh = max(v, 0) / vmax * (h - top - bot); y = h - bot - bh
            ok = True
            if s.thresh:
                lim, lower_better = s.thresh; ok = (v <= lim) if lower_better else (v >= lim)
            q.setPen(Qt.PenStyle.NoPen); q.setBrush(QColor(T['ok'] if (s.thresh and ok) else (T['bc'] if s.thresh else T['ac'])))
            q.drawRoundedRect(QRectF(x, y, bw, bh), 3, 3)
            q.setPen(QColor(T['ink'])); q.drawText(QRectF(x - 10, y - 14, bw + 20, 12), Qt.AlignmentFlag.AlignHCenter, s.fmt.format(v))
            q.setPen(QColor(T['mut'])); lab = labels[i] if i < len(labels) else ''
            if len(lab) > 10: lab = lab[:9] + '…'
            q.save(); q.translate(x + bw / 2, h - bot + 4); q.rotate(-40); q.drawText(QPointF(-len(lab) * 3, 8), lab); q.restore()


class WeightBar(QWidget):
    """Horizontal stacked bar showing the SIH evaluation-stage weightages, with a legend."""
    def __init__(s, segments):
        super().__init__(); s.segments = segments; s.setMinimumHeight(74)

    def paintEvent(s, _):
        q = QPainter(s); q.setRenderHint(QPainter.RenderHint.Antialiasing); w = s.width(); bh = 26
        x = 0; total = sum(v for _, v, _, _ in s.segments)
        for name, v, color, _ in s.segments:
            bw = w * v / total; q.setPen(Qt.PenStyle.NoPen); q.setBrush(QColor(color)); q.drawRect(QRectF(x, 0, bw, bh))
            q.setPen(QColor('#ffffff')); f = q.font(); f.setBold(True); f.setPointSize(9); q.setFont(f)
            if bw > 34: q.drawText(QRectF(x, 0, bw, bh), Qt.AlignmentFlag.AlignCenter, f"{v}%")
            x += bw
        y = bh + 14; f = q.font(); f.setBold(False); f.setPointSize(8); q.setFont(f); lx = 0
        for name, v, color, _ in s.segments:
            q.setPen(Qt.PenStyle.NoPen); q.setBrush(QColor(color)); q.drawRoundedRect(QRectF(lx, y, 10, 10), 2, 2)
            q.setPen(QColor(T['ink'])); tw = q.fontMetrics().horizontalAdvance(name) + 24
            q.drawText(QPointF(lx + 14, y + 9), name); lx += tw
            if lx > w - 80: lx = 0; y += 18

# --------------------------------------------------------------------------- data
# Row layout: (scenario, acquisition, tracking error, target loss, speed, update, lock retention, RMSE)
# — one numeric value per CR column below, in the same order.
SC = [('Static beacon, near', .15, 2.8, 0, 95, 10.5, 94.2, 2.8), ('Moving, far', .78, 7.2, 1, 88, 11.4, 81.5, 7.2),
      ('Crowded environment', 1.12, 11.5, 3, 72, 13.9, 76.1, 11.5), ('Occlusion event', .35, 5.1, 2, 90, 11.1, 88, 5.1),
      ('Low light', .95, 9.8, 2, 82, 12.2, 79.3, 9.8), ('Multi-beacon', .6, 6.5, 1, 78, 12.8, 84.7, 6.5),
      ('Varying angle', .28, 4, 0, 92, 10.9, 90.2, 4), ('High speed', .52, 8.9, 2, 110, 9.1, 86.8, 8.9),
      ('Intermittent signal', 1.45, 14.1, 5, 85, 11.8, 74, 14.1), ('Complex path', .4, 4.8, 1, 80, 12.5, 89.5, 4.8)]
# AP@50-95 (%) has been retired — no ground-truth IoU box is available from the Unity
# feed, so it was only ever a proxy. Replaced with two criteria pulled straight off
# backend.py's own real per-frame metrics (the same "RMSE" and "Lock retention rate"
# PS #26169's Benchmark Performance-2 evaluation table names explicitly): lock
# retention (fraction of frames locked-on since acquisition) and tracking RMSE
# (root-mean-square centroid error over the scenario, backend.TrackingThread.rmse).
CR = ['Acquisition (s)', 'Tracking error (px)', 'Target loss', 'Speed (FPS)', 'Update (ms)', 'Lock retention (%)', 'Tracking RMSE (px)']
FM = [2, 2, 0, 0, 1, 1, 2]
# Pass/fail reference lines pulled from the SIH problem-statement performance table (§16-20).
# None = no hard spec threshold for that column; else (limit, lower_is_better).
SPEC_THRESH = [(2.0, True), (10.0, True), (1.0, True), (20.0, False), None, None, (10.0, True)]
PARAMS = [('noise', 'Noise level', 100, 100, 0), ('blur', 'Blur Level', 100, 100, 0),
          ('atm', 'Atmospheric disturbance', 100, 100, 0), ('sd', 'Max standard deviation (px)', 50, 10, 0),
          ('jit', 'Max camera jitter (px)', 30, 10, 0), ('plat', 'Platform motion', 100, 100, 0),]

# --------------------------------------------------------------------------- spec reference (from problem statement PDF)
# Everything below mirrors PS #26169 (Problem Statement 4) — parameter tables are
# (Parameter, Suggested value, Remarks) exactly as the statement words them.
SPEC_OBJECTIVE = ("Develop a software system that autonomously detects, identifies, and continuously tracks a designated "
                  "moving target within a virtual scene by controlling a virtual camera viewport.")
SPEC_CAMERA = [('1. Screen size (min.)', '2000 x 2000 pixels', 'Optional: user-defined'),
               ('2. Camera type', 'Monochrome, focal plane array', 'Optional: colour'),
               ('3. Camera resolution', '640 x 480 pixels', 'Optional: user-defined'),
               ('4. Camera FOV', 'User-defined', 'Default: 4° x 3°'),
               ('5. Camera update rate', '30 Hz (min.)', '-'),
               ('6. Initial camera position', 'Centre of the screen', '-')]
SPEC_TARGET = [('7. Target type', 'Beacon spot', '-'),
               ('8. Number of targets', '1', 'Mandatory; multiple optional'),
               ('9. Target shape', 'User-defined', 'Default: square'),
               ('10. Target size', '5-20 x 5-20 pixels (user-defined)', 'Default: 10 x 10'),
               ('11. Initial target location', 'User-defined', 'Default: random'),
               ('12. Motion', 'Selectable, at least four: straight line, circular, figure of 8, random',
                'Optional: spiral, sinusoidal, user-defined')]
SPEC_MOTION = [('13. Max. pan speed', '5-10 °/s (user-defined)', 'Default: 5 °/s'),
               ('14. Max. tilt speed', '5-10 °/s (user-defined)', 'Default: 5 °/s'),
               ('15. Update interval', '≥ 20 Hz', '-')]
SPEC_PERF = [('16. Acquisition time', '≤ 2 s'), ('17. Tracking error', '≤ 10 pixels'), ('18. Target loss', '< 5 %'),
             ('19. Re-acquisition time', '≤ 1 s'), ('20. Processing speed', '≥ 20 FPS')]
SPEC_NOISE = [('21. Image noise', '1. Salt & pepper (around 10% of image), 2. Gaussian, 3. Poisson', 'User selectable (one or more)'),
              ('22. Max. std. deviation of noise', '20 pixels', 'User-defined'),
              ('23. Max. camera jitter', '± 20 pixels / frame', 'User-defined'),
              ('24. Atmospheric disturbance', 'Clear, haze, fog, rain, low light', 'User-defined reduction in contrast and brightness'),
              ('25. Platform motion', '± 20 pixels / frame (max.)',
               'User selectable. Default/mandatory: linear. Optional: circular, random, spiral, figure of 8, etc.')]
SPEC_SOLUTION = ['Generate a configurable virtual environment', 'Generate one or more moving targets',
                 'Implement a movable virtual camera', 'Detect the target beacon automatically',
                 'Track the beacon continuously using computer vision', 'Control and reposition the virtual camera',
                 'Generate and introduce disturbances (atmospheric turbulence, platform vibrations, camera motion, noise, etc.) in the virtual camera feed',
                 'Display tracking performance and statistics in real time']
SPEC_COARSE_STEPS = ['Observe the surrounding environment', 'Acquire and detect the remote terminal or beacon',
                     'Estimate the position', 'Continuously adjust the pointing direction to maintain visibility']
SPEC_DELIVERABLES = [('Software application', 'A standalone executable implementing the complete virtual camera tracking system, with all mandatory functions and features.'),
                     ('Source code', 'Complete source code with proper documentation — modular and adequately commented.'),
                     ('Technical report', 'About 10-15 pages: problem understanding, system architecture, software modules, tracking methods, AI methods (if used), test methodology, performance analysis and future improvements.'),
                     ('User manual', 'Description of installation, application operation, parameter configuration, GUI, etc.'),
                     ('Demo video (optional)', 'A 3-5 minute video demonstrating the application.'),
                     ('Performance log', 'Automatically generated report: simulation duration, FPS, acquisition time, average and maximum tracking error, lock retention rate, processing time, etc.')]
SPEC_EVAL = [('Functional verification', 20, '#2f8fd1', '10-15 min live demo. Criteria: implementation of all mandatory functions, operational success, GUI'),
             ('Benchmark performance-1', 30, '#0f9773', 'Team is given a few scenarios. Criteria: execution of the scenario, log of centroiding error, automatically generated performance logs'),
             ('Benchmark performance-2', 30, '#c07a1a', 'Team is given a few .mp4 files @30 fps covering a complete screen with noise and a moving beacon spot; the software bypasses its PTZ camera and feeds the video to the coarse pointing system. Criteria: centroiding error vs predefined values; RMSE, acquisition and re-acquisition time, lock retention rate, FPS, etc.'),
             ('Technical evaluation', 20, '#d24b4b', 'Presentation of approach, methods, architecture and design. Criteria: understanding of the problem, system architecture and software design, algorithm selection, AI and computer vision, innovation and novelty, technical documentation and presentation, technical discussion and Q&A')]
SPEC_META = [('Organization', 'Department of Space / Indian Space Research Organisation'), ('Category', 'Software'),
             ('Theme', 'Smart Automation, Space Technology')]
SPEC_LINK_URL = 'https://www.instagram.com/adiash77/'   # where the Spec reference tab's top-right button sends you

class WheelGuard(QObject):
    """App-wide event filter: the mouse wheel over a slider or dropdown that sits inside a
    scrollable page scrolls the *page* instead of silently changing the control's value
    (the classic 'I was just scrolling and my slider moved' bug). If the page has nothing
    to scroll, the wheel still adjusts the control as usual."""
    def eventFilter(s, obj, ev):
        if ev.type() == QEvent.Type.Wheel and isinstance(obj, (QComboBox, QSlider)):
            w = obj.parentWidget()
            while w is not None:
                if isinstance(w, QScrollArea):
                    sb = w.verticalScrollBar()
                    if sb.maximum() > 0:
                        dy = ev.angleDelta().y()
                        sb.setValue(sb.value() - int(dy / 120 * 60))   # 60 px per wheel notch
                        return True
                    break
                w = w.parentWidget()
        return False

class AutoHeightTable(QTableWidget):
    """Read-only reference table that grows to show every row in full (word-wrapped)
    instead of getting its own inner scrollbar — so the mouse wheel over it scrolls the
    page, and long remarks are never clipped."""
    def __init__(s, *a):
        super().__init__(*a)
        s.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        s.setWordWrap(True); s.setTextElideMode(Qt.TextElideMode.ElideNone)
        s.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def _fit(s):
        h = s.horizontalHeader().height() + sum(s.rowHeight(r) for r in range(s.rowCount())) + 4
        if s.height() != h: s.setFixedHeight(h)

    def resizeEvent(s, e):
        super().resizeEvent(e); s.resizeRowsToContents(); s._fit()

    def showEvent(s, e):
        super().showEvent(e); s.resizeRowsToContents(); s._fit()

def panel(title, *widgets):
    f = QFrame(); f.setObjectName('panel'); l = QVBoxLayout(f); l.setContentsMargins(16, 14, 16, 16); l.setSpacing(10)
    h = QLabel(title); h.setObjectName('h3'); l.addWidget(h)
    for w in widgets: l.addWidget(w, 1 if w is widgets[-1] else 0)
    return f

def panel_with_header_control(title, control, *widgets):
    """Same as panel(), but with an extra widget (e.g. a QComboBox) pushed to the far
    right of the title row via a stretch, rather than sitting right beside the title."""
    f = QFrame(); f.setObjectName('panel'); l = QVBoxLayout(f); l.setContentsMargins(16, 14, 16, 16); l.setSpacing(10)
    hd = QWidget(); hl = QHBoxLayout(hd); hl.setContentsMargins(0, 0, 0, 0)
    h = QLabel(title); h.setObjectName('h3')
    hl.addWidget(h); hl.addStretch(); hl.addWidget(control)
    l.addWidget(hd)
    for w in widgets: l.addWidget(w, 1 if w is widgets[-1] else 0)
    return f

def header(title, sub, *btns):
    w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(12)
    t = QLabel(title); t.setObjectName('h1'); sb = QLabel(sub); sb.setObjectName('mut')
    l.addWidget(t); l.addWidget(sb); l.addStretch()
    for b in btns: l.addWidget(b)
    return w, sb

def btn(text, outline=False, fn=None):
    b = QPushButton(text); b.setProperty('o', outline); b.setCursor(Qt.CursorShape.PointingHandCursor)
    if fn: b.clicked.connect(fn)
    return b

def table(cols, rows=0):
    t = QTableWidget(rows, len(cols)); t.setHorizontalHeaderLabels(cols); t.verticalHeader().hide(); t.setShowGrid(False)
    t.setFrameShape(QFrame.Shape.NoFrame); t.setAlternatingRowColors(True); t.verticalHeader().setDefaultSectionSize(32)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); return t

def scroll_wrap(inner, h_as_needed=True):
    """Wrap a widget in a borderless, theme-transparent QScrollArea. The widget keeps filling
    the viewport (and its layout stretch factors keep working) whenever there's enough room;
    scrollbars only appear once the content genuinely doesn't fit, so nothing gets clipped."""
    sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.Shape.NoFrame)
    sc.setWidget(inner); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if h_as_needed else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sc.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return sc

def page(*items, stretch=None):
    w = QWidget(); w.setObjectName('page'); l = QVBoxLayout(w); l.setContentsMargins(22, 20, 22, 20); l.setSpacing(14)
    for i, x in enumerate(items): l.addWidget(x, stretch[i] if stretch else 0)
    return scroll_wrap(w), l

def row(*ws):
    w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(14)
    for x in ws: l.addWidget(x, 1)
    return w

# --------------------------------------------------------------------------- Scene-Parameters builders
# Everything below is used only by build_scene_params() to keep its four panels'
# rows aligned in a single shared grid per panel (rather than one independent
# mini-layout per row, which is what previously let rows drift out of line with
# each other). See the "SCENE-PARAMETERS LAYOUT" block further down for usage.
def panel_top(title, icon_kind, *widgets):
    """Like panel(), but (a) draws a small icon beside the title, and (b) never
    stretches the last widget to fill leftover panel height — instead any extra
    space collects below everything, so rows stay at their natural height and
    evenly spaced instead of one row silently stretching to fill the panel."""
    f = QFrame(); f.setObjectName('panel'); l = QVBoxLayout(f); l.setContentsMargins(22, 20, 22, 22); l.setSpacing(22)
    hd = QWidget(); hl = QHBoxLayout(hd); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(12)
    ic = QLabel(); ic.setPixmap(draw_icon(icon_kind, T['ac'], 40)); ic.setProperty('iconKind', icon_kind); ic.setProperty('iconSize', 40)
    ttl = QLabel(title); ttl.setObjectName('panelTitle')
    hl.addWidget(ic); hl.addWidget(ttl); hl.addStretch()
    l.addWidget(hd)
    for w in widgets: l.addWidget(w, 0)
    l.addStretch(1)
    return f

def scene_grid(paired=False):
    """A QGridLayout-backed container for one panel's rows. With paired=False every
    row is label|control, control filling the full remaining width (Satellite
    Settings, Disturbances). With paired=True the grid has two label|control pairs
    per row (label|control|label|control) so 'Colour'+'Blink' and 'Shape'+'Size'
    line up with each other and with the full-width rows above/below them
    (Beacon Settings)."""
    w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
    g.setVerticalSpacing(22); g.setHorizontalSpacing(16); g.setColumnStretch(1, 1)
    if paired: g.setColumnStretch(3, 1)
    return w, g

def grid_full(g, row_i, label, widget):
    """One label + one control, the control filling the entire rest of the row."""
    g.addWidget(QLabel(label), row_i, 0); g.addWidget(widget, row_i, 1, 1, 3)

def grid_half(g, row_i, side, label, widget):
    """Left half (side=0, columns 0-1) or right half (side=1, columns 2-3) of a
    paired row — used only with scene_grid(paired=True)."""
    c0 = 0 if side == 0 else 2
    g.addWidget(QLabel(label), row_i, c0); g.addWidget(widget, row_i, c0 + 1)

class _NoScrollerPopupStyle(QProxyStyle):
    """A combo box popup taller than its content can scroll two different
    ways: a real scrollbar next to a plain item list, or a pair of thin
    up/down 'scroller' arrow strips at the very top and bottom of the popup
    (the look Fusion uses by default). SH_ComboBox_Popup selects between
    them; forcing it to 0 gives the plain list + real scrollbar, which is
    also what frees up the row of space the arrow strips were reserving
    (that reserved space was what clipped the last row)."""
    def styleHint(s, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_ComboBox_Popup:
            return 0
        return super().styleHint(hint, option, widget, returnData)

class ScrollComboBox(QComboBox):
    """QComboBox whose popup is *forced* to show only setMaxVisibleItems() rows.

    Qt is supposed to honour setMaxVisibleItems() on its own, but on several
    platforms/styles the popup container still auto-sizes to fit every row
    (ignoring the cap), which is why the dropdown was showing all items at
    once instead of 5 + a scrollbar. Re-sizing the view (and its popup
    container) by hand every time the popup opens works regardless of style
    or platform.
    """
    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        s._popup_style = _NoScrollerPopupStyle(s.style())   # keep a reference, or Qt drops it
        s.setStyle(s._popup_style)

    def showPopup(s):
        super().showPopup()
        view = s.view()
        n = max(1, min(s.maxVisibleItems(), s.count()))
        # Sum actual per-row heights rather than assuming every row is
        # identical, so the popup is sized to fit exactly n full rows with
        # nothing clipped off the last one.
        if s.count():
            total_rows = sum(view.sizeHintForRow(i) for i in range(n))
        else:
            total_rows = view.fontMetrics().height() + 12
        cm = view.contentsMargins()
        height = total_rows + view.frameWidth() * 2 + cm.top() + cm.bottom()
        view.setFixedHeight(height)
        popup = view.parentWidget()   # the floating popup container that hosts the view
        if popup is not None:
            popup.setFixedHeight(height)
            # Belt-and-braces: whichever popup style got picked, explicitly
            # hide any leftover top/bottom scroller-strip widgets (not part
            # of the public API, so matched by class name) so only the item
            # list itself is visible.
            for child in popup.findChildren(QWidget):
                if child is not view and 'Scroller' in type(child).__name__:
                    child.hide(); child.setFixedHeight(0)
            # Force off the scrollbar's own up/down step-arrow buttons
            # directly on the scrollbar instance — popups are separate
            # top-level windows, and stylesheet-selector cascading into them
            # can be unreliable, so this bypasses that entirely.
            sb = view.verticalScrollBar()
            if sb is not None:
                sb.setStyleSheet(
                    f"QScrollBar:vertical {{ background:transparent; width:10px; margin:0px; }}"
                    f"QScrollBar::handle:vertical {{ background:{T['ln']}; border-radius:4px; min-height:24px; }}"
                    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
                    "{ height:0px; width:0px; border:none; background:none; }"
                    "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }"
                )
            # Qt positions the *original* (tall, fit-everything) popup so the
            # currently-selected row lines up with the combo box — which, for an
            # item far down the list, means shifting the whole popup way up the
            # screen. Now that we've shrunk the popup down to 5 rows, that
            # position is stale, so re-anchor it directly under the combo box
            # (or above it, if there isn't room below) like a normal dropdown.
            below = s.mapToGlobal(QPoint(0, s.height()))
            screen = s.screen() if hasattr(s, 'screen') and s.screen() else QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen else None
            if avail is not None and below.y() + height > avail.bottom():
                above = s.mapToGlobal(QPoint(0, 0))
                popup.move(above.x(), above.y() - height)
            else:
                popup.move(below.x(), below.y())
            # Keep the current selection visible inside the now-shrunk 5-row
            # window (it can be scrolled out of the freshly-clamped viewport).
            if s.currentIndex() >= 0:
                view.scrollTo(view.model().index(s.currentIndex(), 0), QAbstractItemView.ScrollHint.PositionAtCenter)

def finish_combo_popup(cb, max_visible=5):
    """Apply the compact, scrollable-popup setup (max_visible rows, styled
    side scrollbar, no stray padding, live hover highlight) to a
    ScrollComboBox. Shared by every dropdown that should behave like the
    Test case scene picker."""
    cb.setMaxVisibleItems(max_visible)
    cb.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    cb.view().setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    cb.view().setSpacing(0)
    cb.view().setContentsMargins(0, 0, 0, 0)
    cb.view().setFrameShape(QFrame.Shape.NoFrame)
    # Make sure the QSS ::item:hover highlight actually tracks the mouse as it
    # moves over the popup (some styles leave this off by default).
    cb.view().setMouseTracking(True)
    cb.view().viewport().setMouseTracking(True)
    return cb

def mk_combo(options):
    cb = ScrollComboBox(); cb.addItems(options); return finish_combo_popup(cb)

def mk_toggle(default=True):
    """ON/OFF pill button for boolean controls (e.g. Blink)."""
    b = QPushButton('ON' if default else 'OFF'); b.setCheckable(True); b.setChecked(default)
    b.setFixedWidth(64); b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.toggled.connect(lambda v: b.setText('ON' if v else 'OFF'))
    return b

def mk_slider(mn, mx, default, suffix='', divisor=1):
    """Returns (row_widget, slider) — row_widget (slider + live value label) is
    what goes in the grid; slider is what save_scene_settings() reads the value from.
    `divisor` lets the slider's raw integer steps (what QSlider natively works in)
    display as a fraction — e.g. mn=0, mx=100, divisor=100 gives a 0.00-1.00 range
    in 0.01 increments per step, while the slider itself just moves 0-100."""
    row_w = QWidget(); rl = QHBoxLayout(row_w); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(8)
    sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(mn, mx); sl.setValue(default)
    sl.setProperty('divisor', divisor)   # read back by save_scene_settings()'s value_of()
    disp = lambda v: f"{v / divisor:g}{suffix}"
    val = QLabel(disp(default)); val.setFixedWidth(52); val.setAlignment(Qt.AlignmentFlag.AlignRight)
    sl.valueChanged.connect(lambda v: val.setText(disp(v)))
    rl.addWidget(sl, 1); rl.addWidget(val)
    return row_w, sl

# ============================================================

# --------------------------------------------------------------------------- main window
class Main(QMainWindow):
    def __init__(s):
        super().__init__(); s.setWindowTitle('FSOC Vision Stabilization Console')
        s.settings = QSettings('FSOCTrackLab', 'VisionStabilizationConsole')
        s.theme_mode = str(s.settings.value('theme_mode', 'light'))   # first launch: light
        if s.theme_mode not in THEME_MODES: s.theme_mode = 'light'
        s._theme_is_dark = None
        T.update(DARK if (system_prefers_dark() if s.theme_mode == 'system' else s.theme_mode == 'dark') else LIGHT)
        s.sim, s.video = Sim(), Video(); s.speed, s.acc, s.vacc = 1.0, 0.0, 0.0

        # ---- Simulation-time gauge (Sim View, far right) ----
        # Total duration comes from whichever scene was last loaded (Load scene /
        # Reset scenario read it out of that scene's .txt file); 90s (1:30) default
        # until a scene has been loaded. Only actually counts down once real detection
        # has begun (see on_frame_ready()/on_stats_ready() below) and freezes whenever
        # the tracking thread is paused (see toggle_sim()). Set up before build_sim()
        # runs (a few lines down), since build_sim() creates the gauge widget itself
        # and needs s.sim_duration_total for its initial label text.
        s.sim_duration_total = 90.0
        s.sim_elapsed = 0.0
        s.sim_gauge_running = False   # currently counting down (unpaused + already started)
        s._first_start_scene_loaded = False   # first Start Simulation press also presses Load scene (once)
        s.sim_gauge_started = False   # True forever after the first real frame arrives
        s.sim_gauge_frozen_full = False   # True once time runs out — locks Start Simulation/Pause

        s.bench = [dict(name='Deck baseline', type='Standard scenario suite', date='From SIH deck', cols=list(range(len(CR))), rows=list(SC))]
        s.sel = 0
        # ---- real Benchmark Performance-1 runner state (see run_bench() and
        # _bench_* below) — bench_running gates _bench_on_stats() so ordinary
        # Sim View use never touches any of this.
        s.bench_running = False
        s.bench_queue = []          # scenario names, in run order (from the Scenarios tab checklist)
        s.bench_durations = {}      # {scenario name: simulation-time seconds, from its .txt file}
        s.bench_idx = -1
        s.bench_current_duration = 0.0
        s.bench_total_time = 0.0
        s.bench_elapsed_done = 0.0  # sum of durations of scenarios already finished this run
        s.bench_cols = list(range(len(CR)))
        s.bench_rows = []
        s.bench_frames = []             # raw stats_ready dicts for the scenario currently running
        s.bench_frames_by_scene = {}    # {scenario name: [stats dicts]} for the whole run — feeds the reports
        root = QWidget(); s.setCentralWidget(root); hl = QHBoxLayout(root); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(0)
        hl.addWidget(s.build_nav()); s.stack = QStackedWidget(); hl.addWidget(s.stack, 1)
        for p in (s.build_sim(), s.build_scene_params(), s.build_video(), s.build_bench(), s.build_data(), s.build_spec()): s.stack.addWidget(p)
        s.sim_rail = s.build_sim_rail(); s.video_rail = s.build_video_rail()
        s.rail_stack = QStackedWidget(); s.rail_stack.setFixedWidth(300)
        s.rail_stack.addWidget(s.sim_rail); s.rail_stack.addWidget(s.video_rail)
        hl.addWidget(s.rail_stack)
        s._wheel_guard = WheelGuard(s)
        for _w in s.findChildren(QComboBox) + s.findChildren(QSlider): _w.installEventFilter(s._wheel_guard)

        # ---- Benchmark-running busy overlay: a subtle dark glass pane that sits on
        # top of the entire central widget (nav + tabs + rail) while a benchmark run
        # is in progress, so nothing else can be clicked or navigated to until it's
        # done. Plain, empty QWidget — no text/spinner — its own mouse events just
        # swallow clicks (default Qt behaviour), and it's kept in sync with the
        # window size via resizeEvent() below. Parented to root so it covers exactly
        # the area root's own layout covers.
        s.busy_overlay = QWidget(root)
        s.busy_overlay.setStyleSheet('background-color: rgba(0, 0, 0, 90);')
        s.busy_overlay.setCursor(Qt.CursorShape.WaitCursor)
        s.busy_overlay.hide()

        s.go(0); s.apply_theme(); s.draw_data()
        s.theme_timer = QTimer(s); s.theme_timer.timeout.connect(s._check_system_theme); s.theme_timer.start(2500)
        try: QApplication.styleHints().colorSchemeChanged.connect(lambda *_: s._check_system_theme())
        except AttributeError: pass
        s.timer = QTimer(s); s.timer.timeout.connect(s.tick); s.timer.start(33)

        # ---- real Unity + YOLO/Kalman tracking (replaces the fake sim in Sim View) ----
        s.real_mode = True
        s.unity_proc = None
        s.unity_hwnd = None
        s._unity_parent = None  # whichever container (Sim View / Video View) currently hosts the Unity window
        s._embed_attempts = 0
        s.rt_events = []  # rolling log fed into the Events panel

        # Rolling window of the real per-frame tracking error (px), used to drive the
        # "Tracking performance" chart in Sim View instead of the fake Sim.err buffer.
        s.rt_err_history = deque(maxlen=300)

        # Latest values reported by backend.TrackingThread.stats_ready, used to fill in
        # the Tracking performance panel's Live column (lock retention, RMSE, acquisition
        # time, target loss count, accuracy) instead of the fake Sim properties.
        s.rt_stats = dict(rmse=None, lock_retention=0.0, acquisition_time=None,
                           target_losses=0, accuracy=0.0)

        s.tracking_thread = TrackingThread()
        s.tracking_thread.frame_ready.connect(s.on_frame_ready)
        s.tracking_thread.status_changed.connect(s.on_status_changed)
        s.tracking_thread.stats_ready.connect(s.on_stats_ready)
        s.tracking_thread.start()
        # Stay paused until "Start Simulation" is pressed on Sim View — Unity launches
        # and connects automatically below, but sends/receives nothing useful until
        # then. pause()/is_paused() are safe to call this early: they just flip a
        # flag the run loop already checks continuously, the same one toggle_sim()
        # uses later while the thread is mid-loop.
        s.tracking_thread.pause()

        # Launch Unity automatically instead of waiting for a button press. The
        # "Start Unity" button still exists (and now starts disabled-after-launch,
        # same as a manual click would leave it) in case Unity needs restarting.
        
        # ........................................................................................
        # QTimer.singleShot(0, s.start_unity)

    # ---- shell
    def build_nav(s):
        f = QFrame(); f.setObjectName('nav'); f.setFixedWidth(76); l = QVBoxLayout(f); l.setContentsMargins(0, 18, 0, 16); l.setSpacing(8)
        logo = QLabel(); logo.setPixmap(beacon_logo()); logo.setAlignment(Qt.AlignmentFlag.AlignCenter); l.addWidget(logo); l.addSpacing(10)
        s.grp = QButtonGroup(s); ctr = Qt.AlignmentFlag.AlignHCenter
        for i, (k, tip) in enumerate((('sim', 'Sim View'), ('scene', 'Scene-Parameters'), ('video', 'Video View'), ('bench', 'Benchmark'), ('data', 'Data-Check'), ('info', 'Spec Reference'))):
            b = QPushButton(); b.setObjectName('nb'); b.setCheckable(True); b.setIcon(make_icon(k)); b.setIconSize(QSize(26, 26))
            b.setFixedSize(52, 48); b.setToolTip(tip); b.setAccessibleName(tip); b.setCursor(Qt.CursorShape.PointingHandCursor)
            s.grp.addButton(b, i); l.addWidget(b, 0, ctr)
        s.grp.idClicked.connect(s.go); l.addStretch()
        th = QPushButton(); th.setObjectName('nb'); th.setIcon(make_icon('theme')); th.setIconSize(QSize(26, 26)); th.setFixedSize(52, 48)
        th.setToolTip('Appearance: light / dark / system'); th.setAccessibleName('Appearance'); th.setCursor(Qt.CursorShape.PointingHandCursor)
        s.theme_btn = th; th.clicked.connect(s.show_theme_menu); l.addWidget(th, 0, ctr); return f

    def go(s, i):
        if getattr(s, 'bench_running', False):
            return   # tab switches are blocked while a benchmark run is in progress (busy overlay)
        s.stack.setCurrentIndex(i); s.grp.button(i).setChecked(True)
        if i == 0: s.refresh_scene_list()   # pick up any scene saved since Sim View was last shown
        if i == 3: s.refresh_bench_scene_list()   # same, for Benchmark's Scenarios tab
        if hasattr(s, 'rail_stack'):
            show_rail = i in (0, 2)
            s.rail_stack.setVisible(show_rail)
            if show_rail:
                s.rail_stack.setCurrentIndex(0 if i == 0 else 1)
                s._refresh_rail_layout()
        s._reparent_unity_to_tab(i)
        s._apply_view_for_tab(i)

    def _apply_view_for_tab(s, i):
        """Video View (i==2) always uses TrackingThread.second_view(). Sim View (i==0)
        always uses whichever of sat_pov() / second_view() the dropdown currently has
        selected — re-invoked every time Sim View is switched to, not just when the
        dropdown itself changes. No-ops until tracking_thread exists (e.g. during the
        early s.go(0) call in __init__) and on any other tab (Scene-Parameters, etc.)."""
        tt = getattr(s, 'tracking_thread', None)
        if tt is None:
            return
        if i == 2:
            tt.third_view()
        elif i == 0:
            s._invoke_sat_view(getattr(s, 'selected_sat_view', 'Satellite Pov'))

    def _reparent_unity_to_tab(s, i):
        """Sim View (i==0) and Video View (i==2) each have their own native-window
        container for the embedded Unity view (s.unity_container / s.vsat). A Win32
        HWND can only be a child of one parent at a time, so re-parent the real Unity
        window to whichever of the two containers is on screen right now. No-ops until
        Unity has actually been started (unity_hwnd set) and safe to call from go(0)
        during __init__, before the real-mode attributes below even exist yet."""
        if not getattr(s, 'unity_hwnd', None):
            return
        target = s.unity_container if i == 0 else (s.vsat if i == 2 else None)
        if target is None or target is s._unity_parent:
            return
        s._unity_parent = target
        win32gui.SetParent(s.unity_hwnd, int(target.winId()))
        s.resize_unity_window()
        QTimer.singleShot(150, s.nudge_unity_render_loop)

    def _refresh_rail_layout(s):
        """Qt's QHBoxLayout can leave a stale cached size for a fixed-width sibling (the
        rail) and for word-wrapped QLabels inside it after the sibling has been hidden and
        re-shown via a QStackedWidget switch. Re-maximizing the window used to be the only
        thing that forced a fresh resize event and fixed it - do the equivalent here instead.
        Operates on whichever of sim_rail / video_rail is the current page of rail_stack."""
        cur = s.rail_stack.currentWidget()
        inner = getattr(cur, 'rail_inner', cur)
        for lab in inner.findChildren(QLabel):
            if lab.wordWrap(): lab.setWordWrap(False); lab.setWordWrap(True)
        lay = inner.layout()
        if lay: lay.invalidate(); lay.activate()
        inner.updateGeometry(); cur.updateGeometry()
        root_lay = s.centralWidget().layout()
        root_lay.invalidate(); root_lay.activate()
        s.centralWidget().updateGeometry()

    def show_theme_menu(s):
        """Light Mode / Dark Mode / System Default picker, opened from the nav's theme
        button. The choice is saved (QSettings) and restored on every launch."""
        def tick_icon(on):
            pm = _pix(14)
            if on:
                q = QPainter(pm); q.setRenderHint(QPainter.RenderHint.Antialiasing); q.setPen(_pen(T['ac'], 2))
                q.drawPolyline(QPolygonF([QPointF(2.5, 7.5), QPointF(5.8, 10.5), QPointF(11.5, 3.8)])); q.end()
            return QIcon(pm)
        menu = QMenu(s)
        for mode, label in (('light', 'Light Mode'), ('dark', 'Dark Mode'), ('system', 'System Default')):
            a = QAction(label, menu); a.setIcon(tick_icon(mode == s.theme_mode))
            a.triggered.connect(lambda _=False, m=mode: s.set_theme_mode(m)); menu.addAction(a)
        b = s.theme_btn
        menu.exec(b.mapToGlobal(QPoint(b.width() + 6, b.height() - menu.sizeHint().height())))

    def set_theme_mode(s, mode):
        if mode not in THEME_MODES: mode = 'light'
        s.theme_mode = mode; s.settings.setValue('theme_mode', mode); s.apply_theme()

    def _check_system_theme(s):
        """While 'System Default' is selected, follow the OS if it flips light<->dark
        while the app is running (also wired to Qt's colorSchemeChanged when available)."""
        if s.theme_mode == 'system' and system_prefers_dark() != s._theme_is_dark:
            s.apply_theme()

    def apply_theme(s, toggle=False):
        dark = system_prefers_dark() if s.theme_mode == 'system' else (s.theme_mode == 'dark')
        s._theme_is_dark = dark
        T.update(DARK if dark else LIGHT)
        k = T['bg'][1:]; T['tick'] = asset('tick' + k, _tick, 16); T['chev'] = asset('chev' + k, _chev, 12)
        pal = QPalette(); R = QPalette.ColorRole
        for role, key in ((R.Window, 'bg'), (R.WindowText, 'ink'), (R.Base, 'pn'), (R.AlternateBase, 'soft'), (R.Text, 'ink'),
                          (R.Button, 'pn'), (R.ButtonText, 'ink'), (R.Highlight, 'ac'), (R.HighlightedText, 'onac')):
            pal.setColor(role, QColor(T[key]))
        QApplication.instance().setPalette(pal); s.setStyleSheet(qss()); s._lk = None
        for lab in s.findChildren(QLabel):
            if lab.property('iconKind'): lab.setPixmap(draw_icon(lab.property('iconKind'), T['ac'], lab.property('iconSize')))
        for w in s.findChildren(QWidget): w.update()

    def build_sim_rail(s):
        f = QFrame(); l = QVBoxLayout(f); l.setContentsMargins(0, 20, 16, 20); l.setSpacing(14)
        s.r = {k: QLabel('-') for k in ('det', 'pan', 'tilt', 'off', 'time', 'fps')}
        s.ls = QLabel('Searching'); s.ls.setAlignment(Qt.AlignmentFlag.AlignCenter)
        def kv(name, w):
            r = QWidget(); h = QHBoxLayout(r); h.setContentsMargins(0, 0, 0, 0); a = QLabel(name); a.setObjectName('mut')
            w.setStyleSheet("font-weight:700; font-family:'JetBrains Mono','Consolas','SF Mono',monospace;"); h.addWidget(a); h.addStretch(); h.addWidget(w); return r
        s.pdet = QLabel('Waiting...')
        s.pkal = QLabel('Not initialized')
        s.punity = QLabel('Not started')
        l.addWidget(panel('Link status', s.ls, kv('Beacon', s.r['det'])))
        l.addWidget(panel('Gimbal', kv('Pan', s.r['pan']), kv('Tilt', s.r['tilt']), kv('Offset from centre', s.r['off']),
                          kv('Elapsed', s.r['time']), kv('Render rate', s.r['fps'])))
        l.addWidget(panel('Pipeline', kv('Scene (Unity)', s.punity), kv('Detector (YOLO)', s.pdet),
                          kv('Kalman filter', s.pkal), kv('Pan/tilt controller', QLabel('Running'))))
        # Fixed-height, scrollable event log — was a plain QLabel that grew
        # taller with every long/wrapped line and pushed 'Simulation time'
        # further down the rail. A read-only QPlainTextEdit keeps the exact
        # same look (transparent, muted text, word-wrap) but never grows past
        # setFixedHeight; once the log overflows that, the normal in-tree
        # scrollbar (styled the same as everywhere else in the app, no arrow
        # buttons) takes over instead.
        s.evl = QPlainTextEdit(); s.evl.setObjectName('evl'); s.evl.setReadOnly(True)
        s.evl.setFixedHeight(120)
        s.evl.setFrameShape(QFrame.Shape.NoFrame)
        s.evl.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        s.evl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        s.evl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        s.evl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        s.evl.setCursorWidth(0)
        s._evl_cache = None   # last-rendered text, so a same-content refresh doesn't reset scroll position
        l.addWidget(panel('Events', s.evl))

        # ---- Simulation-time gauge: fills horizontally (left -> right) as the sim
        # runs, showing time left. Lives below the Events panel, filling the rail's
        # full width, rather than beside the Unity embed. See the state comment in
        # __init__ for exactly when it starts/stops/freezes.
        s.sim_gauge = QProgressBar()
        s.sim_gauge.setOrientation(Qt.Orientation.Horizontal)
        s.sim_gauge.setRange(0, 100); s.sim_gauge.setValue(0); s.sim_gauge.setTextVisible(False)
        s.sim_gauge.setFixedHeight(16)
        s.sim_gauge.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        s.sim_gauge_label = QLabel(s._fmt_time_left(s.sim_duration_total)); s.sim_gauge_label.setObjectName('mut')
        s.sim_gauge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # '+'/'-' buttons sit above the gauge bar and nudge the remaining time by
        # 5s each click (see adjust_sim_time()) — '+' adds time back, '-' burns it.
        s.sim_gauge_minus = QPushButton('−'); s.sim_gauge_plus = QPushButton('+')
        for b in (s.sim_gauge_minus, s.sim_gauge_plus):
            b.setObjectName('gaugeBtn'); b.setFixedSize(34, 30); b.setCursor(Qt.CursorShape.PointingHandCursor)
        s.sim_gauge_minus.clicked.connect(lambda: s.adjust_sim_time(-5))
        s.sim_gauge_plus.clicked.connect(lambda: s.adjust_sim_time(5))
        gauge_btn_row = QWidget(); gbl = QHBoxLayout(gauge_btn_row); gbl.setContentsMargins(0, 0, 0, 0); gbl.setSpacing(8)
        gbl.addWidget(s.sim_gauge_minus); gbl.addStretch(); gbl.addWidget(s.sim_gauge_plus)

        l.addWidget(panel('Simulation time', gauge_btn_row, s.sim_gauge, s.sim_gauge_label))
        l.addStretch(); s.fps, s._last = 30.0, 0
        sc = scroll_wrap(f, h_as_needed=False); sc.rail_inner = f; return sc

    def build_video_rail(s):
        """Right-side panel shown on Video View — distinct from Sim View's rail (gimbal
        telemetry / Unity pipeline). This one summarizes the imported video source,
        detector status and playback position instead."""
        f = QFrame(); l = QVBoxLayout(f); l.setContentsMargins(0, 20, 16, 20); l.setSpacing(14)
        s.vr = {k: QLabel('-') for k in ('src', 'state', 'speed', 'det', 'off', 'pan', 'tilt', 'time', 'frame', 'mode', 'acq', 'rmse', 'lock', 'acc', 'loss', 'reacq')}
        def kv(name, w):
            r = QWidget(); h = QHBoxLayout(r); h.setContentsMargins(0, 0, 0, 0); a = QLabel(name); a.setObjectName('mut')
            w.setStyleSheet("font-weight:700; font-family:'JetBrains Mono','Consolas','SF Mono',monospace;"); h.addWidget(a); h.addStretch(); h.addWidget(w); return r
        for lab in s.vr.values(): lab.setWordWrap(True)
        l.addWidget(panel('Source', kv('File', s.vr['src']), kv('Playback', s.vr['state']), kv('Speed', s.vr['speed'])))
        l.addWidget(panel('Detector', kv('Beacon', s.vr['det']), kv('Offset from centre', s.vr['off']),
                  kv('Pan', s.vr['pan']), kv('Tilt', s.vr['tilt']), kv('Mode', s.vr['mode'])))
        l.addWidget(panel('Tracking performance', kv('Acquisition time', s.vr['acq']), kv('RMSE', s.vr['rmse']), kv('Lock retention', s.vr['lock']),
                          kv('Accuracy', s.vr['acc']), kv('Target losses', s.vr['loss']), kv('Avg reacquisition time', s.vr['reacq'])))
        l.addWidget(panel('Playback position', kv('Elapsed', s.vr['time']), kv('Frame', s.vr['frame'])))
        l.addStretch()
        sc = scroll_wrap(f, h_as_needed=False); sc.rail_inner = f; return sc

    # ---- Sim View
    def build_sim(s):
        s.pz = btn('Start Simulation', True, s.toggle_sim); rs = btn('Reset scenario', False, s.reset_scenario)
        s.start_unity_btn = btn('Start 3D Environment', False, s.start_unity)
        hd, _ = header('Live simulation', 'Python tracker driving the Unity gimbal in a closed loop', s.start_unity_btn, s.pz, rs)

        # ---- real live views (replacing the fake CamView/SatView painters) ----
        s.cam = QLabel('Waiting for tracking feed...')
        s.cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.cam.setStyleSheet('background-color:#070c18; color:#8fa3c7; border-radius:9px;')
        s.cam.setMinimumSize(210, 150)
        s.cam.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        s.unity_container = QWidget()
        s.unity_container.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        s.unity_container.setStyleSheet('background-color:#070c18; border-radius:9px;')
        s.unity_container.setMinimumSize(230, 140)
        s.sat = s.unity_container  # keep name `s.sat` so the shared tick() loop still finds it

        cw = QWidget(); g = QGridLayout(cw); g.setContentsMargins(0, 0, 0, 0); g.setVerticalSpacing(12); g.setColumnStretch(1, 1)
        # Test-case scene picker — options are the .txt scene files saved from the
        # Scene-Parameters tab into SCENES_DIR (see the constant near the top of this
        # file and refresh_scene_list()), placed first so a scenario can be picked
        # before touching the other controls. "Load scene" sends the chosen file's
        # name to backend.TrackingThread.parsed() (load_selected_scene()).
        tc_row = QWidget(); tc_l = QHBoxLayout(tc_row); tc_l.setContentsMargins(0, 0, 0, 0); tc_l.setSpacing(8)
        s.test_case = ScrollComboBox(); s.test_case.currentTextChanged.connect(s.on_test_case_changed)
        finish_combo_popup(s.test_case)   # popup shows 5 rows at a time; scrolls for the rest
        s.load_scene_btn = QPushButton('Load scene'); s.load_scene_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        s.load_scene_btn.clicked.connect(s.load_selected_scene)
        # Small progress bar that fills over ~2.5s as visual feedback while a scene
        # loads (see _play_load_scene_animation()) — hidden the rest of the time.
        s.load_progress = QProgressBar(); s.load_progress.setRange(0, 100)
        s.load_progress.setFixedWidth(110); s.load_progress.setTextVisible(False); s.load_progress.hide()
        tc_l.addWidget(s.test_case, 1); tc_l.addWidget(s.load_scene_btn); tc_l.addWidget(s.load_progress)
        s.refresh_scene_list()   # populate the combo now that it exists
        g.addWidget(QLabel('Test case scene'), 0, 0); g.addWidget(tc_row, 0, 1, 1, 2)
        s.mt = mk_combo(['Straight Horizontal', 'Straight Vertical', 'Figure 8', 'Circular', 'Mixed']); s.mt.setCurrentText('Straight Horizontal')
        s.mt.currentTextChanged.connect(s.on_movement_changed)
        g.addWidget(QLabel('Satellite movement type'), 1, 0); g.addWidget(s.mt, 1, 1, 1, 2)
        for i, (k, lab, mx, dv, df) in enumerate(PARAMS, 2):
            sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(0, mx); sl.setValue(int(df * dv)); val = QLabel(f"{df:g}")
            sl.valueChanged.connect(lambda v, k=k, dv=dv, val=val: (s.sim.p.__setitem__(k, v / dv), val.setText(f"{v / dv:g}"),s.slide_change(list(s.sim.p.values()))))
            g.addWidget(QLabel(lab), i, 0); g.addWidget(sl, i, 1); g.addWidget(val, i, 2)
        pw = QWidget(); pg = QGridLayout(pw); pg.setContentsMargins(0, 0, 0, 0); pg.setVerticalSpacing(8); pg.setColumnStretch(0, 1); s.k = []
        for j, (n, tg) in enumerate((('Lock retention', '98.2%'), ('Tracking RMSE', '6.1 px'), ('Acquisition time', '0.73 s'),
                                     ('Target loss', '<1.8%'), ('Accuracy (within 12 px)', '-'))):
            lv = QLabel('-'); lv.setStyleSheet('font-weight:700'); s.k.append(lv)
            tl = QLabel(tg); tl.setObjectName('mut'); pg.addWidget(QLabel(n), j + 1, 0); pg.addWidget(lv, j + 1, 1); pg.addWidget(tl, j + 1, 2)
        for c, n in enumerate(('Metric', 'Live', 'Deck target')):
            h = QLabel(n); h.setObjectName('mut'); pg.addWidget(h, 0, c)
        # Feed the chart from the real per-frame error history (backend.py's `e` values,
        # collected in rt_err_history) once real tracking is running; fall back to the
        # fake Sim.err buffer only if real_mode is somehow off.
        s.chart = Chart(lambda: list(s.rt_err_history) if s.real_mode else s.sim.err, 120, 60)
        perf = panel('Tracking performance', pw, QLabel('Graphical view: error over time (px)'), s.chart)
        # View-mode picker for the embedded Unity panel — sits on the right of the panel's
        # title row (not right beside the title text) via panel_with_header_control's stretch.
        s.sat_view = mk_combo(['Satellite Pov', 'Satellite motion view'])
        s.sat_view.setFixedWidth(180)
        s.sat_view.currentTextChanged.connect(s.on_sat_view_changed)
        s.selected_sat_view = s.sat_view.currentText()
        unity_panel = panel_with_header_control('Unity (live, embedded)', s.sat_view, s.sat)
        top_row = QWidget(); trl = QHBoxLayout(top_row); trl.setContentsMargins(0, 0, 0, 0); trl.setSpacing(14)
        trl.addWidget(panel('Python-side view (live)', s.cam), 1)
        trl.addWidget(unity_panel, 1)
        w, l = page(hd, top_row, row(panel('Scenario controls', cw), perf), stretch=[0, 5, 4]); return w

    def slide_change(s, val):
        print("sluder was changed:!!!!! and the list: ", val, s.sim.running)
        tt = getattr(s, 'tracking_thread', None)
        if tt is None:
            return
        else:
            tt.set_effects(val[:3])

    def reset_scenario(s):
        """"Reset scenario" button — reloads whatever scene is currently selected in the
        Test case scene dropdown (the last one chosen), exactly as if Load scene had
        been pressed again: same backend.TrackingThread.parsed() call, same Sim View
        stats reset, same loading animation."""
        s.sim.reset()   # keep resetting the fake sim too — harmless housekeeping, matches existing convention
        s.load_selected_scene()

    def toggle_sim(s):
        """First press starts the (already-running, already-connected-to-Unity) tracking
        loop that begins life paused in __init__; every press after that just pauses or
        resumes it same as before. Deliberately does not touch s.sim.running (the fake
        fallback demo) — that belongs to Video View's own Pause button (toggle_video),
        kept fully separate so the two Pause/Resume buttons never affect each other's tab.
        Also freezes/resumes the simulation-time gauge in lockstep — see s.sim_gauge_running."""
        if s.sim_gauge_frozen_full:
            return   # time's up — button should already be disabled, but guard anyway
        tt = getattr(s, 'tracking_thread', None)
        if tt is None:
            return
        if tt.is_paused():
            if not s._first_start_scene_loaded:
                # Very first Start Simulation press of this session: do exactly what the
                # Load scene button does first (loads whichever scene is selected in the
                # Test case scene dropdown), then carry on and start as normal. Never
                # repeated — every later press is a plain pause/resume.
                s._first_start_scene_loaded = True
                if s.test_case.currentText() and not s.load_selected_scene():
                    s._first_start_scene_loaded = False   # load failed (warning already shown) — retry next press
                    return
            tt.resume()
            s.sim_gauge_running = s.sim_gauge_started   # only resumes counting if it's already begun
        else:
            tt.pause()
            s.sim_gauge_running = False
        s.pz.setText('Pause' if not tt.is_paused() else 'Start Simulation')

    def adjust_sim_time(s, delta):
        """'+'/'-' buttons above the simulation-time gauge: nudge the remaining time
        by `delta` seconds (positive adds time back, negative burns it early), without
        touching the total duration the gauge fill is scaled against. Clamped to
        [0, sim_duration_total]. Adding time back after the gauge had frozen full
        (time's up) un-freezes it and, if the sim had already been started, resumes
        the countdown — mirroring what a fresh Load scene / Reset scenario does."""
        was_frozen = s.sim_gauge_frozen_full
        s.sim_elapsed = min(max(s.sim_elapsed - delta, 0.0), s.sim_duration_total)
        if was_frozen and s.sim_elapsed < s.sim_duration_total:
            s.sim_gauge_frozen_full = False
            s.sim_gauge_running = s.sim_gauge_started
            s.pz.setEnabled(True)
        frac = (s.sim_elapsed / s.sim_duration_total) if s.sim_duration_total > 0 else 1.0
        s.sim_gauge.setValue(int(frac * 100))
        s.sim_gauge_label.setText(s._fmt_time_left(s.sim_duration_total - s.sim_elapsed))
        if s.sim_elapsed >= s.sim_duration_total:
            s._sim_time_up()

    def _sim_time_up(s):
        """Simulation-time gauge just filled completely: force-pause the tracking
        thread and lock the Start Simulation/Pause button so nobody can un-pause it
        again — only a fresh Load scene / Reset scenario (which resets the gauge)
        re-enables it."""
        s.sim_gauge_running = False
        s.sim_gauge_frozen_full = True
        tt = getattr(s, 'tracking_thread', None)
        if tt is not None and not tt.is_paused():
            tt.pause()
        s.pz.setText('Start Simulation')
        s.pz.setEnabled(False)
        s.on_status_changed('Simulation time elapsed — paused.')

    def on_movement_changed(s, name):
        """Satellite movement type combo. Keeps the fake Sim's type/reset as harmless
        housekeeping (unused visually while real_mode is on), and forwards the selected
        option straight to backend.TrackingThread.type_chosen(name)."""
        s.sim.type = name; s.sim.reset()
        tt = getattr(s, 'tracking_thread', None)
        if tt is not None:
            tt.type_chosen(name)
        s.on_status_changed(f"Satellite movement type: {name}")

    def on_test_case_changed(s, name):
        """Selecting a scene here just records the choice and logs it — it does not load
        it into the tracker. The "Load scene" button next to this dropdown is what
        actually sends it to backend.TrackingThread.parsed() (see load_selected_scene())."""
        s.selected_test_case = name
        s.on_status_changed(f"Test case scene selected: {name}")

    def refresh_scene_list(s):
        """Re-reads SCENES_DIR for .txt scene files and repopulates the Sim View
        "Test case scene" dropdown with their names (the file names, without the
        .txt extension). Called when Sim View is first built, again every time
        Save is pressed on the Scene-Parameters tab, and again whenever the user
        navigates to Sim View (see go()), so a newly saved scene shows up without
        restarting the app. Keeps the current selection if it still exists.

        Ordering: sorted by file-creation time, oldest first — so the most
        recently created scene file ends up last in the list / at the bottom of
        the dropdown, not alphabetically. os.path.getctime() is what's used to
        determine "creation time" here (see note below)."""
        try:
            paths = glob.glob(os.path.join(SCENES_DIR, '*.txt'))
            # getctime() is the file's creation time on Windows (this app is
            # win32gui-based, i.e. Windows-only) — on Linux/macOS getctime()
            # instead reflects last metadata-change time, so this ordering is
            # only guaranteed to be "creation order" on Windows.
            paths.sort(key=os.path.getctime)
            names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        except OSError:
            names = []
        if not hasattr(s, 'test_case'):
            return
        current = s.test_case.currentText()
        s.test_case.blockSignals(True)
        s.test_case.clear(); s.test_case.addItems(names)
        if current in names: s.test_case.setCurrentText(current)
        s.test_case.blockSignals(False)
        s.selected_test_case = s.test_case.currentText()

    def load_selected_scene(s):
        """"Load scene" button beside the Test case scene dropdown. Passes the
        selected scene's file name (without the .txt extension, exactly as shown
        in the dropdown) to backend.TrackingThread.parsed(), which is expected to
        read that file back out of SCENES_DIR and apply it."""
        name = s.test_case.currentText()
        if not name:
            QMessageBox.warning(s, 'Sim View', f'No scene files found in:\n{SCENES_DIR}')
            return False
        try:
            s.tracking_thread.parsed(name)
            s.on_status_changed(f"Loaded scene: {name}")
        except Exception as e:
            QMessageBox.warning(s, 'Sim View', f'Could not load scene "{name}": {e}')
            return False
        s.clear_sim_view_stats(duration=s._read_scene_duration(name))
        s._play_load_scene_animation()
        return True

    def _read_scene_duration(s, name):
        """Reads the 'Simulation time' value (seconds) out of a saved scene .txt file
        in SCENES_DIR — the same file save_scene_settings() writes, one 'Label: value'
        line per control. Falls back to the 90s (1:30) default if the file is missing,
        unreadable, or doesn't have that field (e.g. an older scene saved before this
        field existed)."""
        path = os.path.join(SCENES_DIR, name if name.lower().endswith('.txt') else name + '.txt')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.lower().startswith('simulation time:'):
                        return float(line.split(':', 1)[1].strip())
        except (OSError, ValueError):
            pass
        return 90.0

    def clear_sim_view_stats(s, duration=None):
        """Resets every displayed Tracking performance / Gimbal value in Sim View back
        to a blank slate, plus the error-over-time chart and the simulation-time gauge.
        Called whenever Load scene successfully loads a (new or the same) scenario —
        acquisition time, elapsed time, etc. all start over from 0 rather than keeping
        whatever was last recorded. `duration` (seconds) sets the gauge's new total;
        omit it to just reset progress without changing the total (e.g. Reset scenario
        re-loading the same scene keeps that scene's duration)."""
        for lv in s.k:
            lv.setText('-')
        for lab in s.r.values():
            lab.setText('-')
        s.rt_stats = dict(rmse=None, lock_retention=0.0, acquisition_time=None,
                           target_losses=0, accuracy=0.0)
        s.rt_err_history.clear()
        s.chart.update()
        if duration is not None:
            s.sim_duration_total = duration
        s.sim_elapsed = 0.0
        s.sim_gauge.setValue(0)
        s.sim_gauge_label.setText(s._fmt_time_left(s.sim_duration_total))
        # A fresh scene load always clears a time's-up freeze, but does NOT itself
        # restart the gauge or resume tracking — that still only happens once the
        # (now re-enabled) Start Simulation button is pressed, same as on first launch.
        s.sim_gauge_frozen_full = False
        s.sim_gauge_running = False
        s.sim_gauge_started = False
        tt = getattr(s, 'tracking_thread', None)
        if tt is not None and not tt.is_paused():
            tt.pause()
        s.pz.setEnabled(True)
        s.pz.setText('Start Simulation')

    def _fmt_time_left(s, seconds_left):
        seconds_left = max(0, int(seconds_left))
        mm, ss = divmod(seconds_left, 60)
        return f"{mm:02d}:{ss:02d} left"

    def _play_load_scene_animation(s):
        """~2.5s progress-bar fill next to the Load scene button, purely visual
        feedback that a scene just loaded. The button is disabled for the duration
        so a second click can't stack another animation on top of this one."""
        s.load_scene_btn.setEnabled(False)
        s.load_progress.setValue(0); s.load_progress.show()
        anim = QPropertyAnimation(s.load_progress, b'value', s)
        anim.setDuration(2500); anim.setStartValue(0); anim.setEndValue(100)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def _finish():
            s.load_progress.hide(); s.load_scene_btn.setEnabled(True)

        anim.finished.connect(_finish)
        s._load_scene_anim = anim   # keep a reference — an unparented QPropertyAnimation
        anim.start()                # can get garbage-collected mid-animation otherwise

    def on_sat_view_changed(s, name):
        """Switching between 'Satellite Pov' and 'Satellite motion view' logs the choice
        and calls the matching TrackingThread method right away (this combo only lives on
        Sim View, so the user is necessarily on that tab when it fires)."""
        s.selected_sat_view = name
        s.on_status_changed(f"Unity view mode selected: {name}")
        s._invoke_sat_view(name)

    def _invoke_sat_view(s, name):
        """Calls backend.TrackingThread.sat_pov() for 'Satellite Pov', or .second_view()
        for 'Satellite motion view' (and as the fallback for any other value)."""
        tt = getattr(s, 'tracking_thread', None)
        if tt is None:
            return
        if name == 'Satellite Pov':
            tt.sat_pov()
        else:
            tt.second_view()

    # ============================================================
    # SCENE-PARAMETERS LAYOUT (Figma: Beacon Settings / Satellite Settings /
    # Disturbances / Camera Settings, 2x2 grid of panels).
    #
    # Every panel's rows live in ONE scene_grid() per panel (builders defined
    # near panel()/row(), under "Scene-Parameters builders") so labels and
    # controls line up down the whole panel instead of each row picking its own
    # spacing. To add/remove/rename a control, edit the grid_full(...)/
    # grid_half(...) calls below; each one also registers its widget in
    # s.scene_widgets so Save picks up new controls automatically. Camera
    # Settings has no controls yet per spec — its panel is just a placeholder.
    # ============================================================
    def build_scene_params(s):
        save_btn = btn('Save', False, s.save_scene_settings)
        s.scene_filename = QLineEdit(); s.scene_filename.setPlaceholderText('File name')
        s.scene_filename.setText('scene_settings1'); s.scene_filename.setFixedWidth(160)
        hd, _ = header('Scene parameters', 'Configure the simulated scene before running Sim View.',
                       s.scene_filename, save_btn)

        s.scene_widgets = {}   # {panel title: {control label: widget}} — read back by save_scene_settings()
        def reg(group, label, widget):
            s.scene_widgets.setdefault(group, {})[label] = widget
            return widget

        # ---- Beacon Settings: paired grid (label|control|label|control) so the
        # two half-width rows (Colour+Blink, Shape+Size) line up with each other
        # and with the two full-width rows (Start Position, Movement Type).
        G = 'Beacon Settings'
        bw, bg = scene_grid(paired=True)
        grid_full(bg, 0, 'Start Position', reg(G, 'Start Position',
            mk_combo(['Out Of Fov', 'Center', 'Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right', 'Random'])))
        grid_half(bg, 1, 0, 'Colour', reg(G, 'Colour', mk_combo(['Red', 'Green', 'Blue', 'White', 'Yellow', 'Orange'])))
        grid_half(bg, 1, 1, 'Blink', reg(G, 'Blink', mk_toggle(True)))
        grid_half(bg, 2, 0, 'Shape', reg(G, 'Shape', mk_combo(['Circle', 'Square', 'Triangle', 'Cross'])))
        grid_half(bg, 2, 1, 'Size', reg(G, 'Size', mk_combo(['2px', '4px', '6px', '8px', '10px', '12px'])))
        grid_full(bg, 3, 'Movement Type', reg(G, 'Movement Type', mk_combo(['Straight Horizontal', 'Straight Vertical', 'Figure 8', 'Circular', 'Mixed', 'Random'])))
        beacon_panel = panel_top('Beacon Settings', 'beacon', bw)

        # ---- Satellite Settings: single-column grid, every control fills the
        # full remaining row width.
        G = 'Satellite Settings'
        sw, sg = scene_grid()
        dist_row, dist_sl = mk_slider(0, 300, 100)
        grid_full(sg, 0, 'Distance from beacon', dist_row); reg(G, 'Distance from beacon', dist_sl)
        grid_full(sg, 1, 'No. Of Beacons', reg(G, 'No. Of Beacons', mk_combo(['1', '2', '3', '4', '5'])))
        grid_full(sg, 2, 'Extra Beacon Position', reg(G, 'Extra Beacon Position', mk_combo(['Random', 'Fixed'])))
        grid_full(sg, 3, 'Satellite Position', reg(G, 'Satellite Position', mk_combo(['Random', 'Fixed'])))
        grid_full(sg, 4, 'Satellite Rotation', reg(G, 'Satellite Rotation', mk_combo(['Random', 'Fixed'])))
        satellite_panel = panel_top('Satellite Settings', 'satellite', sw)

        # ---- Disturbances: single-column grid of slider rows, all the same
        # height, so they sit evenly spaced instead of one row stretching to
        # fill whatever height the panel ends up with.
        G = 'Disturbances'
        dw, dg = scene_grid()
        # Noise/Blur/Atmospheric Disturbance: raw slider steps 0-100 with divisor=100
        # display as 0.00-1.00 in 0.01 increments (one slider step = 0.01), matching
        # the same 0-100/divisor-100/default-0 pattern Sim View's own Scenario controls
        # sliders already use for these same three (see PARAMS near the top of this file).
        for i, (label, mn, mx, default_raw, dv) in enumerate((
            ('Noise Level', 0, 100, 0, 100), ('Blur Level', 0, 100, 0, 100), ('Atmospheric Disturbance', 0, 100, 0, 100),
            ('Max Standard Deviation (px)', 0, 20, 5, 1), ('Max Camera Jitter (px)', 0, 20, 3, 1), ('Platform Motion', 0, 10, 1, 1),
        )):
            row_w, sl = mk_slider(mn, mx, default_raw, divisor=dv)
            grid_full(dg, i, label, row_w); reg(G, label, sl)
        disturbances_panel = panel_top('Disturbances', 'disturbance', dw)

        # ---- Camera Settings: Simulation time (30s-5min, 5s per slider notch).
        # Reusing mk_slider()'s existing divisor mechanism rather than adding a new
        # parameter: raw slider notches run 6-60, divisor=0.2 turns that into
        # displayed/saved seconds via value/divisor = value*5 (30s .. 300s), so every
        # notch is exactly 5 real seconds whether dragged or stepped with arrow keys.
        G = 'Camera Settings'
        cmw, cmg = scene_grid()
        sim_time_row, sim_time_sl = mk_slider(6, 60, 18, suffix=' s', divisor=0.2)   # 18*5 = 90s default
        grid_full(cmg, 0, 'Simulation time', sim_time_row); reg(G, 'Simulation time', sim_time_sl)
        # Max Horizontal Pan / Max Vertical Tilt: 5°-20° in 0.5° steps — raw slider
        # notches 10-40 with divisor=2, so every notch is exactly 0.5° (same mechanism
        # as Simulation time above). Saved to the scene file as plain degree values.
        pan_row, pan_sl = mk_slider(10, 40, 20, suffix='°', divisor=2)    # 10° default
        grid_full(cmg, 1, 'Max Horizontal Pan', pan_row); reg(G, 'Max Horizontal Pan', pan_sl)
        tilt_row, tilt_sl = mk_slider(10, 40, 20, suffix='°', divisor=2)   # 10° default
        grid_full(cmg, 2, 'Max Vertical Tilt', tilt_row); reg(G, 'Max Vertical Tilt', tilt_sl)
        soon = QLabel('more settings will be provided soon.'); soon.setObjectName('soon')
        cmg.setRowMinimumHeight(3, 26); cmg.addWidget(soon, 4, 0, 1, 4)
        camera_panel = panel_top('Camera Settings', 'camera', cmw)

        w, l = page(hd, row(beacon_panel, satellite_panel), row(disturbances_panel, camera_panel), stretch=[0, 1, 1])
        return w

    def save_scene_settings(s):
        """Save button on the Scene-Parameters tab. Reads every control currently
        registered in s.scene_widgets (built in build_scene_params(), grouped by
        panel) and writes them to <file name field>.txt inside SCENES_DIR (see the
        constant near the top of this file), one 'Label: value' line per control
        under its panel's name. The file name comes from the text field beside
        Save (defaults to 'scene_settings1' if left blank); '.txt' is appended
        automatically if not typed. Overwrites the file on every click. Also
        refreshes the Sim View "Test case scene" dropdown so a newly saved scene
        shows up there immediately."""
        def value_of(w):
            if isinstance(w, QComboBox): return w.currentText()
            if isinstance(w, QSlider):
                dv = w.property('divisor') or 1
                return f"{w.value() / dv:g}"
            if isinstance(w, QPushButton) and w.isCheckable(): return 'ON' if w.isChecked() else 'OFF'
            return ''
        lines = []
        for group, fields in s.scene_widgets.items():
            lines.append(f"{group}:")
            if not fields:
                lines.append("  (not configured yet)")
            for label, widget in fields.items():
                lines.append(f"  {label}: {value_of(widget)}")
            lines.append("")
        name = s.scene_filename.text().strip() or 'scene_settings1'
        if not name.lower().endswith('.txt'): name += '.txt'
        path = os.path.join(SCENES_DIR, name)
        try:
            os.makedirs(SCENES_DIR, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines).rstrip() + '\n')
            s.refresh_scene_list()
            s.refresh_bench_scene_list()
            QMessageBox.information(s, 'Scene parameters', f'Saved to {path}')
        except Exception as e:
            QMessageBox.warning(s, 'Scene parameters', f'Could not save settings: {e}')

    # ---- Video View
    def build_video(s):
        hd, s.vsub = header('Video view', 'Import an MP4 to run beacon detection on your own footage.')
        v = s.video; s.vcam = CamView(v)
        # Real Unity view container (replaces the old painted SatView fake scene). This is
        # a plain native-window widget, exactly like s.unity_container on Sim View — the
        # real Unity HWND gets re-parented into whichever of the two is visible, via
        # _reparent_unity_to_tab(), so switching tabs moves the live embedded view with it.
        s.vsat = QWidget()
        s.vsat.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        s.vsat.setStyleSheet('background-color:#070c18; border-radius:9px;')
        s.vsat.setMinimumSize(230, 140)
        s.path = QLineEdit(); s.path.setReadOnly(True); s.path.setPlaceholderText('File location')
        top = QWidget(); tl = QHBoxLayout(top); tl.setContentsMargins(0, 0, 0, 0); tl.addWidget(btn('Import MP4', False, s.import_video)); tl.addWidget(s.path)
        s.vp = btn('Pause', False, s.toggle_video); s.vp.setEnabled(False); s.scrub = QSlider(Qt.Orientation.Horizontal); s.scrub.setRange(0, 1000)
        s.scrub.sliderMoved.connect(s.seek); s.tm = QLabel('0s'); s.spd = QComboBox(); s.spd.addItems(['0.25x', '0.5x', '1x', '2x', '4x']); s.spd.setCurrentIndex(2)
        s.spd.currentIndexChanged.connect(s.on_speed_changed)
        ctl = QWidget(); cl = QHBoxLayout(ctl); cl.setContentsMargins(0, 0, 0, 0)
        for x in (s.vp, s.scrub, s.tm, s.spd): cl.addWidget(x)
        src = QWidget(); sl = QVBoxLayout(src); sl.setContentsMargins(0, 0, 0, 0); sl.addWidget(top); sl.addWidget(QLabel('Video controls')); sl.addWidget(ctl)
        s.dt = table(['Time', 'X offset', 'Y offset', 'State'], 5); s.vchart = Chart(lambda: v.err, 150, auto=True)
        s.vview = QComboBox(); s.vview.addItems(['Python side view', 'Video View']); s.vview.currentIndexChanged.connect(s.on_vview_changed)
        w, l = page(hd, row(panel_with_header_control('Python-side overview', s.vview, s.vcam), panel('3D camera pan/tilt', s.vsat)),
                    row(panel('Source', src), panel('Latest detections', s.dt)), panel('Beacon offset from centre (px)', s.vchart), stretch=[0, 5, 3, 2]); return w

    def import_video(s):
        p, _ = QFileDialog.getOpenFileName(s, 'Import video', '', 'Video (*.mp4 *.avi *.mov *.mkv)')
        if not p: return
        probe = cv2.VideoCapture(p); ok = probe.isOpened(); probe.release()
        if not ok: return QMessageBox.warning(s, 'Import video', 'Could not open that video file.')
        s.stop_video()
        v = s.video; v.reset(); v.active = True; v.paused = False
        w = VideoWorker(p); w.speed = s.speed
        w.result.connect(s.on_video_result); w.status.connect(s.on_video_status)
        w.failed.connect(s.on_video_failed); w.ended.connect(s.on_video_ended)
        v.worker = w; w.start()
        s.path.setText(p); s.vsub.setText('Loading YOLO model...'); s.vp.setText('Pause'); s.vp.setEnabled(True); s.scrub.setValue(0)

    def stop_video(s):
        """Stop the video + detection thread and close the OpenCV preview window."""
        v = s.video; w = v.worker
        if w is not None:
            w.stop(); w.wait(); v.worker = None
        v.active = False

    def toggle_video(s):
        """Pause/resume Video View ONLY. Pause stops both the MP4 playback and the beacon
        detection; Play resumes both. Deliberately does not touch tracking_thread — that
        belongs to Sim View's own Pause button (toggle_sim), kept fully separate."""
        v = s.video
        if not v.active or v.worker is None: return
        v.paused = not v.paused
        if v.paused: v.worker.pause()
        else: v.worker.play()
        s.vp.setText('Play' if v.paused else 'Pause')

    def seek(s, val):
        # detection restarts (re-acquires) from the new position
        if s.video.worker is not None: s.video.worker.seek_to(val / 1000)

    def on_vview_changed(s, i):
        s.vcam.mode = 'raw' if i == 1 else 'py'; s.vcam.update()

    def on_speed_changed(s, i):
        s.speed = (.25, .5, 1, 2, 4)[i]
        # VideoWorker.run() paces frames with `next_t += 1 / (fps * s.speed)` (see near
        # the top of this file), so this actually changes how fast the video plays and
        # is detected — it is not just a label change in the UI.
        if s.video.worker is not None: s.video.worker.speed = s.speed

    def on_video_result(s, img, raw, info):
        v = s.video
        if s.sender() is not v.worker: return            # late frame from a replaced worker
        v.update(img, raw, info)
        # Feed the Video View's own pan/tilt (from BeaconTracker, independent of Sim View's
        # TrackingThread loop) into backend.TrackingThread for as long as this video plays.
        tt = getattr(s, 'tracking_thread', None)
        if tt is not None and hasattr(tt, 'placehold'):
            tt.placehold([float(info['pan']), float(info['tilt'])])
        if not s.scrub.isSliderDown():
            s.scrub.blockSignals(True); s.scrub.setValue(int(1000 * info['pos_frac'])); s.scrub.blockSignals(False)

    def on_video_status(s, text):
        if s.sender() is s.video.worker: s.vsub.setText(text)

    def on_video_failed(s, text):
        if s.sender() is not s.video.worker: return
        s.stop_video(); s.video.reset(); s.vp.setEnabled(False); s.vp.setText('Pause'); s.path.clear()
        s.vsub.setText('Import an MP4 to run beacon detection on your own footage.')
        QMessageBox.warning(s, 'Video detection', text)

    def on_video_ended(s, msg):
        if s.sender() is not s.video.worker: return
        s.video.paused = True; s.vp.setText('Play'); s.vsub.setText(f'Video finished. {msg}. Press Play to run it again.')
        print(msg)

    # ---- Benchmark
    def build_bench(s):
        hd, _ = header('Benchmark', 'Run a scenario suite and save the results to Data-Check')
        s.bname = QLineEdit('Run 2'); s.btype = QComboBox()
        s.btype.addItems(['Benchmark Performance-1 (scenario suite)', 'Benchmark Performance-2 (video-based)', 'Disturbance sweep', 'Custom'])
        det = QWidget(); dl = QGridLayout(det); dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(QLabel('Name'), 0, 0); dl.addWidget(s.bname, 1, 0); dl.addWidget(QLabel('Benchmark type'), 0, 1); dl.addWidget(s.btype, 1, 1)
        spec = QLabel('Pass targets from the problem statement — Acquisition ≤ 2 s · Tracking error ≤ 10 px · '
                       'Target loss < 5% · Re-acquisition ≤ 1 s · Processing ≥ 20 FPS')
        spec.setObjectName('mut'); spec.setWordWrap(True)

        # ---- "Criteria" tab — unchanged from before, just moved into a tab instead
        # of being the left panel's only content.
        s.crit = [QCheckBox(c) for c in CR]; cw = QWidget(); cl = QVBoxLayout(cw); cl.setContentsMargins(12, 12, 12, 12)
        for c in s.crit: c.setChecked(True); cl.addWidget(c)
        s.go_b = btn('Run benchmark', False, s.run_bench); cl.addWidget(s.go_b); cl.addStretch()

        # ---- "Scenarios" tab — a checklist of every .txt scene file in SCENES_DIR
        # (the same source the Sim View "Test case scene" dropdown pulls from — see
        # refresh_scene_list()), with a live "N scenes ready" count and its own Run
        # benchmark button below the list. The count label and button live OUTSIDE
        # the QScrollArea (in scen_tab's own layout, not scen_list_w's), so they stay
        # pinned at the bottom of the tab regardless of how many scenarios there are —
        # only the checkbox list itself scrolls. refresh_bench_scene_list() (re)builds
        # just that checkbox list and is called here once, then again wherever the Sim
        # View equivalent (refresh_scene_list()) is: on Save in Scene-Parameters, and
        # whenever the Benchmark tab is opened (see go()).
        scen_list_w = QWidget(); s.bench_scen_layout = QVBoxLayout(scen_list_w)
        s.bench_scen_layout.setContentsMargins(0, 0, 0, 0); s.bench_scen_layout.setSpacing(6)
        s.bench_scen_checks = []   # [(QCheckBox, scene name), ...], rebuilt by refresh_bench_scene_list()
        scen_scroll = scroll_wrap(scen_list_w, h_as_needed=False)

        s.bench_scen_count = QLabel('0 scenes ready for benchmark running')
        s.bench_scen_count.setObjectName('mut'); s.bench_scen_count.setWordWrap(True)
        s.go_b_scen = btn('Run benchmark', False, s.run_bench)

        scen_tab = QWidget(); stl = QVBoxLayout(scen_tab)
        stl.setContentsMargins(12, 12, 12, 12); stl.setSpacing(10)
        stl.addWidget(scen_scroll, 1)          # only this part scrolls
        stl.addWidget(s.bench_scen_count)      # pinned footer …
        stl.addWidget(s.go_b_scen)             # … stays visible no matter how long the list is

        s.refresh_bench_scene_list()

        tabs = QTabWidget()
        tabs.tabBar().setExpanding(True)   # Criteria/Scenarios split the header evenly instead of hugging the left edge
        tabs.addTab(cw, 'Criteria'); tabs.addTab(scen_tab, 'Scenarios')

        # Same framed-panel look as panel()'s QFrame#panel styling, just without a
        # single fixed title label since the tab headers themselves serve that role.
        left = QFrame(); left.setObjectName('panel'); ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 14, 16, 16); ll.setSpacing(10); ll.addWidget(tabs, 1)

        s.bar = QProgressBar(); s.bar.setRange(0, 1); s.bar.setFormat('%v/1')
        s.bench_time_left = QLabel('Select scenarios and press Run benchmark'); s.bench_time_left.setObjectName('mut')
        s.blog = table(['Run', 'Scenario', 'Result'])
        s.blog.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(panel('Progress', s.bench_time_left, s.bar)); rl.addWidget(panel('Progress log', s.blog), 1)
        body = row(left, right); body.layout().setStretch(1, 1); body.layout().itemAt(0).widget().setFixedWidth(270)
        w, l = page(hd, panel('Benchmark details', det, spec), body, stretch=[0, 0, 1]); return w

    def refresh_bench_scene_list(s):
        """(Re)builds the Scenarios tab's checkbox list from SCENES_DIR's .txt files —
        same source and file-name convention as refresh_scene_list() (Sim View's Test
        case scene dropdown). Keeps whichever scenes were already checked, checked.
        s.bench_scen_layout only ever holds checkboxes (plus its own trailing stretch)
        now — the count label and Run benchmark button live in scen_tab's own layout
        instead (see build_bench()), so they're untouched by this rebuild."""
        try:
            names = sorted(os.path.splitext(os.path.basename(p))[0]
                            for p in glob.glob(os.path.join(SCENES_DIR, '*.txt')))
        except OSError:
            names = []
        previously_checked = {name for cb, name in s.bench_scen_checks if name is not None and cb.isChecked()}
        # Fully clear the layout (checkboxes AND the stretch spacer) — addStretch()
        # adds a spacer item, not a widget, so only clearing tracked checkboxes would
        # leave one behind on every refresh and the list would drift down a bit more
        # each time.
        while s.bench_scen_layout.count():
            item = s.bench_scen_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        s.bench_scen_checks = []
        # The Scenarios tab's list has no horizontal scrollbar (by design — see
        # scen_scroll's h_as_needed=False in build_bench()), so a long scene filename
        # would otherwise just get silently clipped with no way to see the rest. Elide
        # it to a width that comfortably fits the fixed-270px left panel instead, and
        # keep the full name available as a tooltip.
        elide_width = 150
        for name in names:
            fm = QFontMetrics(QFont())
            display = fm.elidedText(name, Qt.TextElideMode.ElideRight, elide_width)
            cb = QCheckBox(display); cb.setChecked(name in previously_checked)
            if display != name: cb.setToolTip(name)
            cb.stateChanged.connect(s._update_bench_scene_count)
            s.bench_scen_layout.addWidget(cb)
            s.bench_scen_checks.append((cb, name))
        if not names:
            none_lbl = QLabel(f'No scene files found in:\n{SCENES_DIR}'); none_lbl.setObjectName('mut'); none_lbl.setWordWrap(True)
            s.bench_scen_layout.addWidget(none_lbl); s.bench_scen_checks.append((none_lbl, None))
        s.bench_scen_layout.addStretch()
        s._update_bench_scene_count()

    def _update_bench_scene_count(s):
        n = sum(1 for cb, name in s.bench_scen_checks if name is not None and cb.isChecked())
        s.bench_scen_count.setText(f"{n} scene{'s' if n != 1 else ''} ready for benchmark running")

    # ============================================================
    # BENCHMARK PERFORMANCE-1 — real run driven through Sim View's own
    # backend.TrackingThread, one scenario at a time, for exactly the
    # "Simulation time" recorded in each scene's .txt file (PS #26169,
    # Benchmark Performance-1: "Log of centroiding error" + "automatically
    # generated performance logs").
    # ============================================================

    def run_bench(s):
        if s.bench_running: return
        selected = [name for cb, name in s.bench_scen_checks if name is not None and cb.isChecked()]
        if not selected:
            QMessageBox.warning(s, 'Benchmark', 'Check at least one scenario in the Scenarios tab first.')
            return
        tt = getattr(s, 'tracking_thread', None)
        if tt is None:
            QMessageBox.warning(s, 'Benchmark', 'Tracking thread is not available.')
            return
        s.bench_cols = [i for i, c in enumerate(s.crit) if c.isChecked()] or list(range(len(CR)))
        s.bench_queue = list(selected)
        s.bench_durations = {name: s._read_scene_duration(name) for name in s.bench_queue}
        s.bench_total_time = sum(s.bench_durations.values())
        s.bench_elapsed_done = 0.0
        s.bench_idx = -1
        s.bench_rows = []
        s.bench_frames_by_scene = {}
        s.blog.setRowCount(0)
        s.bar.setRange(0, len(s.bench_queue)); s.bar.setFormat(f'%v/{len(s.bench_queue)}'); s.bar.setValue(0)
        s.bench_time_left.setText(s._fmt_time_left(s.bench_total_time) + f'  ·  {len(s.bench_queue)} scenario(s) queued')
        s.bench_running = True
        s.go_b.setText('Running'); s.go_b_scen.setText('Running')
        s.go_b.setEnabled(False); s.go_b_scen.setEnabled(False)
        s._set_bench_busy(True)
        s._bench_advance_scene()

    def _bench_advance_scene(s):
        """Loads the next queued scenario through the exact same call Sim View's
        "Load scene" button uses (backend.TrackingThread.parsed()), which itself
        resets acquisition/Kalman/metrics and resumes tracking. Also mirrors the
        reset onto the Sim View tab (clear_sim_view_stats) so the run is visible
        there too, then explicitly resumes since clear_sim_view_stats pauses."""
        s.bench_idx += 1
        if s.bench_idx >= len(s.bench_queue):
            return
        name = s.bench_queue[s.bench_idx]
        dur = s.bench_durations[name]
        s.bench_current_duration = dur
        s.bench_frames = []
        s.bench_frames_by_scene[name] = s.bench_frames
        tt = s.tracking_thread
        try:
            tt.parsed(name)
        except Exception as e:
            s.on_status_changed(f"Benchmark: could not load '{name}': {e}")
            s._bench_finish_scene()
            return
        s.clear_sim_view_stats(duration=dur)
        tt.resume()
        s.on_status_changed(f"Benchmark: running '{name}' for {dur:.0f}s "
                             f"({s.bench_idx + 1}/{len(s.bench_queue)})")

    def _bench_on_stats(s, stats):
        """Hooked from on_stats_ready() while a benchmark is running. Records every
        real frame's stats for this scenario and checks whether the scenario's own
        configured simulation time has now elapsed."""
        if not s.bench_running or s.bench_idx < 0 or s.bench_idx >= len(s.bench_queue):
            return
        s.bench_frames.append(dict(stats))
        elapsed = stats.get('elapsed', 0.0) or 0.0
        dur = s.bench_current_duration
        remaining = max(0.0, s.bench_total_time - s.bench_elapsed_done - min(elapsed, dur))
        s.bench_time_left.setText(s._fmt_time_left(remaining) +
                                   f'  ·  scenario {s.bench_idx + 1}/{len(s.bench_queue)}')
        if elapsed >= dur:
            s._bench_finish_scene()

    def _bench_aggregate(s, frames, duration):
        """Turns one scenario's raw stats_ready stream into the 7 Data-Check
        criteria — all read straight off backend.py's own running metrics, taken
        from the last frame processed for this scenario (RMSE/lock retention/
        accuracy/target-loss count are already cumulative-since-reset there)."""
        if not frames:
            return (duration, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
        last = frames[-1]
        acq = last.get('acquisition_time')
        acq = acq if acq is not None else duration   # never acquired inside the scenario's time budget
        rmse = last.get('rmse') or 0.0
        losses = int(last.get('target_losses') or 0)
        fps_vals = [f.get('fps') for f in frames if f.get('fps')]
        fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0
        update_ms = (1000.0 / fps) if fps > 0 else (duration * 1000.0 / len(frames))
        # Lock retention (%) — the fraction of frames locked onto the beacon since
        # acquisition, cumulative-since-reset in backend.TrackingThread and reported
        # on every stats_ready frame, so the last frame's value is the scenario total.
        lock_retention = last.get('lock_retention') or 0.0
        # Tracking RMSE (px) — same underlying backend.py 'rmse' stream as the
        # "Tracking error (px)" column above; shown as its own criterion because
        # PS #26169's Benchmark Performance-2 evaluation table names RMSE
        # separately from centroiding/tracking error.
        return (round(acq, 2), round(rmse, 2), losses, round(fps), round(update_ms, 1),
                round(lock_retention, 1), round(rmse, 2))

    def _bench_finish_scene(s):
        name = s.bench_queue[s.bench_idx]
        dur = s.bench_current_duration
        acq, err, loss, fps, upd, lock, rmse = s._bench_aggregate(s.bench_frames, dur)
        s.bench_rows.append((name, acq, err, loss, fps, upd, lock, rmse))
        s.bench_elapsed_done += dur
        s.bar.setValue(s.bench_idx + 1)
        s.blog.insertRow(0)
        for c, t in enumerate((str(s.bench_idx + 1), name, f"{acq:.2f}s acq, {err:.1f}px RMSE, {loss} lost")):
            s.blog.setItem(0, c, QTableWidgetItem(t))
        if s.bench_idx + 1 >= len(s.bench_queue):
            s._bench_finish_all()
        else:
            s._bench_advance_scene()

    def _bench_finish_all(s):
        from datetime import datetime
        tt = getattr(s, 'tracking_thread', None)
        if tt is not None and not tt.is_paused():
            tt.pause()
        entry = dict(name=s.bname.text().strip() or 'Untitled', type=s.btype.currentText(),
                     date=datetime.now().strftime('%d %b %Y, %H:%M'), cols=s.bench_cols, rows=s.bench_rows)
        s.bench.insert(0, entry)
        s.sel = 0
        s.bench_running = False
        s.go_b.setText('Run benchmark'); s.go_b_scen.setText('Run benchmark')
        s.go_b.setEnabled(True); s.go_b_scen.setEnabled(True)
        s._set_bench_busy(False)
        s.bname.setText(f"Run {len(s.bench) + 1}")
        s.draw_data()
        s.blog.insertRow(0); s.blog.setItem(0, 1, QTableWidgetItem('Saved to Data-Check.'))
        s.bench_time_left.setText('Done — generating reports…')
        s._bench_export_reports(entry)

    def _bench_export_reports(s, entry):
        """Writes the Excel Performance Log + Word report named after Benchmark
        Details' Name field (see benchmark_report.py). Failure here never loses
        the benchmark itself — it's already saved to Data-Check by this point."""
        if benchmark_report is None:
            s.bench_time_left.setText('Done. Reports NOT written — see warning.')
            QMessageBox.warning(s, 'Benchmark reports',
                                 f'Benchmark saved to Data-Check, but the Excel/Word report could not be '
                                 f'generated:\n\n{BENCH_REPORT_ERROR}\n\n'
                                 f'Install the missing package(s) (pip install openpyxl python-docx) and '
                                 f're-run the benchmark to get the report files.')
            return
        try:
            xlsx_path, docx_path = benchmark_report.export_reports(entry, s.bench_frames_by_scene, REPORTS_DIR)
        except Exception as e:
            s.bench_time_left.setText('Done. Report generation failed — see warning.')
            QMessageBox.warning(s, 'Benchmark reports', f'Could not write the report files:\n\n{e}')
            return
        s.bench_time_left.setText(f'Done — reports saved to {REPORTS_DIR}')
        s.on_status_changed(f"Benchmark reports saved: {os.path.basename(xlsx_path)}, {os.path.basename(docx_path)}")

    # ---- Data-Check
    def build_data(s):
        hd, _ = header('Data-Check', 'Saved benchmarks, spec compliance and trends across runs')
        s.bl = QListWidget(); s.bl.currentRowChanged.connect(lambda i: i >= 0 and (setattr(s, 'sel', i), s.draw_details()))
        s.add_bench_btn = btn('Add benchmark', False, s._show_add_benchmark_menu)
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.addWidget(s.add_bench_btn); ll.addWidget(s.bl)
        s.dn = QLabel(); s.dn.setObjectName('h3'); s.dmeta = QLabel(); s.dmeta.setObjectName('mut'); s.kpi = QLabel(); s.kpi.setTextFormat(Qt.TextFormat.RichText)
        s.dtab = table(['Scenario'])
        vb = btn('View in Video tab', True, lambda: (s.go(2), s.vsub.setText(f"Reviewing benchmark: {s.bench[s.sel]['name']}. Import its MP4 to replay.")))
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0); tr = QHBoxLayout(); tr.addWidget(s.dn); tr.addStretch(); tr.addWidget(vb)
        rl.addLayout(tr); rl.addWidget(s.dmeta); rl.addWidget(s.kpi); rl.addWidget(s.dtab, 1)
        # criterion picker + per-scenario bar chart (this benchmark) + trend chart (across all saved benchmarks)
        s.dcrit = mk_combo(CR); s.dcrit.setCurrentIndex(1)
        s.dcrit.currentIndexChanged.connect(lambda _: s.draw_details())
        s.dbar = BarChart(lambda: s._bar_data(), fmt='{:.1f}')
        s.dtrend = BarChart(lambda: s._trend_data(), fmt='{:.1f}')
        choose = QWidget(); chl = QHBoxLayout(choose); chl.setContentsMargins(0, 0, 0, 0)
        chl.addWidget(QLabel('Criterion')); chl.addWidget(s.dcrit); chl.addStretch()
        charts = row(panel('By scenario (this benchmark)', s.dbar), panel('By run (all saved benchmarks)', s.dtrend))
        body = row(panel('Benchmark list', left), panel('Details', right)); body.layout().setStretch(1, 1); body.layout().itemAt(0).widget().setFixedWidth(270)
        w, l = page(hd, body, panel('Compliance charts', choose, charts), stretch=[0, 3, 2]); return w

    def _show_add_benchmark_menu(s):
        """'Add benchmark' button — offers a choice instead of jumping straight to
        the Benchmark tab: run a fresh scenario suite here, or import an already-
        exported Performance Log .xlsx (e.g. from a teammate's run) as a benchmark."""
        menu = QMenu(s)
        menu.addAction('Run Custom Benchmark', lambda: s.go(3))
        menu.addAction('Import Benchmark', s.import_benchmark)
        menu.exec(s.add_bench_btn.mapToGlobal(s.add_bench_btn.rect().bottomLeft()))

    def import_benchmark(s):
        if s.bench_running:
            QMessageBox.warning(s, 'Import benchmark', 'Wait for the current benchmark run to finish first.')
            return
        path, _ = QFileDialog.getOpenFileName(s, 'Import benchmark', REPORTS_DIR, 'Excel files (*.xlsx)')
        if not path:
            return
        try:
            entry = s._read_benchmark_xlsx(path)
        except Exception as e:
            QMessageBox.warning(s, 'Import benchmark', f'Could not read this file as a benchmark:\n\n{e}')
            return
        s.bench.insert(0, entry)
        s.sel = 0
        s.draw_data()
        s.on_status_changed(f"Benchmark imported: {entry['name']}")

    def _read_benchmark_xlsx(s, path):
        """Reads a Performance Log .xlsx of the kind export_excel() (benchmark_report.py)
        writes — the 'Summary' and 'Scenario results' sheets — and turns it back into
        a Data-Check benchmark entry, so an exported run (this app's or a teammate's)
        can be re-imported and plotted alongside benchmarks run here.
        Columns are matched by header name, not fixed position, so it tolerates any
        extra trailing columns the sheet carries (Sim. duration, Accuracy, ...)."""
        from datetime import datetime
        from openpyxl import load_workbook
        wb = load_workbook(path, data_only=True)
        if 'Scenario results' not in wb.sheetnames:
            raise ValueError("No 'Scenario results' sheet found — this doesn't look like an exported Performance Log.")
        ws = wb['Scenario results']
        header_row = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        if not header_row or header_row[0] != 'Scenario':
            raise ValueError("Unexpected header row in 'Scenario results'.")
        col_idx = {name: header_row.index(name) for name in CR if name in header_row}
        if not col_idx:
            raise ValueError('None of the known criteria columns were found in this sheet.')
        rows = []
        for r in ws.iter_rows(min_row=2, values_only=True):
            if not r or r[0] in (None, ''):
                continue
            vals = []
            for name in CR:
                j = col_idx.get(name)
                v = r[j] if (j is not None and j < len(r)) else None
                vals.append(float(v) if v not in (None, '') else 0.0)
            rows.append(tuple([r[0]] + vals))
        if not rows:
            raise ValueError("No scenario rows found in 'Scenario results'.")
        name = os.path.splitext(os.path.basename(path))[0]
        btype, date = 'Imported', datetime.now().strftime('%d %b %Y, %H:%M')
        if 'Summary' in wb.sheetnames:
            for row in wb['Summary'].iter_rows(min_row=1, max_row=6, values_only=True):
                if not row or row[0] is None: continue
                if row[0] == 'Benchmark name' and row[1]: name = str(row[1])
                elif row[0] == 'Type' and row[1]: btype = str(row[1])
                elif row[0] == 'Date' and row[1]: date = str(row[1])
        cols = [i for i, c in enumerate(CR) if c in col_idx]
        return dict(name=name, type=btype, date=date, cols=cols, rows=rows)

    def draw_data(s):
        s.bl.blockSignals(True); s.bl.clear()
        for b in s.bench: s.bl.addItem(QListWidgetItem(f"{b['name']}\n{b['type']}, {len(b['rows'])} runs"))
        s.bl.setCurrentRow(s.sel); s.bl.blockSignals(False); s.draw_details()

    def _bar_data(s):
        b = s.bench[s.sel]; j = s.dcrit.currentIndex()
        return [r[0] for r in b['rows']], [r[j + 1] for r in b['rows']]

    def _trend_data(s):
        j = s.dcrit.currentIndex()
        labels = [b['name'] for b in s.bench][::-1]
        values = [sum(r[j + 1] for r in b['rows']) / len(b['rows']) for b in s.bench][::-1]
        return labels, values

    def draw_details(s):
        b = s.bench[s.sel]; cols = b['cols']; s.dn.setText(b['name']); s.dmeta.setText(f"{b['type']}  ·  {b['date']}")
        avg = lambda j: sum(r[j + 1] for r in b['rows']) / len(b['rows'])
        def badge(j):
            th = SPEC_THRESH[j]
            if not th: return ''
            lim, lower = th; v = avg(j); ok = (v <= lim) if lower else (v >= lim)
            return f" <span style='background:{T['ok'] if ok else T['bc']};color:{T['onac']};border-radius:6px;padding:1px 6px;font-size:10px;font-weight:800'>{'PASS' if ok else 'FAIL'}</span>"
        s.kpi.setText(''.join(f"<span style='font-size:17px;font-weight:800'>{avg(j):.{FM[j] or 1}f}</span> "
                               f"<span style='color:{T['mut']}'>avg {CR[j]}</span>{badge(j)}&nbsp;&nbsp;&nbsp;&nbsp;" for j in cols))
        s.dtab.setColumnCount(len(cols) + 1); s.dtab.setHorizontalHeaderLabels(['Scenario'] + [CR[j] for j in cols]); s.dtab.setRowCount(len(b['rows']))
        for i, r in enumerate(b['rows']):
            s.dtab.setItem(i, 0, QTableWidgetItem(r[0]))
            for c, j in enumerate(cols):
                it = QTableWidgetItem(f"{r[j + 1]:.{FM[j]}f}"); th = SPEC_THRESH[j]
                if th:
                    lim, lower = th; ok = (r[j + 1] <= lim) if lower else (r[j + 1] >= lim)
                    it.setForeground(QColor(T['ok'] if ok else T['bc']))
                s.dtab.setItem(i, c + 1, it)
        if hasattr(s, 'dbar') and s.dcrit.currentIndex() not in cols: s.dcrit.setCurrentIndex(cols[0])
        if hasattr(s, 'dbar'): s.dbar.thresh = s.dtrend.thresh = SPEC_THRESH[s.dcrit.currentIndex()]; s.dbar.update(); s.dtrend.update()

    # ---- Spec Reference
    def open_spec_pdf(s):
        """'Open problem statement (PDF)' button at the top of the Spec reference tab.
        Opens SPEC_LINK_URL (see the constant near the SPEC_* tables) in the system's
        default browser."""
        if not QDesktopServices.openUrl(QUrl(SPEC_LINK_URL)):
            QMessageBox.warning(s, 'Spec reference', f"Couldn't open {SPEC_LINK_URL} — no default web browser is available.")

    def build_spec(s):
        hd, _ = header('Spec reference', 'Problem Statement 4 · SIH PS #26169 — Coarse alignment of mobile FSOC terminals',
                       btn('Open problem statement (PDF)', False, s.open_spec_pdf))

        # Mid-tone accents chosen to read on both the light and dark palettes; the
        # translucent tints (rgba) sit on whichever panel colour is underneath, so the
        # page looks right in either mode without needing a rebuild on theme change.
        BLUE, GREEN, AMBER, RED, VIOLET, TEAL = '#2f8fd1', '#0f9773', '#c07a1a', '#d24b4b', '#7c5cd6', '#14a3b8'
        CYCLE = (BLUE, GREEN, AMBER, VIOLET, TEAL, RED)
        def tint(hexc, a):
            return f"rgba({int(hexc[1:3], 16)},{int(hexc[3:5], 16)},{int(hexc[5:7], 16)},{a})"

        def cpanel(title, color, *widgets):
            """Section card: accent bar + coloured title, then the content, top-aligned."""
            f = QFrame(); f.setObjectName('panel'); l = QVBoxLayout(f); l.setContentsMargins(22, 18, 22, 20); l.setSpacing(14)
            hdw = QWidget(); hl = QHBoxLayout(hdw); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(10)
            bar = QFrame(); bar.setObjectName('accentBar'); bar.setFixedSize(4, 20)
            bar.setStyleSheet(f"QFrame#accentBar {{ background:{color}; border:0; border-radius:2px; }}")
            ttl = QLabel(title); ttl.setStyleSheet(f"font-size:16px; font-weight:800; color:{color};")
            hl.addWidget(bar); hl.addWidget(ttl); hl.addStretch()
            l.addWidget(hdw)
            for w in widgets: l.addWidget(w)
            l.addStretch(1); return f

        def rich(html):
            lab = QLabel(html); lab.setWordWrap(True); lab.setTextFormat(Qt.TextFormat.RichText); return lab

        def hi(text, color=BLUE):
            return f"<span style='color:{color}; font-weight:700;'>{text}</span>"

        def badge_list(items, color, cols=1):
            w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0); g.setHorizontalSpacing(28); g.setVerticalSpacing(12)
            per_col = -(-len(items) // cols)
            for i, t in enumerate(items):
                it = QWidget(); h = QHBoxLayout(it); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(12)
                b = QLabel(str(i + 1)); b.setFixedSize(24, 24); b.setAlignment(Qt.AlignmentFlag.AlignCenter)
                b.setStyleSheet(f"background:{color}; color:#ffffff; border-radius:12px; font-size:11px; font-weight:800;")
                tx = QLabel(t); tx.setWordWrap(True); tx.setContentsMargins(0, 3, 0, 0)
                h.addWidget(b, 0, Qt.AlignmentFlag.AlignTop); h.addWidget(tx, 1)
                g.addWidget(it, i % per_col, i // per_col)
            for c in range(cols): g.setColumnStretch(c, 1)
            return w

        def card(title, body, color, big=None, name='card'):
            c = QFrame(); c.setObjectName(name)
            c.setStyleSheet(f"QFrame#{name} {{ background:{tint(color, .10)}; border:1px solid {tint(color, .38)}; border-radius:10px; }}")
            v = QVBoxLayout(c); v.setContentsMargins(16, 14, 16, 14); v.setSpacing(5)
            if big:
                bl = QLabel(big); bl.setStyleSheet(f"font-size:28px; font-weight:800; color:{color};"); v.addWidget(bl)
            tl = QLabel(title); tl.setWordWrap(True); tl.setStyleSheet(f"font-size:14px; font-weight:800; color:{color};"); v.addWidget(tl)
            if body:
                d = QLabel(body); d.setWordWrap(True); d.setObjectName('mut'); v.addWidget(d)
            v.addStretch(1); return c

        def spec_table(rows, color, headers=('Parameter', 'Suggested value', 'Remarks'), fit_cols=(0,)):
            t = AutoHeightTable(len(rows), len(headers)); t.setHorizontalHeaderLabels(headers); t.verticalHeader().hide()
            t.setShowGrid(False); t.setFrameShape(QFrame.Shape.NoFrame); t.setAlternatingRowColors(True)
            t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            t.setStyleSheet(f"QHeaderView::section {{ background:{tint(color, .14)}; color:{color}; font-weight:800; border:0; "
                            f"border-bottom:2px solid {color}; padding:9px 12px; text-align:left; }}"
                            f"QTableWidget::item {{ padding:8px 12px; }}")
            for c in range(len(headers)):
                t.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents if c in fit_cols else QHeaderView.ResizeMode.Stretch)
            t.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            bold = QFont(t.font()); bold.setBold(True)
            for r, vals in enumerate(rows):
                for c, val in enumerate(vals):
                    it = QTableWidgetItem(str(val))
                    if c == 0: it.setFont(bold)
                    t.setItem(r, c, it)
            return t

        # ---- Mission brief + functional objective
        brief = rich(
            "<p style='line-height:150%; margin:0 0 9px 0;'>Free-space optical communication (FSOC) offers "
            f"{hi('gigabit-to-terabit data rates')}, licence-free spectrum and high immunity to electromagnetic interference — "
            "but between mobile platforms (satellites, UAVs) its narrow laser beam makes "
            f"{hi('pointing, acquisition and tracking (PAT)')} a severe challenge.</p>"
            f"<p style='line-height:150%; margin:0 0 9px 0;'>PAT happens in two stages. {hi('Coarse alignment', GREEN)} — what this console "
            "simulates — has the transmitting terminal locate the remote terminal and keep it inside its camera field of view; "
            f"{hi('fine alignment', AMBER)} takes over afterwards. Even a small angular error can break the link.</p>"
            "<p style='line-height:150%; margin:0;'>Testing on real cameras, pan-tilt mechanisms and optics is expensive, so a "
            f"{hi('software virtual camera', VIOLET)} gives an inexpensive, accessible platform for developing and validating "
            "the tracking algorithms.</p>")
        quote = QLabel(SPEC_OBJECTIVE); quote.setWordWrap(True)
        quote.setStyleSheet(f"background:{tint(VIOLET, .10)}; border-left:4px solid {VIOLET}; border-radius:6px; padding:11px 14px; font-size:13px;")
        must = QLabel("The coarse alignment stage must:"); must.setStyleSheet("font-weight:700;")

        # ---- Performance specifications as coloured KPI chips (3 per row)
        perf = QWidget(); pg = QGridLayout(perf); pg.setContentsMargins(0, 0, 0, 0); pg.setHorizontalSpacing(12); pg.setVerticalSpacing(12)
        for i, (name, tgt) in enumerate(SPEC_PERF):
            pg.addWidget(card(name.split('. ', 1)[-1], None, CYCLE[i % len(CYCLE)], big=tgt, name='kpi'), i // 3, i % 3)
        for c in range(3): pg.setColumnStretch(c, 1)

        # ---- Deliverables as cards (3 per row)
        deliv = QWidget(); dg = QGridLayout(deliv); dg.setContentsMargins(0, 0, 0, 0); dg.setHorizontalSpacing(14); dg.setVerticalSpacing(14)
        for i, (name, desc) in enumerate(SPEC_DELIVERABLES):
            dg.addWidget(card(name, desc, CYCLE[i % len(CYCLE)]), i // 3, i % 3)
        for c in range(3): dg.setColumnStretch(c, 1)

        # ---- Evaluation: stacked weightage bar + one card per stage
        stage = QWidget(); sl = QHBoxLayout(stage); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(14)
        for n, v, colr, d in SPEC_EVAL: sl.addWidget(card(n, d, colr, big=f"{v}%", name='stage'), 1)

        # ---- Footer chips
        meta = QWidget(); ml = QHBoxLayout(meta); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(10)
        for i, (k, v) in enumerate(SPEC_META):
            chip = QLabel(f"<span style='color:{CYCLE[i]}; font-weight:800;'>{k}</span>&nbsp;&nbsp;{v}")
            chip.setTextFormat(Qt.TextFormat.RichText)
            chip.setStyleSheet(f"background:{tint(CYCLE[i], .10)}; border:1px solid {tint(CYCLE[i], .35)}; border-radius:14px; padding:6px 14px;")
            ml.addWidget(chip)
        ml.addStretch()

        w, l = page(hd,
                    row(cpanel('Mission brief', BLUE, brief), cpanel('Functional objective', VIOLET, quote, must, badge_list(SPEC_COARSE_STEPS, VIOLET))),
                    cpanel('Expected solution — the software shall be able to', GREEN, badge_list(SPEC_SOLUTION, GREEN, cols=2)),
                    row(cpanel('Camera parameters', BLUE, spec_table(SPEC_CAMERA, BLUE)), cpanel('Target parameters', TEAL, spec_table(SPEC_TARGET, TEAL))),
                    row(cpanel('Camera motion constraints', AMBER, spec_table(SPEC_MOTION, AMBER)), cpanel('Performance specifications', VIOLET, perf)),
                    cpanel('Disturbances & noise', RED, spec_table(SPEC_NOISE, RED)),
                    cpanel('Deliverables', GREEN, deliv),
                    cpanel('Evaluation method and criteria', AMBER, WeightBar(SPEC_EVAL), stage),
                    meta)
        return w

    # ---- main loop
    def tick(s):
        m, v = s.sim, s.video
        if m.running:
            s.acc += s.speed
            while s.acc >= 1: m.step(); s.acc -= 1
        # ---- Simulation-time gauge ----
        # Runs regardless of which tab is active (so it keeps counting even if the
        # user switches to another tab mid-simulation), unlike the per-tab repaint
        # logic below. 33ms per tick (s.timer's interval) is close enough for a UI
        # countdown; not meant to be a precise stopwatch.
        if s.sim_gauge_running and not s.sim_gauge_frozen_full:
            s.sim_elapsed = min(s.sim_elapsed + 0.033, s.sim_duration_total)
            frac = (s.sim_elapsed / s.sim_duration_total) if s.sim_duration_total > 0 else 1.0
            s.sim_gauge.setValue(int(frac * 100))
            s.sim_gauge_label.setText(s._fmt_time_left(s.sim_duration_total - s.sim_elapsed))
            if s.sim_elapsed >= s.sim_duration_total:
                s._sim_time_up()
        i = s.stack.currentIndex()
        if i not in (0, 2): return
        for w in (s.cam, s.sat, s.chart) if i == 0 else (s.vcam, s.vsat, s.vchart): w.update()
        if i == 0:
            if s.real_mode:
                # Live values come from backend.TrackingThread via on_stats_ready();
                # this just makes sure the panel repaints even between signal emissions.
                s.update_perf_labels()
            else:
                acq = m.acq_avg
                for lab, txt in zip(s.k, (f"{m.retention:.1f}%", f"{m.rmse:.1f} px", f"{acq:.2f} s" if acq is not None else '-', f"{m.loss_pct:.1f}%", f"{m.accuracy:.1f}%")): lab.setText(txt)
        else:
            s.tm.setText(f"{v.t:.0f}s")
            rows = v.rows
            s.dt.setRowCount(len(rows))
            for r, vals in enumerate(rows):
                for c, t in enumerate(vals): s.dt.setItem(r, c, QTableWidgetItem(t))
            s.update_video_rail()
        if not s.real_mode:
            if s._lk != m.lock:
                s._lk = m.lock; s.ls.setText('LOCKED' if m.lock else 'SEARCHING')
                s.ls.setStyleSheet(f"background:{T['ok'] if m.lock else T['bc']};color:{T['onac']};border-radius:10px;padding:10px;font-size:16px;font-weight:800")
            s.r['det'].setText('Detected' if m.det else 'Not found'); s.r['pan'].setText(f"{m.pan * .05:.2f}°"); s.r['tilt'].setText(f"{m.tilt * .05:.2f}°")
            s.r['off'].setText(f"{math.hypot(m.ox, m.oy):.1f} px"); s.r['time'].setText(f"{m.t:.1f} s")
            import time; n = time.perf_counter(); s.fps += (1 / max(n - s._last, 1e-3) - s.fps) * .05; s._last = n; s.r['fps'].setText(f"{s.fps:.0f} fps")
            s._refresh_events(m.events)

    # ============================================================
    # REAL TRACKING SIGNAL HANDLERS (Sim View, real_mode)
    # ============================================================

    def update_perf_labels(s):
        """Push the latest backend.TrackingThread metrics (in s.rt_stats) into the five
        'Tracking performance' labels on Sim View — the real-data replacement for the
        fake Sim.retention / Sim.rmse / Sim.acq_avg / Sim.loss_pct / Sim.accuracy values."""
        rs = s.rt_stats
        rmse = rs.get('rmse')
        acq = rs.get('acquisition_time')
        lk = rs.get('lock_retention') or 0.0
        losses = rs.get('target_losses', 0)
        acc = rs.get('accuracy') or 0.0
        vals = (
            f"{lk:.1f}%",
            f"{rmse:.1f} px" if rmse is not None else '-',
            f"{acq:.2f} s" if acq is not None else '-',
            str(losses),
            f"{acc:.1f}%",
        )
        for lab, txt in zip(s.k, vals): lab.setText(txt)

    def update_video_rail(s):
        """Fills the Video View side panel (s.vr labels) from the imported MP4's detector output."""
        v, i = s.video, s.video.info
        vals = dict.fromkeys(('state', 'det', 'off', 'pan', 'tilt', 'time', 'frame', 'mode', 'acq', 'rmse', 'lock', 'acc', 'loss', 'reacq'), '-')
        vals['src'] = 'No video imported'
        if v.active:
            vals['src'] = s.path.text() or '-'
            vals['state'] = 'Paused' if v.paused else 'Playing'
            if i:
                bo = i['beacon_offset']
                # Avg reacquisition time only means something once there's been more than
                # one target loss to average across; with 0 or 1 losses so far, mirror
                # Acquisition time instead of showing an empty/single-sample average.
                acq = i.get('acquisition_time')
                reacq = i.get('avg_reacquisition_time')
                if i['target_loss_count'] <= 1 or reacq is None:
                    reacq = acq
                vals.update(det='Detected' if v.det else 'Not found', off=f"{bo[0]:.1f}, {bo[1]:.1f}" if bo else '-',
                            pan=f"{i['pan']:.2f}°", tilt=f"{i['tilt']:.2f}°",
                            time=f"{v.t:.1f} s", frame=f"{i['frame_index']}/{i['total']}" if i['total'] else f"{i['frame_index']}",
                            mode=i['status'].title(), acq=f"{acq:.2f} s" if acq is not None else '-',
                            rmse=f"{i['rmse']:.1f} px", lock=f"{i['lock_retention']:.1f}%",
                            acc=f"{i['accuracy']:.1f}%", loss=str(i['target_loss_count']),
                            reacq=f"{reacq:.2f} s" if reacq is not None else '-')
        for k, t in vals.items(): s.vr[k].setText(t)
        s.vr['speed'].setText(f"{s.speed:g}x")

    def on_frame_ready(s, frame):
        if not s.sim_gauge_started and not s.sim_gauge_frozen_full:
            # "Gauge starts filling when the python side starts detection" — i.e. the
            # first real processed frame arrives, not merely when Start Simulation is
            # clicked (there's model warm-up / Unity-connect lag in between the two).
            s.sim_gauge_started = True
            s.sim_gauge_running = True
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            s.cam.width(), s.cam.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        s.cam.setPixmap(pixmap)

    def _refresh_events(s, items):
        """Render an events list (newest first, as stored) into s.evl newest-
        at-the-bottom, like a normal log. Skips the update entirely when the
        text hasn't changed (avoids resetting scroll position on every tick,
        since this is called every frame), and only snaps to the bottom on
        a real update if the view was already at the bottom — so scrolling
        up to read older events isn't interrupted by new ones arriving."""
        text = '\n'.join(reversed(items))
        if text == s._evl_cache:
            return
        s._evl_cache = text
        sb = s.evl.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        s.evl.setPlainText(text)
        if at_bottom:
            sb.setValue(sb.maximum())

    def on_status_changed(s, text):
        s.rt_events.insert(0, text)
        del s.rt_events[200:]
        s._refresh_events(s.rt_events)

        if 'connected' in text.lower():
            s.punity.setText('Connected')
        if 'tracking' in text.lower():
            s.ls.setText('LOCKED'); s.ls.setStyleSheet(f"background:{T['ok']};color:{T['onac']};border-radius:10px;padding:10px;font-size:16px;font-weight:800")
        elif 'acquiring' in text.lower():
            s.ls.setText('SEARCHING'); s.ls.setStyleSheet(f"background:{T['bc']};color:{T['onac']};border-radius:10px;padding:10px;font-size:16px;font-weight:800")

    def on_stats_ready(s, stats):
        if not s.sim_gauge_started and not s.sim_gauge_frozen_full:
            s.sim_gauge_started = True
            s.sim_gauge_running = True
        fps = stats.get('fps', 0.0)
        yolo = stats.get('yolo')
        kalman = stats.get('kalman')
        elapsed = stats.get('elapsed', 0.0)
        pan = stats.get("pan")
        tilt = stats.get("tilt")
        offs_x = stats.get("offset_x")
        offs_y = stats.get("offset_y")
        losses_pct = stats.get('target_loss_pct', 0.0)


        s.r['fps'].setText(f"{fps:.1f} fps")
        s.r['time'].setText(f"{elapsed:.1f} s")
        s.r['det'].setText('Detected' if yolo is not None else 'Not found')
        s.r["pan"].setText(f"{pan:7.2f}")
        s.r['tilt'].setText(f"{tilt:7.2f}")
        s.r['off'].setText(f"{offs_x}, {offs_y}")

        s.pdet.setText(f"({yolo[0]:.0f}, {yolo[1]:.0f})" if yolo is not None else 'No detection')
        s.pkal.setText(f"({kalman[0]:.0f}, {kalman[1]:.0f})" if kalman is not None else 'Not initialized')

        # ---- Tracking performance panel (RMSE, lock retention, acquisition time,
        # target loss, accuracy) — sourced straight from backend.py's stats dict,
        # which computes them from the real YOLO detections + Kalman filter output.
        s.rt_stats['rmse'] = stats.get('rmse')
        s.rt_stats['lock_retention'] = stats.get('lock_retention', s.rt_stats['lock_retention'])
        s.rt_stats['acquisition_time'] = stats.get('acquisition_time')
        # s.rt_stats['target_losses'] = stats.get('target_losses', s.rt_stats['target_losses'])
        s.rt_stats['target_losses'] = f"{stats.get('target_loss_pct', 0.0):.1f}%"
        s.rt_stats['accuracy'] = stats.get('accuracy', s.rt_stats['accuracy'])

        # Per-frame Euclidean error (hypot(ox, oy)) feeds the chart's rolling window.
        # Only present once Kalman tracking has started (not during initial acquisition).
        err = stats.get('error')
        if err is not None:
            s.rt_err_history.append(err)

        s.update_perf_labels()

        if s.bench_running:
            s._bench_on_stats(stats)

    # ============================================================
    # UNITY LAUNCH + EMBED (from App.py, unchanged win32 logic)
    # ============================================================

    def start_unity(s):
        if s.unity_proc is not None:
            return
        s.start_unity_btn.setEnabled(False)
        s.start_unity_btn.setText('3d Environment Loading....')
        s.punity.setText('Starting...')

        try:
            s.unity_proc = subprocess.Popen(
                [UNITY_EXE, "-popupwindow", "-screen-fullscreen", "0"]
            )
        except OSError as e:
            # e.g. FileNotFoundError when UNITY_EXE doesn't point to a real file —
            # fail loudly in the UI instead of raising out of a QTimer callback,
            # which would otherwise leave the button stuck disabled/mid-text.
            s.unity_proc = None
            s.start_unity_btn.setEnabled(True)
            s.start_unity_btn.setText('Start 3D Environment')
            s.punity.setText('Not found')
            s.on_status_changed(f"Could not launch Unity: {e}")
            QMessageBox.warning(s, 'Start 3D Environment',
                                 f"Could not launch Unity at:\n{UNITY_EXE}\n\n{e}")
            return

        QTimer.singleShot(800, s.find_and_embed_unity)

    def find_unity_hwnd_by_pid(s, pid):
        found = []

        def enum_handler(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid:
                rect = win32gui.GetWindowRect(hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 50 and h > 50:
                    found.append(hwnd)

        win32gui.EnumWindows(enum_handler, None)
        return found[0] if found else 0

    def find_and_embed_unity(s):
        hwnd = s.find_unity_hwnd_by_pid(s.unity_proc.pid)

        if hwnd == 0:
            s._embed_attempts += 1
            if s._embed_attempts > 40:
                s.start_unity_btn.setText('Could not find Unity window')
                s.punity.setText('Not found')
                return
            QTimer.singleShot(500, s.find_and_embed_unity)
            return

        s.unity_hwnd = hwnd

        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        style &= ~(win32con.WS_POPUP | win32con.WS_CAPTION | win32con.WS_THICKFRAME |
                   win32con.WS_SYSMENU | win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX)
        style |= win32con.WS_CHILD | win32con.WS_VISIBLE
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)

        # Unity always starts life embedded in whichever tab is currently on screen
        # (normally Sim View, since that's where the Start Unity button lives).
        s._unity_parent = s.unity_container if s.stack.currentIndex() == 0 else s.vsat
        win32gui.SetParent(hwnd, int(s._unity_parent.winId()))

        s.resize_unity_window()
        QTimer.singleShot(300, s.nudge_unity_render_loop)

        s.start_unity_btn.setText('Running....')
        s.punity.setText('Running (waiting for connection)')

    def _unity_container_size(s):
        """Native client-area size of the container, in the same physical-pixel
        units Win32 resize calls expect. Using Qt's width()/height() directly
        can under-size the embedded window on displays with scaling enabled
        (125%/150%/etc.), since Qt reports logical pixels while SetWindowPos/
        MoveWindow work in physical pixels — that mismatch is what causes the
        Unity window to sit small in the top-left corner instead of filling
        the panel."""
        parent = s._unity_parent or s.unity_container
        left, top, right, bottom = win32gui.GetClientRect(int(parent.winId()))
        return right - left, bottom - top

    def resize_unity_window(s):
        if not s.unity_hwnd:
            return
        w, h = s._unity_container_size()
        if w <= 0 or h <= 0:
            return
        win32gui.SetWindowPos(
            s.unity_hwnd, 0, 0, 0, w, h,
            win32con.SWP_FRAMECHANGED | win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
        )
        win32gui.MoveWindow(s.unity_hwnd, 0, 0, w, h, True)

    def nudge_unity_render_loop(s):
        if not s.unity_hwnd:
            return
        w, h = s._unity_container_size()
        if w <= 0 or h <= 0:
            return
        win32gui.MoveWindow(s.unity_hwnd, 0, 0, max(w - 1, 1), h, True)
        win32gui.MoveWindow(s.unity_hwnd, 0, 0, w, h, True)
        win32gui.SendMessage(s.unity_hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        win32gui.SendMessage(s.unity_hwnd, win32con.WM_SETFOCUS, 0, 0)
        try:
            win32gui.SetFocus(s.unity_hwnd)
        except Exception:
            pass

    def resizeEvent(s, event):
        super().resizeEvent(event)
        s.resize_unity_window()
        if hasattr(s, 'busy_overlay') and s.busy_overlay.isVisible():
            s.busy_overlay.setGeometry(s.centralWidget().rect())

    def _set_bench_busy(s, busy):
        """Shown while a benchmark run is in progress (see run_bench()/_bench_finish_all()):
        a subtle dark overlay over the whole window that blocks tab switches and clicks
        on anything else until the run finishes."""
        if busy:
            s.busy_overlay.setGeometry(s.centralWidget().rect())
            s.busy_overlay.show()
            s.busy_overlay.raise_()
        else:
            s.busy_overlay.hide()

    def closeEvent(s, event):
        s.stop_video()
        s.tracking_thread.stop()
        s.tracking_thread.wait(2000)
        if s.unity_proc:
            s.unity_proc.terminate()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv); app.setStyle('Fusion'); fnt = QFont(); fnt.setFamilies(['Inter', 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', 'Arial']); fnt.setPointSize(10); app.setFont(fnt); win = Main(); win.resize(1400, 880); win.showMaximized(); sys.exit(app.exec())