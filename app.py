"""
app.py — Real-Time Vehicle Speed Detection & Monitoring System
India Roads | YOLOv8 + Built-in ByteTrack | Single File
HuggingFace Spaces Ready

"""

# ── stdlib ──────────────────────────────────────────────────────────────
import os, time, tempfile, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── third-party ──────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Vehicle Speed Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — dark professional dashboard
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');

html, body, .main, [data-testid="stAppViewContainer"] {
    background: #07090f !important;
}
.block-container { padding: 1rem 1.5rem 2rem !important; max-width: 100% !important; }

section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d40 !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* KPI card */
.kpi-card {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 14px 10px 10px;
    text-align: center;
    transition: border-color .2s;
}
.kpi-card:hover { border-color: #1f6feb; }
.kpi-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
    color: #e6edf3;
}
.kpi-lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: .14em;
    color: #484f58;
    margin-top: 4px;
    text-transform: uppercase;
}

/* Zone badge */
.z-green  { color: #3fb950 !important; }
.z-yellow { color: #d29922 !important; }
.z-red    { color: #f85149 !important; }

/* Section header */
.sec-hdr {
    font-family: 'Share Tech Mono', monospace;
    font-size: .6rem;
    letter-spacing: .18em;
    color: #484f58;
    border-bottom: 1px solid #1e2d40;
    padding-bottom: 5px;
    margin: 18px 0 10px;
    text-transform: uppercase;
}

/* Live pill */
.pill-live {
    display:inline-block;
    background: rgba(63,185,80,.12);
    color: #3fb950;
    border: 1px solid #3fb950;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem;
    letter-spacing: .1em;
}
.pill-idle {
    display:inline-block;
    background: rgba(72,79,88,.12);
    color: #484f58;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem;
}

/* Vehicle count chip */
.vc-chip {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 6px;
    padding: 8px 6px;
    text-align: center;
    margin: 3px 0;
}
.vc-chip .num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    color: #e6edf3;
}
.vc-chip .lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: .58rem;
    letter-spacing: .12em;
    color: #484f58;
}

/* Sidebar button */
.stButton > button {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    width: 100%;
    transition: all .2s;
}
.stButton > button:hover {
    border-color: #1f6feb !important;
    background: #161b22 !important;
}

/* Zone threshold display */
.zone-bar {
    display: flex;
    gap: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .72rem;
    margin: 6px 0 2px;
    padding: 6px 10px;
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 6px;
}

/* Idle placeholder */
.idle-box {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    height: 420px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# COCO class ids → vehicle label
VEHICLE_CLASSES: Dict[int, str] = {
    1: "Bicycle",
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck",
}

# Class display color (RGB)
CLASS_COLORS: Dict[int, Tuple] = {
    1: (100, 210, 255),   # Bicycle  — sky blue
    2: (72,  201, 112),   # Car      — green
    3: (255, 165, 60),    # Bike     — orange
    5: (200, 100, 255),   # Bus      — purple
    7: (255,  85,  85),   # Truck    — red
}

ZONE_RGB: Dict[str, Tuple] = {
    "green":  (63,  185,  80),
    "yellow": (210, 153,  34),
    "red":    (248,  81,  73),
}

ZONE_HEX = {"green": "#3fb950", "yellow": "#d29922", "red": "#f85149"}

MAX_SPEED_KMH = 200.0
EMA_ALPHA     = 0.18   # stronger smoothing — was 0.35 (too reactive)
MIN_PIX_MOVE  = 2.0


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedVehicle:
    track_id:   int
    class_id:   int
    class_name: str
    color_rgb:  Tuple

    # (cx, cy, timestamp_sec)
    positions: deque = field(default_factory=lambda: deque(maxlen=60))

    current_bbox:   Optional[np.ndarray] = None
    current_speed:  float = 0.0
    ema_speed:      float = 0.0
    zone:           str   = "green"
    confidence:     float = 0.0

    crossed_line:   bool  = False
    is_active:      bool  = True
    frames_lost:    int   = 0
    first_frame:    int   = 0
    last_frame:     int   = 0

    @property
    def center(self):
        if self.positions:
            return self.positions[-1][0], self.positions[-1][1]
        return None


# ════════════════════════════════════════════════════════════════════════════
# SPEED ESTIMATION (inline, no supervision)
# ════════════════════════════════════════════════════════════════════════════

def pixels_per_meter(p1: Tuple, p2: Tuple, dist_m: float) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return max(np.sqrt(dx*dx + dy*dy), 1.0) / max(dist_m, 0.1)

def compute_speed(
    vehicle: TrackedVehicle,
    ppm: float,
    ema_state: Dict,
) -> Tuple[float, str]:
    """
    Speed using a multi-frame window (up to 5 frames apart) for stability.
    Frame-to-frame delta (1 frame) is extremely noisy at high FPS.
    Using a 5-frame window averages out detection jitter.
    """
    pos = vehicle.positions
    if len(pos) < 3:
        return 0.0, "green"

    # Use up to 5-frame lookback — larger window = more stable speed
    lookback = min(5, len(pos) - 1)
    px1, py1, t1 = pos[-1 - lookback]
    px2, py2, t2 = pos[-1]

    dt = t2 - t1
    if dt < 1e-4:
        return ema_state.get(vehicle.track_id, 0.0), vehicle.zone

    pix_dist = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
    if pix_dist < MIN_PIX_MOVE * lookback:
        raw = 0.0
    else:
        raw = min((pix_dist / ppm / dt) * 3.6, MAX_SPEED_KMH)

    # EMA smoothing
    prev = ema_state.get(vehicle.track_id, raw)
    smoothed = EMA_ALPHA * raw + (1 - EMA_ALPHA) * prev
    ema_state[vehicle.track_id] = smoothed
    return round(smoothed, 1), ""

def classify_zone(speed: float, g_max: float, y_max: float, r_max: float) -> str:
    if speed <= g_max:   return "green"
    if speed <= y_max:   return "yellow"
    return "red"


# ════════════════════════════════════════════════════════════════════════════
# ANNOTATION
# ════════════════════════════════════════════════════════════════════════════

def _bgr(rgb): return (rgb[2], rgb[1], rgb[0])

def annotate_frame(
    frame: np.ndarray,
    vehicles: List[TrackedVehicle],
    line_y: Optional[int],
    show_trail: bool,
    calib_p1, calib_p2, calib_dist, show_calib,
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    h_frame, w_frame = out.shape[:2]

    for v in vehicles:
        if v.current_bbox is None:
            continue

        x1, y1, x2, y2 = v.current_bbox.astype(int)
        zone_bgr = _bgr(ZONE_RGB[v.zone])

        # Motion trail
        if show_trail and len(v.positions) >= 2:
            pts = list(v.positions)[-25:]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                col = tuple(int(c * alpha * 0.8) for c in zone_bgr)
                cv2.line(out,
                         (int(pts[i-1][0]), int(pts[i-1][1])),
                         (int(pts[i][0]),   int(pts[i][1])),
                         col, 2)

        # Bounding box — 2px solid zone color
        cv2.rectangle(out, (x1, y1), (x2, y2), zone_bgr, 2)

        # Label pill
        l1 = f"ID:{v.track_id} {v.class_name}"
        l2 = f"{v.current_speed:.1f} km/h"
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        (w1, h1), _ = cv2.getTextSize(l1, font, fs, th)
        (w2, h2), _ = cv2.getTextSize(l2, font, fs, th)
        bw = max(w1, w2) + 10
        bh = h1 + h2 + 14
        ly = max(y1 - bh - 2, 0)
        cv2.rectangle(out, (x1, ly), (x1+bw, ly+bh), zone_bgr, -1)
        cv2.putText(out, l1, (x1+4, ly+h1+3),     font, fs, (255,255,255), th, cv2.LINE_AA)
        cv2.putText(out, l2, (x1+4, ly+h1+h2+10), font, fs, (255,255,255), th, cv2.LINE_AA)

        # Red zone alert flash border (thicker ring for violations)
        if v.zone == "red":
            cv2.rectangle(out, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 1)

    # Counting line
    if line_y is not None:
        cv2.line(out, (0, line_y), (w_frame, line_y), (0, 220, 220), 2)
        cv2.putText(out, "COUNT LINE", (8, line_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 220), 1, cv2.LINE_AA)

    # Calibration line
    if show_calib and calib_p1 and calib_p2:
        cv2.line(out, calib_p1, calib_p2, (0, 255, 200), 2)
        cv2.circle(out, calib_p1, 5, (0, 255, 200), -1)
        cv2.circle(out, calib_p2, 5, (0, 255, 200), -1)
        mid = ((calib_p1[0]+calib_p2[0])//2, (calib_p1[1]+calib_p2[1])//2)
        cv2.putText(out, f"{calib_dist:.0f}m", (mid[0]+5, mid[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    # ── Vehicle count panel — top-right corner ─────────────────────────────
    # Count active vehicles by class
    cls_counts: Dict[str, int] = {}
    for v in vehicles:
        cls_counts[v.class_name] = cls_counts.get(v.class_name, 0) + 1

    panel_lines = [f"Total: {len(vehicles)}"] + [f"{k}: {n}" for k, n in cls_counts.items()]
    font_p, fs_p, th_p = cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1
    line_h = 20
    panel_w = 130
    panel_h = len(panel_lines) * line_h + 10
    px = w_frame - panel_w - 8
    py = 8

    # Semi-transparent dark background
    overlay_p = out.copy()
    cv2.rectangle(overlay_p, (px - 6, py - 4), (px + panel_w, py + panel_h), (10, 14, 26), -1)
    cv2.addWeighted(overlay_p, 0.72, out, 0.28, 0, out)

    for i, line in enumerate(panel_lines):
        color = (56, 189, 248) if i == 0 else (200, 200, 200)
        cv2.putText(out, line,
                    (px, py + (i + 1) * line_h - 4),
                    font_p, fs_p, color, th_p, cv2.LINE_AA)

    return out


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

def _init():
    defs = dict(
        processing=False,
        model=None,
        tracks={},            # track_id → TrackedVehicle
        ema_state={},         # track_id → float (ema speed)
        violation_log=[],
        total_unique=0,
        class_counts=defaultdict(int),
        crossed_counts=defaultdict(int),
        fastest={"speed": 0.0, "class": "—", "id": -1},
        slowest={"speed": 9999.0, "class": "—", "id": -1},
        fps_actual=0.0,
        _fps_frames=0,
        _fps_t=time.time(),
        output_path=None,
        frame_idx=0,
        # Video bytes cached here — uploaded_file becomes None after st.rerun()
        video_bytes=None,
        video_name=None,
        # Violation alert state
        violation_alert=False,
        alert_vehicle={"class": "—", "speed": 0.0, "id": -1},
        alert_shown_ids=set(),    # track_ids already alerted this session
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ════════════════════════════════════════════════════════════════════════════
# BYTETRACK YAML — write once to temp
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(name: str):
    m = YOLO(name)
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    m.predict(dummy, verbose=False, conf=0.9)
    return m


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='padding:10px 0 6px;'>"
        "<span style='font-family:Share Tech Mono,monospace;font-size:1rem;"
        "color:#e6edf3;font-weight:700;'>SPEED MONITOR</span><br>"
        "<span style='font-family:Share Tech Mono,monospace;font-size:.6rem;"
        "color:#484f58;letter-spacing:.15em;'>INDIA ROADS — YOLOv8</span>"
        "</div>", unsafe_allow_html=True)
    st.divider()

    # ── Input source ──────────────────────────────────
    st.markdown("<div class='sec-hdr'>INPUT SOURCE</div>", unsafe_allow_html=True)
    src = st.selectbox("Source", ["Video File", "Webcam", "IP / RTSP Camera"],
                       label_visibility="collapsed")

    uploaded_file = webcam_idx = rtsp_url = None
    if src == "Video File":
        uploaded_file = st.file_uploader("Upload", type=["mp4","avi","mov","mkv"],
                                          label_visibility="collapsed")
    elif src == "Webcam":
        webcam_idx = st.number_input("Camera index", 0, 5, 0)
    else:
        rtsp_url = st.text_input("Stream URL",
            placeholder="rtsp://192.168.x.x:554/stream")
        st.caption("DroidCam: http://192.168.x.x:4747/video")

    # ── Model ─────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>DETECTION MODEL</div>", unsafe_allow_html=True)
    MODEL_OPTIONS = ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"]
    model_choice = st.selectbox("Model", MODEL_OPTIONS, index=1,
                                label_visibility="collapsed",
                                help="n=fastest  x=most accurate")
    c1, c2 = st.columns(2)
    conf_thr = c1.slider("Conf", 0.10, 0.90, 0.35, 0.05)
    iou_thr  = c2.slider("IOU",  0.10, 0.90, 0.45, 0.05)

    # ── Speed zones ───────────────────────────────────
    st.markdown("<div class='sec-hdr'>SPEED ZONES (km/h)</div>", unsafe_allow_html=True)

    green_max  = st.slider("Green max — safe",    5,  120,  40, 5,
                            help="Speed up to this value = GREEN (safe)")
    yellow_max = st.slider("Yellow max — warning", green_max+5, 180, 70, 5,
                            help="Speed up to this value = YELLOW (warning)")
    red_max    = st.slider("Red max — over-speed", yellow_max+5, 250, 120, 5,
                            help="Speed up to this value = RED (violation). Above = also RED.")

    st.markdown(
        f"<div class='zone-bar'>"
        f"<span class='z-green'>GREEN &le;{green_max}</span>"
        f"<span style='color:#1e2d40;'>|</span>"
        f"<span class='z-yellow'>YELLOW &le;{yellow_max}</span>"
        f"<span style='color:#1e2d40;'>|</span>"
        f"<span class='z-red'>RED &le;{red_max}</span>"
        f"<span style='color:#1e2d40;'>|</span>"
        f"<span class='z-red' style='opacity:.6;'>EXTREME &gt;{red_max}</span>"
        f"</div>",
        unsafe_allow_html=True)

    # ── Calibration ───────────────────────────────────
    st.markdown("<div class='sec-hdr'>SPEED CALIBRATION</div>", unsafe_allow_html=True)
    ref_dist = st.number_input("Reference distance (m)", 1.0, 500.0, 20.0, 1.0,
        help="Real-world distance between P1 and P2 on road")
    st.caption("Lane=3.5m | Car=4.5m | Bus=12m | Divider markings")
    cc1, cc2 = st.columns(2)
    p1x = cc1.number_input("P1 X", 0, 9999, 100)
    p1y = cc2.number_input("P1 Y", 0, 9999, 400)
    p2x = cc1.number_input("P2 X", 0, 9999, 400)
    p2y = cc2.number_input("P2 Y", 0, 9999, 400)
    _p1, _p2 = (p1x, p1y), (p2x, p2y)
    ppm = pixels_per_meter(_p1, _p2, ref_dist)
    st.caption(f"Calibrated: **{ppm:.2f} px/m**")

    # ── Counting line ─────────────────────────────────
    st.markdown("<div class='sec-hdr'>COUNTING LINE</div>", unsafe_allow_html=True)
    show_line   = st.toggle("Show counting line", True)
    line_pct    = st.slider("Position (% of height)", 10, 90, 60) if show_line else 60

    # ── Display ───────────────────────────────────────
    st.markdown("<div class='sec-hdr'>DISPLAY OPTIONS</div>", unsafe_allow_html=True)
    show_trail  = st.toggle("Motion trails", True)
    show_calib  = st.toggle("Show calibration line", False)
    save_video  = st.toggle("Save output video", False)

    # ── Control ───────────────────────────────────────
    st.markdown("<div class='sec-hdr'>CONTROL</div>", unsafe_allow_html=True)

    can_start = (
        (src == "Video File"        and uploaded_file is not None) or
        (src == "Webcam")                                           or
        (src == "IP / RTSP Camera"  and rtsp_url and rtsp_url.strip())
    )

    if not st.session_state.processing:
        if st.button("START PROCESSING", use_container_width=True):
            if not can_start:
                st.error("Provide a valid input source first.")
            else:
                # Reset all state
                st.session_state.tracks        = {}
                st.session_state.ema_state     = {}
                st.session_state.violation_log = []
                st.session_state.total_unique  = 0
                st.session_state.class_counts  = defaultdict(int)
                st.session_state.crossed_counts= defaultdict(int)
                st.session_state.fastest       = {"speed":0.0,"class":"—","id":-1}
                st.session_state.slowest       = {"speed":9999.0,"class":"—","id":-1}
                st.session_state.violation_alert = False
                st.session_state.alert_vehicle   = {"class":"—","speed":0.0,"id":-1}
                st.session_state.alert_shown_ids = set()
                st.session_state.fps_actual    = 0.0
                st.session_state._fps_frames   = 0
                st.session_state._fps_t        = time.time()
                st.session_state.frame_idx     = 0
                st.session_state.output_path   = None
                # Cache video bytes BEFORE rerun — uploader resets after rerun
                if src == "Video File" and uploaded_file is not None:
                    st.session_state.video_bytes = uploaded_file.read()
                    st.session_state.video_name  = uploaded_file.name
                else:
                    st.session_state.video_bytes = None
                    st.session_state.video_name  = None
                # Load model
                with st.spinner(f"Loading {model_choice} …"):
                    st.session_state.model = load_model(model_choice)
                st.session_state.processing = True
                st.rerun()
    else:
        if st.button("STOP", use_container_width=True):
            st.session_state.processing = False
            st.rerun()

    if st.session_state.processing:
        st.markdown("<div class='pill-live'>LIVE</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='pill-idle'>IDLE</div>", unsafe_allow_html=True)

    st.divider()

    # ── Export ────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>EXPORT</div>", unsafe_allow_html=True)
    if st.session_state.violation_log:
        df_exp = pd.DataFrame(st.session_state.violation_log)
        st.download_button(
            "Download Violation Report (CSV)",
            df_exp.to_csv(index=False),
            f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Violation CSV appears here after detection.")

    if st.session_state.output_path and os.path.exists(st.session_state.output_path):
        with open(st.session_state.output_path, "rb") as f:
            st.download_button(
                "Download Processed Video",
                f, "speed_output.mp4", "video/mp4",
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Header
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-family:Share Tech Mono,monospace;font-size:1.3rem;"
    "color:#e6edf3;margin:0 0 2px;'>"
    "Real-Time Vehicle Speed Detection</h1>"
    "<p style='font-family:Share Tech Mono,monospace;font-size:.65rem;"
    "color:#484f58;letter-spacing:.12em;margin:0 0 14px;'>"
    "INDIA ROADS &nbsp;|&nbsp; YOLOv8 + BYTETRACK &nbsp;|&nbsp; LIVE SPEED ZONES</p>",
    unsafe_allow_html=True)

# ── KPI strip ──────────────────────────────────────────────────────────────
def kpi(col, lbl, val, color="#e6edf3"):
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-val' style='color:{color};'>{val}</div>"
        f"<div class='kpi-lbl'>{lbl}</div>"
        f"</div>", unsafe_allow_html=True)

kc = st.columns(6)
_s = st.session_state
active_vehicles_list = [v for v in _s.tracks.values() if v.is_active]
# Use median of stable speeds (vehicles with history >= 8) to avoid spike display
stable_speeds = [
    v.current_speed for v in active_vehicles_list
    if v.current_speed > 2.0 and len(v.positions) >= 8
]
max_now   = max(stable_speeds, default=0.0)
max_color = "#f85149" if max_now > yellow_max else "#d29922" if max_now > green_max else "#3fb950"

fastest_spd = _s.fastest["speed"] if _s.fastest["id"] != -1 else 0.0
slowest_spd = _s.slowest["speed"] if _s.slowest["id"] != -1 and _s.slowest["speed"] < 9999.0 else 0.0

kpi(kc[0], "TOTAL DETECTED",   str(_s.total_unique))
kpi(kc[1], "ACTIVE IN FRAME",  str(len(active_vehicles_list)), "#388bfd")
kpi(kc[2], "MAX SPEED NOW",    f"{max_now:.0f} km/h", max_color)
kpi(kc[3], "SESSION FASTEST",  f"{fastest_spd:.0f} km/h", "#f85149")
kpi(kc[4], "SESSION SLOWEST",  f"{slowest_spd:.0f} km/h", "#3fb950")
kpi(kc[5], "PROCESSING FPS",   f"{_s.fps_actual:.1f}",      "#388bfd")

st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

# ── Violation alert banner ─────────────────────────────────────────────────
alert_ph = st.empty()

def render_alert():
    if _s.violation_alert:
        av = _s.alert_vehicle
        alert_ph.markdown(
            f"<div style='background:rgba(239,68,68,0.12);border:1px solid #ef4444;"
            f"border-radius:8px;padding:10px 18px;display:flex;align-items:center;"
            f"gap:14px;font-family:Share Tech Mono,monospace;'>"
            f"<span style='font-size:1.2rem;'>&#9888;</span>"
            f"<span style='color:#f85149;font-weight:700;font-size:.85rem;letter-spacing:.08em;'>"
            f"SPEED VIOLATION DETECTED</span>"
            f"<span style='color:#e6edf3;font-size:.8rem;'>"
            f"&nbsp;|&nbsp; {av['class']} &nbsp;|&nbsp; "
            f"<span style='color:#f85149;font-weight:700;'>{av['speed']:.0f} km/h</span>"
            f" &nbsp;|&nbsp; ID: {av['id']}</span>"
            f"</div>",
            unsafe_allow_html=True)
    else:
        alert_ph.empty()

render_alert()

# ── Video + stats layout ────────────────────────────────────────────────────
vid_col, stat_col = st.columns([3, 1])

with vid_col:
    video_ph   = st.empty()
    prog_ph    = st.empty()

with stat_col:
    st.markdown("<div class='sec-hdr'>VEHICLE COUNTS</div>", unsafe_allow_html=True)
    count_phs = {name: st.empty() for name in list(VEHICLE_CLASSES.values()) + ["Crossed"]}

    def render_counts():
        for cls, ph in count_phs.items():
            if cls == "Crossed":
                n = sum(_s.crossed_counts.values())
                color = "#d29922"
            else:
                n = _s.class_counts.get(cls, 0)
                color = "#e6edf3"
            ph.markdown(
                f"<div class='vc-chip'>"
                f"<div class='num' style='color:{color};'>{n}</div>"
                f"<div class='lbl'>{cls.upper()}</div>"
                f"</div>", unsafe_allow_html=True)

    render_counts()

    st.markdown("<div class='sec-hdr'>SPEED RECORDS</div>", unsafe_allow_html=True)
    fastest_ph = st.empty()
    slowest_ph = st.empty()
    violations_count_ph = st.empty()

    def render_fastest():
        f = _s.fastest
        s = _s.slowest
        spd_f = f["speed"] if f["id"] != -1 else 0.0
        spd_s = s["speed"] if s["id"] != -1 and s["speed"] < 9999.0 else 0.0
        fastest_ph.markdown(
            f"<div class='vc-chip'>"
            f"<div class='num z-red'>{spd_f:.0f} <span style='font-size:.9rem;'>km/h</span></div>"
            f"<div class='lbl'>FASTEST — {f['class']}</div>"
            f"</div>", unsafe_allow_html=True)
        slowest_ph.markdown(
            f"<div class='vc-chip'>"
            f"<div class='num z-green'>{spd_s:.0f} <span style='font-size:.9rem;'>km/h</span></div>"
            f"<div class='lbl'>SLOWEST — {s['class']}</div>"
            f"</div>", unsafe_allow_html=True)
        violations_count_ph.markdown(
            f"<div class='vc-chip'>"
            f"<div class='num z-red'>{len(_s.violation_log)}</div>"
            f"<div class='lbl'>VIOLATIONS</div>"
            f"</div>", unsafe_allow_html=True)

    render_fastest()

# ── Speed Violation Dashboard — bottom panel ───────────────────────────────
st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>"
    "<div style='width:8px;height:8px;border-radius:50%;background:#f85149;"
    "box-shadow:0 0 8px #f85149;animation:pulse 1.2s infinite;'></div>"
    "<span style='font-family:Share Tech Mono,monospace;font-size:.62rem;"
    "letter-spacing:.18em;color:#484f58;text-transform:uppercase;'>"
    "HIGH SPEED VIOLATION LOG — VEHICLES THAT CROSSED RED ZONE</span>"
    "</div>"
    "<style>@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.3;}}</style>",
    unsafe_allow_html=True)

violations_ph = st.empty()

def render_violations():
    logs = _s.violation_log
    if not logs:
        violations_ph.markdown(
            "<div style='background:#0d1117;border:1px dashed #1e2d40;"
            "border-radius:8px;padding:20px;text-align:center;"
            "font-family:Share Tech Mono,monospace;font-size:.75rem;"
            "color:#484f58;letter-spacing:.1em;'>"
            "NO VIOLATIONS RECORDED &nbsp;·&nbsp; ALL VEHICLES IN SAFE ZONE"
            "</div>",
            unsafe_allow_html=True)
        return

    # Build styled rows — most recent first, show last 100
    recent = list(reversed(logs[-100:]))

    rows_html = ""
    for i, e in enumerate(recent):
        spd   = e["speed_kmh"]
        cls   = e["class"]
        tid   = e["track_id"]
        ts    = e["time"]
        conf  = e.get("conf", 0.0)
        # Speed badge color: deeper red for extreme speeds
        if spd > 100:
            spd_color = "#ff3b3b"
            badge_bg  = "rgba(255,59,59,0.18)"
        else:
            spd_color = "#f85149"
            badge_bg  = "rgba(248,81,73,0.12)"

        row_bg = "rgba(248,81,73,0.04)" if i % 2 == 0 else "transparent"

        rows_html += (
            f"<tr style='background:{row_bg};border-bottom:1px solid #1a2332;'>"
            f"<td style='padding:8px 12px;color:#484f58;font-size:.72rem;white-space:nowrap;'>{ts}</td>"
            f"<td style='padding:8px 12px;'>"
            f"<span style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:4px;padding:2px 8px;color:#c9d1d9;"
            f"font-size:.72rem;'>ID {tid}</span>"
            f"</td>"
            f"<td style='padding:8px 12px;color:#c9d1d9;font-size:.75rem;'>{cls}</td>"
            f"<td style='padding:8px 14px;text-align:center;'>"
            f"<span style='background:{badge_bg};border:1px solid {spd_color};"
            f"border-radius:6px;padding:3px 10px;"
            f"color:{spd_color};font-weight:700;font-size:.82rem;letter-spacing:.04em;'>"
            f"{spd:.1f} km/h</span>"
            f"</td>"
            f"<td style='padding:8px 12px;text-align:center;'>"
            f"<span style='background:rgba(248,81,73,0.15);border:1px solid #f85149;"
            f"border-radius:20px;padding:2px 10px;"
            f"color:#f85149;font-size:.65rem;letter-spacing:.1em;'>"
            f"RED ZONE</span>"
            f"</td>"
            f"<td style='padding:8px 12px;color:#484f58;font-size:.7rem;'>{conf:.0%}</td>"
            f"</tr>"
        )

    table_html = (
        f"<div style='background:#0d1117;border:1px solid #2a1f2d;"
        f"border-radius:10px;overflow:hidden;'>"
        f"<table style='width:100%;border-collapse:collapse;"
        f"font-family:Share Tech Mono,monospace;'>"
        f"<thead>"
        f"<tr style='background:#130d1a;border-bottom:1px solid #2a1f2d;'>"
        f"<th style='padding:9px 12px;text-align:left;font-size:.6rem;"
        f"letter-spacing:.14em;color:#6e40c9;font-weight:600;'>TIME</th>"
        f"<th style='padding:9px 12px;text-align:left;font-size:.6rem;"
        f"letter-spacing:.14em;color:#6e40c9;font-weight:600;'>TRACK ID</th>"
        f"<th style='padding:9px 12px;text-align:left;font-size:.6rem;"
        f"letter-spacing:.14em;color:#6e40c9;font-weight:600;'>VEHICLE</th>"
        f"<th style='padding:9px 14px;text-align:center;font-size:.6rem;"
        f"letter-spacing:.14em;color:#6e40c9;font-weight:600;'>SPEED</th>"
        f"<th style='padding:9px 12px;text-align:center;font-size:.6rem;"
        f"letter-spacing:.14em;color:#6e40c9;font-weight:600;'>ZONE</th>"
        f"<th style='padding:9px 12px;text-align:left;font-size:.6rem;"
        f"letter-spacing:.14em;color:#6e40c9;font-weight:600;'>CONF</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
        f"<div style='padding:6px 12px;border-top:1px solid #1e2d40;"
        f"font-size:.62rem;color:#484f58;font-family:Share Tech Mono,monospace;"
        f"display:flex;justify-content:space-between;'>"
        f"<span>TOTAL VIOLATIONS: {len(logs)}</span>"
        f"<span>SHOWING LATEST {min(len(logs),100)}</span>"
        f"</div>"
        f"</div>"
    )
    violations_ph.markdown(table_html, unsafe_allow_html=True)

render_violations()


# ════════════════════════════════════════════════════════════════════════════
# TRACK MANAGEMENT (no supervision)
# ════════════════════════════════════════════════════════════════════════════

TRACK_BUFFER  = 25   # frames to keep a lost track alive
IOU_THRESHOLD = 0.25 # min IoU to match detection to existing track
_next_id      = [1]  # mutable counter in a list so inner functions can mutate it


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def update_tracks(
    raw_boxes:   Optional[np.ndarray],   # [N,4] xyxy
    raw_classes: Optional[np.ndarray],   # [N]
    raw_confs:   Optional[np.ndarray],   # [N]
    timestamp:   float,
    frame_idx:   int,
):
    """
    Pure IoU multi-object tracker — no ultralytics tracker, no lap, no yaml.

    Algorithm (greedy Hungarian-free):
      1. Collect active track bboxes.
      2. Build NxM IoU matrix (tracks × detections).
      3. Greedily match highest-IoU pairs above threshold.
      4. Update matched tracks, create new tracks for unmatched dets,
         increment frames_lost for unmatched tracks.
    """
    tracks = _s.tracks

    # Filter detections to vehicle classes only
    dets: List[Tuple] = []   # (bbox[4], class_id, conf)
    if raw_boxes is not None and len(raw_boxes):
        for i in range(len(raw_boxes)):
            cid = int(raw_classes[i])
            if cid in VEHICLE_CLASSES:
                dets.append((raw_boxes[i].astype(int), cid, float(raw_confs[i])))

    active_tracks = [(tid, v) for tid, v in tracks.items() if v.is_active]

    # Build IoU matrix  [len(active_tracks) x len(dets)]
    matched_tracks = set()
    matched_dets   = set()

    if active_tracks and dets:
        iou_mat = np.zeros((len(active_tracks), len(dets)), dtype=np.float32)
        for ti, (tid, v) in enumerate(active_tracks):
            if v.current_bbox is None:
                continue
            for di, (bbox, _, _) in enumerate(dets):
                iou_mat[ti, di] = _iou(v.current_bbox, bbox)

        # Greedy matching: repeatedly pick global max
        while True:
            ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[ti, di] < IOU_THRESHOLD:
                break
            matched_tracks.add(ti)
            matched_dets.add(di)
            iou_mat[ti, :] = -1
            iou_mat[:, di] = -1

            tid, v = active_tracks[ti]
            bbox, cid, conf = dets[di]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            v.current_bbox = bbox
            v.confidence   = conf
            v.last_frame   = frame_idx
            v.frames_lost  = 0
            v.is_active    = True
            v.positions.append((cx, cy, timestamp))

    # New tracks for unmatched detections
    for di, (bbox, cid, conf) in enumerate(dets):
        if di in matched_dets:
            continue
        tid = _next_id[0]; _next_id[0] += 1
        cx  = (bbox[0] + bbox[2]) / 2.0
        cy  = (bbox[1] + bbox[3]) / 2.0
        v   = TrackedVehicle(
            track_id=tid, class_id=cid,
            class_name=VEHICLE_CLASSES[cid],
            color_rgb=CLASS_COLORS.get(cid, (180,180,180)),
            first_frame=frame_idx,
        )
        v.current_bbox = bbox
        v.confidence   = conf
        v.last_frame   = frame_idx
        v.positions.append((cx, cy, timestamp))
        tracks[tid] = v
        _s.total_unique += 1
        _s.class_counts[VEHICLE_CLASSES[cid]] += 1

    # Age unmatched active tracks
    for ti, (tid, v) in enumerate(active_tracks):
        if ti not in matched_tracks:
            v.frames_lost += 1
            if v.frames_lost > TRACK_BUFFER:
                v.is_active = False


def update_speeds(g_max, y_max, r_max):
    for v in _s.tracks.values():
        if not v.is_active:
            continue
        spd, _ = compute_speed(v, ppm, _s.ema_state)
        v.current_speed = spd
        v.zone = classify_zone(spd, g_max, y_max, r_max)


def check_line_crossing(line_y: int):
    for v in _s.tracks.values():
        if not v.is_active or v.crossed_line or len(v.positions) < 2:
            continue
        prev_y = v.positions[-2][1]
        curr_y = v.positions[-1][1]
        if (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y):
            v.crossed_line = True
            _s.crossed_counts[v.class_name] += 1


def log_violations(ts: str):
    for v in _s.tracks.values():
        if not v.is_active or v.current_speed <= 0:
            continue
        if v.zone == "red":
            # Log once per track per zone entry
            recent = [e for e in _s.violation_log[-30:] if e["track_id"] == v.track_id]
            if not recent or recent[-1]["zone"] != "RED":
                _s.violation_log.append({
                    "time":     ts,
                    "track_id": v.track_id,
                    "class":    v.class_name,
                    "speed_kmh":round(v.current_speed, 1),
                    "zone":     "RED",
                    "conf":     round(v.confidence, 2),
                })
            # Trigger alert banner for new violations
            if v.track_id not in _s.alert_shown_ids:
                _s.alert_shown_ids.add(v.track_id)
                _s.violation_alert = True
                _s.alert_vehicle = {
                    "class": v.class_name,
                    "speed": round(v.current_speed, 1),
                    "id":    v.track_id,
                }


def update_fastest():
    """
    Update session fastest/slowest KPIs.
    Requires vehicle to have at least 8 tracked positions (filters out
    new/transient detections that produce spike readings).
    Uses the median of the last 5 ema values from ema_state history
    for extra stability — not the raw instantaneous current_speed.
    """
    for v in _s.tracks.values():
        if not v.is_active:
            continue
        # Must have enough history to be a reliable reading
        if len(v.positions) < 8:
            continue
        spd = v.current_speed
        # Ignore zero and implausible spikes
        if spd < 2.0 or spd > MAX_SPEED_KMH:
            continue
        if spd > _s.fastest["speed"]:
            _s.fastest = {"speed": spd, "class": v.class_name, "id": v.track_id}
        if spd < _s.slowest["speed"]:
            _s.slowest = {"speed": spd, "class": v.class_name, "id": v.track_id}


def tick_fps():
    _s._fps_frames += 1
    elapsed = time.time() - _s._fps_t
    if elapsed >= 1.0:
        _s.fps_actual     = _s._fps_frames / elapsed
        _s._fps_frames    = 0
        _s._fps_t         = time.time()


# ════════════════════════════════════════════════════════════════════════════
# VIDEO WRITER HELPER
# ════════════════════════════════════════════════════════════════════════════

def make_writer(cap, path: str):
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


# ════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.processing:

    model        = _s.model
    _next_id[0]  = max((v.track_id for v in _s.tracks.values()), default=0) + 1

    # Open source
    tmp_path = None
    if src == "Video File":
        # Use cached bytes from session_state — uploaded_file is None after rerun
        v_bytes = _s.video_bytes
        v_name  = _s.video_name or "video.mp4"
        if not v_bytes:
            st.error("Video file not found. Please upload again and press Start.")
            _s.processing = False
            st.stop()
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(v_name).suffix)
        tmp.write(v_bytes); tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        tmp_path = tmp.name
    elif src == "Webcam":
        cap = cv2.VideoCapture(int(webcam_idx))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(rtsp_url.strip())
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap or not cap.isOpened():
        st.error("Cannot open video source. Check input and try again.")
        _s.processing = False
        st.stop()

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_dt = 1.0 / src_fps

    line_y = int(frame_h * line_pct / 100) if show_line else None

    writer = None
    if save_video:
        out_path = str(Path(tempfile.gettempdir()) / "speed_out.mp4")
        _s.output_path = out_path
        writer = make_writer(cap, out_path)

    _s.frame_idx = 0

    while _s.processing:
        t_loop = time.time()

        ret, frame = cap.read()
        if not ret:
            if src == "Video File":
                st.success("Video processing complete.")
            break

        _s.frame_idx += 1
        timestamp = _s.frame_idx / src_fps

        # ── Detect with YOLOv8 predict (no tracker dependency) ───────────
        results = model.predict(
            source=frame,
            conf=conf_thr,
            iou=iou_thr,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        raw_boxes = raw_classes = raw_confs = None
        if results and results[0].boxes is not None:
            r = results[0].boxes
            if len(r) > 0:
                raw_boxes   = r.xyxy.cpu().numpy()
                raw_classes = r.cls.cpu().numpy().astype(int)
                raw_confs   = r.conf.cpu().numpy()

        # ── Update IoU tracker ────────────────────────────────────────────
        update_tracks(raw_boxes, raw_classes, raw_confs, timestamp, _s.frame_idx)
        update_speeds(green_max, yellow_max, red_max)
        if line_y is not None:
            check_line_crossing(line_y)
        update_fastest()

        ts_str = datetime.now().strftime("%H:%M:%S")
        log_violations(ts_str)
        tick_fps()

        # ── Annotate ──────────────────────────────────────────────────────
        active = [v for v in _s.tracks.values() if v.is_active]
        annotated = annotate_frame(
            frame, active, line_y,
            show_trail, _p1, _p2, ref_dist, show_calib,
            _s.fps_actual,
        )

        if writer:
            writer.write(annotated)

        # ── Display ───────────────────────────────────────────────────────
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_ph.image(rgb, channels="RGB", use_container_width=True)

        if src == "Video File" and total_fr > 0:
            prog_ph.progress(
                min(_s.frame_idx / total_fr, 1.0),
                text=f"Frame {_s.frame_idx}/{total_fr}")

        render_counts()
        render_fastest()
        render_violations()
        render_alert()

        # ── Frame pacing ──────────────────────────────────────────────────
        elapsed = time.time() - t_loop
        sleep_t = max(0.0, target_dt - elapsed - 0.002)
        if sleep_t > 0:
            time.sleep(sleep_t)

    cap.release()
    if writer:
        writer.release()
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    _s.processing = False
    st.rerun()

else:
    # ── Idle placeholder ──────────────────────────────────────────────────
    video_ph = st.empty()
    video_ph.markdown(
        "<div class='idle-box'>"
        "<div style='font-size:2.2rem;color:#1e2d40;'>&#9654;</div>"
        "<div style='font-family:Share Tech Mono,monospace;font-size:.8rem;"
        "color:#484f58;letter-spacing:.15em;'>SELECT SOURCE &amp; PRESS START</div>"
        "</div>",
        unsafe_allow_html=True)

# ── Dashboard copyright footer ─────────────────────────────────────────────
st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='"
    "border-top:1px solid #1e2d40;"
    "padding:14px 0 6px;"
    "text-align:center;"
    "font-family:Share Tech Mono,monospace;"
    "font-size:0.72rem;"
    "letter-spacing:0.12em;"
    "'>"
    "<span style='color:#fb923c;font-weight:700;'>© SAI VIGNESH</span>"
    "<span style='color:#30363d;'> &nbsp;|&nbsp; </span>"
    "<span style='color:#484f58;'>REAL-TIME VEHICLE SPEED DETECTION SYSTEM</span>"
    "<span style='color:#30363d;'> &nbsp;|&nbsp; </span>"
    "<span style='color:#484f58;'>INDIA ROADS &nbsp;·&nbsp; </span>"
    "</div>",
    unsafe_allow_html=True)
