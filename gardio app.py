"""
Vehicle Speed Detection — India Roads
Single-file | Gradio | Local + HuggingFace
Run: python app.py
"""
import os, time, tempfile, subprocess, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")
import cv2, numpy as np, pandas as pd, gradio as gr
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
COCO_VEH = {
    1:"Bicycle", 2:"Car", 3:"Bike",
    5:"Bus",     7:"Truck", 6:"Train", 8:"Boat"
}
VEH_IDS = list(COCO_VEH.keys())
VEH_LEN = {1:1.8, 2:4.5, 3:2.2, 5:12.0, 7:8.0, 6:20.0, 8:6.0}
ZC = {
    "green" : (63,185,80),
    "yellow": (210,153,34),
    "red"   : (248,81,73),
}
INFER_W = 320      # YOLO inference width
DISP_W  = 640      # display/output width
MAX_KMH = 200.0

MODEL_OPTS = {
    "yolov8n — Fastest (CPU)" : "yolov8n.pt",
    "yolov8s — Balanced"      : "yolov8s.pt",
    "yolov8m — Most Accurate" : "yolov8m.pt",
}

# Tracker thresholds
HIT_THRESH  = 2   # frames before track is "confirmed real vehicle"
LOST_THRESH = 20  # frames before track deleted

# ═══════════════════════════════════════════════════════════════════
# TRACK DATACLASS
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Track:
    tid:    int
    cid:    int
    cname:  str
    bbox:   np.ndarray
    conf:   float = 0.0
    hits:   int   = 0
    lost:   int   = 0
    active: bool  = True
    spd:    float = 0.0
    zone:   str   = "green"
    # trail = (cx, cy, video_ts_sec) — video time, NOT wall clock
    trail: deque = field(default_factory=lambda: deque(maxlen=90))

# ═══════════════════════════════════════════════════════════════════
# MODEL CACHE
# ═══════════════════════════════════════════════════════════════════
_mcache: dict = {}
def load_model(name: str) -> YOLO:
    if name not in _mcache:
        m = YOLO(name)
        m.predict(np.zeros((INFER_W,INFER_W,3), dtype=np.uint8),
                  verbose=False, conf=0.9, imgsz=INFER_W)
        _mcache[name] = m
    return _mcache[name]

# ═══════════════════════════════════════════════════════════════════
# DETECTION
# ═══════════════════════════════════════════════════════════════════
def run_detect(model, frame, conf_t, iou_t):
    """YOLO → list of (bbox_xyxy in original coords, coco_id, conf)"""
    H, W = frame.shape[:2]
    sc   = INFER_W / max(W, 1)
    IH   = max(1, int(H * sc))
    sm   = cv2.resize(frame, (INFER_W, IH))
    res  = model.predict(source=sm, conf=conf_t, iou=iou_t,
                         classes=VEH_IDS, verbose=False, imgsz=INFER_W)
    dets = []
    if res and res[0].boxes is not None:
        r = res[0].boxes
        if len(r) > 0:
            xyxy  = r.xyxy.cpu().numpy().copy()
            clses = r.cls.cpu().numpy().astype(int)
            confs = r.conf.cpu().numpy()
            xyxy[:,[0,2]] /= sc
            xyxy[:,[1,3]] /= sc
            for i in range(len(r)):
                dets.append((xyxy[i], int(clses[i]), float(confs[i])))
    return dets

# ═══════════════════════════════════════════════════════════════════
# TRACKER  (greedy IoU — ByteTrack Stage-1 equivalent)
# ═══════════════════════════════════════════════════════════════════
_nid = [1]

def _iou(a, b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    inter  = max(0,x2-x1)*max(0,y2-y1)
    if inter == 0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

class Tracker:
    def __init__(self):
        self.tracks: dict = {}

    def reset(self):
        self.tracks.clear()
        _nid[0] = 1

    def update(self, dets, ts: float):
        dets   = [(b,c,f) for b,c,f in dets if c in COCO_VEH]
        active = [t for t in self.tracks.values() if t.active]
        mt, md = set(), set()

        if active and dets:
            M = np.zeros((len(active), len(dets)))
            for ti,trk in enumerate(active):
                for di,(b,_,_) in enumerate(dets):
                    M[ti,di] = _iou(trk.bbox, b)
            while True:
                ti,di = np.unravel_index(np.argmax(M), M.shape)
                if M[ti,di] < 0.25: break
                mt.add(ti); md.add(di)
                M[ti,:]=0; M[:,di]=0
                trk = active[ti]; b,c,cf = dets[di]
                trk.bbox = b.copy(); trk.conf = cf
                trk.lost = 0; trk.hits += 1
                cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
                trk.trail.append((cx,cy,ts))
                # ── confirmed on hit == HIT_THRESH ───────────────
                if trk.hits == HIT_THRESH:
                    S.total        += 1
                    S.ccnt[trk.cname] += 1

        for di,(b,c,cf) in enumerate(dets):
            if di in md: continue
            tid = _nid[0]; _nid[0] += 1
            cname = COCO_VEH.get(c,"Other")
            t = Track(tid=tid, cid=c, cname=cname,
                      bbox=b.copy(), conf=cf, hits=1)
            t.trail.append(((b[0]+b[2])/2,(b[1]+b[3])/2,ts))
            self.tracks[tid] = t

        for ti,trk in enumerate(active):
            if ti not in mt:
                trk.lost += 1
                if trk.lost > LOST_THRESH:
                    trk.active = False

        return [t for t in self.tracks.values()
                if t.active and t.hits >= HIT_THRESH]

# ═══════════════════════════════════════════════════════════════════
# SPEED  — auto-PPM + multi-window median, video timestamps
# ═══════════════════════════════════════════════════════════════════
_ppm_hist: dict = defaultdict(list)
_spd_hist: dict = defaultdict(list)
_ema:      dict = {}

def _get_ppm(t: Track) -> Optional[float]:
    b = t.bbox
    w = max(b[2]-b[0], 1); h = max(b[3]-b[1], 1)
    real_m = VEH_LEN.get(t.cid, 4.5)
    # Use the LARGER dimension as the vehicle's pixel size
    # For side view: width is vehicle length
    # For top view: height or width ≈ vehicle length  
    # Larger is safer — avoids underestimating PPM
    if w > h * 1.2:
        px = w        # side view — width = vehicle length
    elif h > w * 1.2:
        px = h        # near-top view
    else:
        px = max(w, h)  # diagonal — use max
    if px < 20: return None
    est = px / real_m
    _ppm_hist[t.tid].append(est)
    if len(_ppm_hist[t.tid]) > 15: _ppm_hist[t.tid].pop(0)
    # Use 60th percentile (not median) — avoids underestimating from partial detections
    return float(np.percentile(_ppm_hist[t.tid], 60))

def compute_speed(t: Track, fallback_ppm=40.0) -> float:
    """
    Accurate per-vehicle speed.
    Key fixes vs naive approach:
      * ts = fi/SFPS  (video time, not wall clock)
      * Auto-PPM from bbox (12-sample median)
      * Multi-window (2-6 frames) median — robust to jitter
      * 3-sample temporal history — fast convergence
      * EMA init = first reading (not 0)  ← biggest fix
      * alpha=0.55 — responsive, not lagging
    """
    p = t.trail
    if len(p) < 3: return 0.0
    ppm  = _get_ppm(t) or fallback_ppm
    raws = []
    for w in [3, 4, 5, 6, 8, 10]:
        if len(p) <= w: continue
        x1,y1,t1 = p[-1-w];  x2,y2,t2 = p[-1]
        dt = t2 - t1
        if dt < 0.01: continue
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if dist < 1.5: continue          # pure bbox jitter — skip
        raws.append(min((dist/ppm/dt)*3.6, MAX_KMH))
    if not raws:
        return round(_ema.get(t.tid, 0.0), 1)
    med = float(np.median(raws))
    h   = _spd_hist[t.tid]
    h.append(med)
    if len(h) > 3: h.pop(0)
    tm   = float(np.median(h))
    prev = _ema.get(t.tid, None)
    sm   = tm if prev is None else 0.70*tm + 0.30*prev  # fast convergence
    _ema[t.tid] = sm
    return round(sm, 1)

def cleanup_speed(alive: set):
    for d in [_ppm_hist, _spd_hist, _ema]:
        for k in list(d.keys()):
            if k not in alive: del d[k]

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
class S:
    tracker     = Tracker()
    total       = 0               # confirmed unique vehicles
    ccnt: dict  = defaultdict(int)  # per-class confirmed count
    xcnt: dict  = defaultdict(int)  # per-class line-crossed count
    counted_ids: set = set()        # IDs already counted (no dups)
    fastest     = dict(spd=0.0,  cls="---")
    slowest     = dict(spd=9999.0, cls="---")
    fps         = 0.0; _fn=0; _ft=0.0
    density     = "LOW";  density_pct=0.0
    road_type   = "Road"
    vlog:  list = []
    alids: set  = set()

    @classmethod
    def reset(cls):
        cls.tracker.reset()
        _ppm_hist.clear(); _spd_hist.clear(); _ema.clear()
        cls.total=0
        cls.ccnt=defaultdict(int); cls.xcnt=defaultdict(int)
        cls.counted_ids=set()
        cls.fastest=dict(spd=0.0,cls="---")
        cls.slowest=dict(spd=9999.0,cls="---")
        cls.fps=0.0; cls._fn=0; cls._ft=time.time()
        cls.density="LOW"; cls.density_pct=0.0
        cls.road_type="Road"
        cls.vlog=[]; cls.alids=set()

# ═══════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════════════
def gzone(s,g,y): return "green" if s<=g else "yellow" if s<=y else "red"

def count_line(tracks, line_y):
    """Each unique track ID counted once when centre crosses line_y."""
    for t in tracks:
        if t.tid in S.counted_ids or len(t.trail)<2: continue
        py,cy = t.trail[-2][1], t.trail[-1][1]
        if (py<line_y<=cy) or (py>line_y>=cy):
            S.counted_ids.add(t.tid)
            S.xcnt[t.cname] += 1

def upd_density(tracks, fw, fh):
    occ = sum(
        max(0,t.bbox[2]-t.bbox[0]) * max(0,t.bbox[3]-t.bbox[1])
        for t in tracks if t.bbox is not None)
    S.density_pct = round(min(occ / max(fw*fh,1) * 100, 100), 1)
    dp = S.density_pct
    if   dp < 8:  S.density="LOW"
    elif dp < 22: S.density="MEDIUM"
    elif dp < 42: S.density="HIGH"
    else:         S.density="VERY HIGH"

def upd_road(tracks):
    sp  = [t.spd for t in tracks if t.spd>1]
    avg = float(np.mean(sp)) if sp else 0.0
    n   = len(tracks)
    if   avg>80 and n<6: S.road_type="Highway"
    elif avg>55:         S.road_type="Express Road"
    elif avg<25 and n>8: S.road_type="City Traffic"
    elif avg<35:         S.road_type="City/Village"
    else:                S.road_type="Urban Road"

def upd_records(tracks):
    for t in tracks:
        if t.spd < 1 or len(t.trail) < 5: continue
        if t.spd > S.fastest["spd"]:
            S.fastest = dict(spd=t.spd, cls=t.cname)
        # Slowest: only vehicles clearly moving (>5 km/h) to avoid
        # parked/stationary noise pulling slowest to 1 km/h
        if 3 < t.spd < S.slowest["spd"]:
            S.slowest = dict(spd=t.spd, cls=t.cname)

def log_vio(tracks, r_max):
    ts = datetime.now().strftime("%H:%M:%S")
    for t in tracks:
        # Log violation when vehicle is in red zone (spd > yellow threshold)
        if t.zone != "red": continue
        already = [e for e in S.vlog[-30:] if e["tid"]==t.tid]
        # Update existing entry with latest speed, or add new
        if already:
            already[-1]["spd"] = round(max(already[-1]["spd"], t.spd), 1)
        else:
            S.vlog.append(dict(time=ts, tid=t.tid, cls=t.cname,
                               spd=round(t.spd,1), conf=round(t.conf,2)))
        if t.tid not in S.alids:
            S.alids.add(t.tid)

def tick():
    S._fn += 1; e = time.time() - S._ft
    if e >= 1.0: S.fps = S._fn/e; S._fn=0; S._ft=time.time()

# ═══════════════════════════════════════════════════════════════════
# ANNOTATION
# ═══════════════════════════════════════════════════════════════════
def _bgr(r): return (r[2],r[1],r[0])

def annotate(frame, tracks, line_y, show_trail, fps_v):
    out = frame.copy(); H,W = out.shape[:2]
    fn  = cv2.FONT_HERSHEY_SIMPLEX
    for t in tracks:
        if t.bbox is None: continue
        x1,y1,x2,y2 = t.bbox.astype(int)
        zc = _bgr(ZC[t.zone])
        if show_trail and len(t.trail)>=2:
            pts = list(t.trail)[-20:]
            for i in range(1, len(pts)):
                a = i/len(pts)
                cv2.line(out,
                    (int(pts[i-1][0]),int(pts[i-1][1])),
                    (int(pts[i][0]),  int(pts[i][1])),
                    tuple(int(c*a*.85) for c in zc), 2)
        cv2.rectangle(out,(x1,y1),(x2,y2),zc,2)
        if t.zone=="red":
            cv2.rectangle(out,(x1-3,y1-3),(x2+3,y2+3),(0,0,200),2)
        l1 = f"{t.cname} #{t.tid}"
        l2 = f"{t.spd:.0f} km/h"
        (w1,h1),_ = cv2.getTextSize(l1,fn,.43,1)
        (w2,h2),_ = cv2.getTextSize(l2,fn,.48,1)
        bw=max(w1,w2)+10; bh=h1+h2+14; ty=max(y1-bh-3,0)
        cv2.rectangle(out,(x1,ty),(x1+bw,ty+bh),(18,18,18),-1)
        cv2.rectangle(out,(x1,ty),(x1+bw,ty+bh),zc,1)
        cv2.putText(out,l1,(x1+4,ty+h1+3),fn,.43,(200,200,200),1,cv2.LINE_AA)
        # Speed: black outline + cyan text → visible on any background
        cv2.putText(out,l2,(x1+4,ty+h1+h2+11),fn,.48,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,l2,(x1+4,ty+h1+h2+11),fn,.48,(0,255,180),1,cv2.LINE_AA)
    if line_y:
        cv2.line(out,(0,line_y),(W,line_y),(0,220,220),2)
        cv2.putText(out,"COUNT LINE",(8,line_y-6),fn,.42,(0,220,220),1,cv2.LINE_AA)
    # Top-right panel
    ac={}
    for t in tracks: ac[t.cname]=ac.get(t.cname,0)+1
    lines=[f"Live:{len(tracks)}"]+[f"{k}:{v}" for k,v in ac.items()]
    px=W-112; py=6; ov=out.copy()
    cv2.rectangle(ov,(px-4,py-2),(px+110,py+len(lines)*17+4),(8,10,20),-1)
    cv2.addWeighted(ov,.78,out,.22,0,out)
    for i,ln in enumerate(lines):
        cv2.putText(out,ln,(px,py+(i+1)*17-2),fn,.42,
            (56,189,248) if i==0 else (165,165,165),1,cv2.LINE_AA)
    cv2.putText(out,f"FPS:{fps_v:.1f}",(8,20),fn,.42,(100,100,100),1,cv2.LINE_AA)
    return out

# ═══════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def _card(val,lbl,col="#e6edf3"):
    return (f"<div style='background:#0d1117;border:1px solid #1e2d40;"
            f"border-radius:8px;padding:12px 6px;text-align:center;flex:1;min-width:75px;'>"
            f"<div style='font-family:monospace;font-size:1.2rem;font-weight:700;"
            f"color:{col};line-height:1.1;'>{val}</div>"
            f"<div style='font-family:monospace;font-size:.48rem;letter-spacing:.1em;"
            f"color:#484f58;text-transform:uppercase;margin-top:3px;'>{lbl}</div></div>")

def html_kpi(tracks, g_max, y_max):
    try:
        spds = [t.spd for t in tracks if t.spd>1 and len(t.trail)>=5]
        mx   = max(spds, default=0.0)
        mc   = "#f85149" if mx>y_max else "#d29922" if mx>g_max else "#3fb950"
        fs   = S.fastest["spd"]
        ss   = S.slowest["spd"] if S.slowest["spd"]<9999 else 0.0
        dc   = {"LOW":"#3fb950","MEDIUM":"#d29922",
                "HIGH":"#f85149","VERY HIGH":"#ff2222"}.get(S.density,"#3fb950")
        dp   = S.density_pct; bw=min(int(dp),100)
        dbar = (f"<div style='background:rgba(255,255,255,.04);border:1px solid {dc};"
                f"border-radius:8px;padding:10px 14px;margin-top:8px;'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"font-family:monospace;margin-bottom:5px;'>"
                f"<span style='font-size:.58rem;color:#484f58;'>TRAFFIC DENSITY</span>"
                f"<span style='font-size:.88rem;font-weight:700;color:{dc};'>{S.density}</span>"
                f"<span style='font-size:.56rem;color:#484f58;'>"
                f"{len(tracks)} live | Area: {dp:.0f}% | {S.road_type}</span></div>"
                f"<div style='background:#1e2d40;border-radius:4px;height:5px;'>"
                f"<div style='background:{dc};width:{bw}%;height:100%;"
                f"border-radius:4px;'></div></div></div>")
        return (f"<div style='display:flex;gap:5px;flex-wrap:wrap;'>"
                f"{_card(S.total,'Total Detected')}"
                f"{_card(len(tracks),'Active In Frame','#388bfd')}"
                f"{_card(f'{mx:.0f} km/h','Max Speed Now',mc)}"
                f"{_card(f'{fs:.0f} km/h','Session Fastest','#f85149')}"
                f"{_card(f'{ss:.0f} km/h','Session Slowest','#3fb950')}"
                f"{_card(f'{S.fps:.1f}','Proc FPS','#388bfd')}"
                f"</div>{dbar}")
    except Exception as e:
        return f"<p style='color:#484f58;font-family:monospace;padding:8px;'>Loading... ({e})</p>"

def html_counts():
    try:
        rows=""
        for cls in sorted(set(COCO_VEH.values())):
            n   = S.ccnt.get(cls,0)
            col = "#e6edf3" if n>0 else "#484f58"
            rows += (f"<div style='display:flex;justify-content:space-between;"
                     f"padding:5px 10px;background:#0d1117;border:1px solid #1e2d40;"
                     f"border-radius:5px;margin:2px 0;font-family:monospace;font-size:.72rem;'>"
                     f"<span style='color:#484f58;'>{cls}</span>"
                     f"<span style='color:{col};font-weight:700;'>{n}</span></div>")
        crossed = sum(S.xcnt.values())
        rows += (f"<div style='display:flex;justify-content:space-between;"
                 f"padding:5px 10px;background:#0d1117;border:1px solid #d29922;"
                 f"border-radius:5px;margin:4px 0;font-family:monospace;font-size:.72rem;'>"
                 f"<span style='color:#d29922;'>Line Crossed</span>"
                 f"<span style='color:#d29922;font-weight:700;'>{crossed}</span></div>")
        fs = S.fastest["spd"]
        ss = S.slowest["spd"] if S.slowest["spd"]<9999 else 0.0
        rows += (f"<div style='margin-top:8px;font-family:monospace;font-size:.51rem;"
                 f"color:#484f58;border-bottom:1px solid #1e2d40;padding-bottom:3px;"
                 f"margin-bottom:5px;letter-spacing:.11em;'>SPEED RECORDS</div>"
                 f"<div style='display:flex;justify-content:space-between;padding:5px 10px;"
                 f"background:#0d1117;border:1px solid #1e2d40;border-radius:5px;margin:2px 0;"
                 f"font-family:monospace;font-size:.71rem;'>"
                 f"<span style='color:#484f58;'>Fastest</span>"
                 f"<span style='color:#f85149;font-weight:700;'>"
                 f"{fs:.0f} km/h — {S.fastest['cls']}</span></div>"
                 f"<div style='display:flex;justify-content:space-between;padding:5px 10px;"
                 f"background:#0d1117;border:1px solid #1e2d40;border-radius:5px;margin:2px 0;"
                 f"font-family:monospace;font-size:.71rem;'>"
                 f"<span style='color:#484f58;'>Slowest (moving)</span>"
                 f"<span style='color:#3fb950;font-weight:700;'>"
                 f"{ss:.0f} km/h — {S.slowest['cls']}</span></div>"
                 f"<div style='display:flex;justify-content:space-between;padding:5px 10px;"
                 f"background:#0d1117;border:1px solid #1e2d40;border-radius:5px;margin:2px 0;"
                 f"font-family:monospace;font-size:.71rem;'>"
                 f"<span style='color:#484f58;'>Violations</span>"
                 f"<span style='color:#f85149;font-weight:700;'>{len(S.vlog)}</span></div>")
        return f"<div>{rows}</div>"
    except Exception as e:
        return f"<p style='color:#484f58;font-family:monospace;padding:8px;'>Loading... ({e})</p>"

def html_vio():
    try:
        if not S.vlog:
            return ("<div style='background:#0d1117;border:1px dashed #1e2d40;"
                    "border-radius:8px;padding:14px;text-align:center;"
                    "font-family:monospace;font-size:.7rem;color:#484f58;'>"
                    "No violations — vehicles exceeding Red zone appear here</div>")
        rows=""
        for i,e in enumerate(reversed(S.vlog[-60:])):
            sp=e["spd"]; bg="rgba(248,81,73,.04)" if i%2==0 else "transparent"
            sc="#ff3b3b" if sp>100 else "#f85149"
            rows+=(f"<tr style='background:{bg};border-bottom:1px solid #1a2332;'>"
                   f"<td style='padding:5px 9px;color:#484f58;font-size:.64rem;'>{e['time']}</td>"
                   f"<td style='padding:5px 9px;color:#c9d1d9;font-size:.65rem;'>#{e['tid']}</td>"
                   f"<td style='padding:5px 9px;color:#c9d1d9;font-size:.67rem;'>{e['cls']}</td>"
                   f"<td style='padding:5px 9px;text-align:center;'>"
                   f"<span style='background:rgba(248,81,73,.15);border:1px solid {sc};"
                   f"border-radius:4px;padding:2px 7px;color:{sc};font-weight:700;"
                   f"font-size:.74rem;'>{sp:.0f} km/h</span></td>"
                   f"<td style='padding:5px 9px;color:#484f58;font-size:.62rem;'>{e['conf']:.0%}</td></tr>")
        return (f"<div style='background:#0d1117;border:1px solid #2a1f2d;"
                f"border-radius:10px;overflow:hidden;'>"
                f"<table style='width:100%;border-collapse:collapse;font-family:monospace;'>"
                f"<thead><tr style='background:#130d1a;'>"
                f"<th style='padding:6px 9px;text-align:left;font-size:.51rem;color:#6e40c9;'>TIME</th>"
                f"<th style='padding:6px 9px;text-align:left;font-size:.51rem;color:#6e40c9;'>ID</th>"
                f"<th style='padding:6px 9px;text-align:left;font-size:.51rem;color:#6e40c9;'>VEHICLE</th>"
                f"<th style='padding:6px 9px;text-align:center;font-size:.51rem;color:#6e40c9;'>SPEED</th>"
                f"<th style='padding:6px 9px;text-align:left;font-size:.51rem;color:#6e40c9;'>CONF</th>"
                f"</tr></thead><tbody>{rows}</tbody></table>"
                f"<div style='padding:4px 10px;border-top:1px solid #1e2d40;"
                f"font-size:.55rem;color:#484f58;font-family:monospace;'>"
                f"Total: {len(S.vlog)}</div></div>")
    except Exception as e:
        return f"<p style='color:#484f58;font-family:monospace;padding:8px;'>Loading... ({e})</p>"

_W  = "<p style='color:#484f58;font-family:monospace;font-size:.72rem;padding:10px;'>Upload a video and press Start.</p>"
_WC = "<p style='color:#484f58;font-family:monospace;font-size:.72rem;padding:10px;'>Waiting...</p>"
_WV = ("<div style='background:#0d1117;border:1px dashed #1e2d40;border-radius:8px;"
       "padding:14px;text-align:center;font-family:monospace;font-size:.7rem;color:#484f58;'>"
       "No violations yet</div>")

# ═══════════════════════════════════════════════════════════════════
# WEBCAM LIVE DETECTION
# ═══════════════════════════════════════════════════════════════════
def webcam_fn(frame, model_choice, conf_t, iou_t, g_max, y_max, r_max):
    """Process single webcam frame — called every 250ms by Gradio stream."""
    if frame is None: return None
    try:
        model = load_model(MODEL_OPTS.get(model_choice,"yolov8n.pt"))
        H,W   = frame.shape[:2]
        sc    = INFER_W/max(W,1); IH=max(1,int(H*sc))
        sm    = cv2.resize(frame,(INFER_W,IH))
        res   = model.predict(source=sm,conf=conf_t,iou=iou_t,
                              classes=VEH_IDS,verbose=False,imgsz=INFER_W)
        out   = frame.copy(); fn=cv2.FONT_HERSHEY_SIMPLEX
        if res and res[0].boxes is not None:
            r = res[0].boxes
            for i in range(len(r)):
                cid  = int(r.cls[i]); cf=float(r.conf[i])
                name = COCO_VEH.get(cid,"Vehicle")
                x1,y1,x2,y2=[int(v/sc) for v in r.xyxy[i].cpu().numpy()]
                x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
                col=(72,201,112); zc=_bgr(col)
                cv2.rectangle(out,(x1,y1),(x2,y2),zc,2)
                lbl=f"{name} {cf:.0%}"
                cv2.putText(out,lbl,(x1+3,max(y1-5,14)),fn,.44,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(out,"WEBCAM LIVE",(8,H-10),fn,.44,(56,189,248),1,cv2.LINE_AA)
        return out
    except Exception:
        return frame

# ═══════════════════════════════════════════════════════════════════
# MAIN VIDEO PIPELINE  (Gradio generator)
# ═══════════════════════════════════════════════════════════════════
def process_video(video_file, model_choice, conf_t, iou_t,
                  g_max, y_max, r_max,
                  show_line, line_pct, show_trail,
                  progress=gr.Progress()):

    if video_file is None:
        yield None,_W,_WC,_WV,None,None; return

    S.reset()
    model = load_model(MODEL_OPTS[model_choice])

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        yield None,_W,_WC,_WV,None,None; return

    FW    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    FH    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    SFPS  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dsc   = DISP_W/max(FW,1)
    DH    = max(1, int(FH*dsc))
    LY    = int(FH*line_pct/100) if show_line else None

    SKIP    = 1   # skip 1 frame → detect every 2nd frame — better speed resolution
    out_fps = max(SFPS/(SKIP+1), 6.0)

    raw_p = str(Path(tempfile.gettempdir())/"spd_raw.mp4")
    fin_p = str(Path(tempfile.gettempdir())/"spd_fin.mp4")
    aud_p = str(Path(tempfile.gettempdir())/"spd_aud.mp4")
    # Try H264 directly so browser can play without ffmpeg conversion
    for codec in ["avc1", "H264", "mp4v"]:
        wrt = cv2.VideoWriter(raw_p, cv2.VideoWriter_fourcc(*codec),
                              out_fps, (DISP_W,DH))
        if wrt.isOpened(): break

    fi = 0
    # Keep last known HTML — never send None (causes dashboard blink)
    last_kpi = _W; last_cnt = _WC; last_vio = _WV
    # Time-gate: max 8 yields/sec to prevent websocket overload → lag
    YIELD_INTERVAL = 0.12   # seconds between yields (≈8 fps display)
    last_yield_t   = 0.0

    while True:
        for _ in range(SKIP):
            ok,_ = cap.read()
            if not ok: break
            fi += 1
        ret, frame = cap.read()
        if not ret or frame is None: break
        fi += 1

        ts     = fi / SFPS
        dets   = run_detect(model, frame, conf_t, iou_t)
        tracks = S.tracker.update(dets, ts)

        for t in tracks:
            t.spd  = compute_speed(t)
            t.zone = gzone(t.spd, g_max, y_max)

        if LY: count_line(tracks, LY)

        upd_density(tracks, FW, FH)
        upd_road(tracks)
        upd_records(tracks)
        log_vio(tracks, r_max)
        cleanup_speed({t.tid for t in S.tracker.tracks.values() if t.active})
        tick()

        ann  = annotate(frame, tracks, LY, show_trail, S.fps)
        disp = cv2.resize(ann, (DISP_W,DH))
        wrt.write(disp)
        if TOTAL > 0: progress(fi/TOTAL)

        now = time.time()
        if now - last_yield_t >= YIELD_INTERVAL:
            last_yield_t = now
            last_kpi = html_kpi(tracks,g_max,y_max)
            last_cnt = html_counts()
            last_vio = html_vio()
            # Encode as JPEG first — smaller payload, smoother Gradio render
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            yield rgb, last_kpi, last_cnt, last_vio, None, None

    cap.release(); wrt.release()

    final_kpi = html_kpi(tracks,g_max,y_max)
    final_cnt = html_counts()
    final_vio = html_vio()

    # ── Convert to browser-playable H264 ───────────────────────────
    # Gradio gr.Video needs: libx264 + yuv420p + faststart
    # Without these Chrome/Safari show NaN:NaN duration
    play_p = None
    try:
        # With original audio
        r = subprocess.run([
            "ffmpeg","-y",
            "-i", raw_p,
            "-i", video_file,
            "-map","0:v:0",
            "-map","1:a:0?",
            "-c:v","libx264",
            "-preset","ultrafast",
            "-crf","23",
            "-pix_fmt","yuv420p",
            "-movflags","+faststart",
            "-c:a","aac",
            "-b:a","128k",
            "-shortest",
            aud_p
        ], capture_output=True, timeout=600)
        if os.path.exists(aud_p) and os.path.getsize(aud_p) > 10000:
            play_p = aud_p
    except Exception:
        pass

    if not play_p:
        try:
            # Without audio fallback
            r2 = subprocess.run([
                "ffmpeg","-y",
                "-i", raw_p,
                "-c:v","libx264",
                "-preset","ultrafast",
                "-crf","23",
                "-pix_fmt","yuv420p",
                "-movflags","+faststart",
                "-an",
                fin_p
            ], capture_output=True, timeout=600)
            if os.path.exists(fin_p) and os.path.getsize(fin_p) > 10000:
                play_p = fin_p
        except Exception:
            pass

    # Last resort: serve raw file (may not play in all browsers)
    if not play_p:
        play_p = raw_p

    yield None, final_kpi, final_cnt, final_vio, play_p, play_p

# ═══════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════
CSS = """
body{background:#07090f;}
.gradio-container{background:#07090f!important;max-width:100%!important;}
footer{display:none!important;}
label{color:#c9d1d9!important;font-family:monospace!important;font-size:.82rem!important;}
.svelte-1ipelgc{color:#c9d1d9!important;}
span.svelte-1gfknih{color:#c9d1d9!important;}
input[type=range]{accent-color:#388bfd;}
.tabitem{background:#0d1117!important;border:1px solid #1e2d40!important;}
button.selected{background:#1f6feb!important;color:#fff!important;}
"""

def SH(t):
    return (f"<div style='font-family:monospace;font-size:.62rem;font-weight:700;"
            f"letter-spacing:.16em;color:#8b949e;border-bottom:1px solid #30363d;"
            f"padding-bottom:5px;margin-bottom:10px;margin-top:4px;'>{t}</div>")

with gr.Blocks(css=CSS, title="Vehicle Speed Monitor") as demo:

    gr.HTML("""<div style='text-align:center;padding:16px 0 8px;'>
      <h1 style='font-family:monospace;font-size:1.4rem;color:#e6edf3;margin:0;'>
        Real-Time Vehicle Speed Detection</h1>
      <p style='font-family:monospace;font-size:.57rem;color:#484f58;
        letter-spacing:.12em;margin:4px 0 0;'>
        INDIA ROADS | YOLOv8 | LIVE | SAI VIGNESH</p>
    </div>""")

    with gr.Row():

        # ── LEFT: Controls ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            gr.HTML(SH("INPUT VIDEO"))
            with gr.Tabs():

                with gr.TabItem("Upload Video"):
                    video_in = gr.Video(
                        label="Upload (any size)", sources=["upload"])

                with gr.TabItem("Webcam Live"):
                    gr.HTML("<p style='font-family:monospace;font-size:.6rem;"
                            "color:#3fb950;margin:4px 0;'>"
                            "Allow camera → detection runs automatically</p>")
                    webcam_in = gr.Image(
                        sources=["webcam"], streaming=True,
                        label="", show_label=False, height=220)
                    gr.HTML("<p style='font-family:monospace;font-size:.58rem;"
                            "color:#484f58;margin:2px 0;'>"
                            "Live boxes shown in Preview panel on the right</p>")

                with gr.TabItem("Mobile Camera"):
                    gr.HTML(
                        "<p style='font-family:monospace;font-size:.6rem;"
                        "color:#484f58;margin:4px 0;'>"
                        "<b style='color:#e6edf3;'>Option A — Phone browser:</b><br>"
                        "Open this page on phone → Webcam Live tab → Allow camera<br><br>"
                        "<b style='color:#e6edf3;'>Option B — DroidCam (Android):</b><br>"
                        "1. Install DroidCam on phone + PC<br>"
                        "2. Connect → phone becomes PC webcam<br>"
                        "3. Use Webcam Live tab above<br><br>"
                        "<b style='color:#e6edf3;'>Option C — Record &amp; Upload:</b><br>"
                        "Record on phone → Upload tab → Start Processing</p>")

            gr.HTML(SH("DETECTION MODEL"))
            model_dd = gr.Dropdown(
                choices=list(MODEL_OPTS.keys()),
                value=list(MODEL_OPTS.keys())[0],
                show_label=False)
            gr.HTML("<p style='font-family:monospace;font-size:.58rem;color:#484f58;"
                    "margin:2px 0 6px;'>n=fastest on CPU | s=balanced | m=most accurate</p>")
            with gr.Row():
                conf_sl = gr.Slider(.1,.9,.28,.05, label="Confidence",
                                    info="Lower = detect more vehicles")
                iou_sl  = gr.Slider(.1,.9,.45,.05, label="IOU")

            gr.HTML(SH("SPEED ZONES km/h"))
            g_sl = gr.Slider(5,  120, 40,  5, label="Green — Safe")
            y_sl = gr.Slider(10, 180, 80,  5, label="Yellow — Warning")
            r_sl = gr.Slider(15, 250, 120, 5, label="Red — Violation")

            gr.HTML(SH("DISPLAY"))
            with gr.Row():
                line_cb  = gr.Checkbox(value=True,  label="Count line")
                trail_cb = gr.Checkbox(value=True,  label="Motion trails")
            line_pct_sl = gr.Slider(10,90,55,1, label="Count line position %")

            run_btn = gr.Button(
                "▶  START PROCESSING", variant="primary", size="lg")

        # ── RIGHT: Outputs ──────────────────────────────────────────
        with gr.Column(scale=2):

            gr.HTML(SH("LIVE DETECTION PREVIEW"))
            live_img = gr.Image(label="", show_label=False, height=420)

            gr.HTML(SH("DASHBOARD STATS"))
            kpi_html = gr.HTML(value=_W)

            with gr.Row():
                with gr.Column(scale=1):
                    counts_html = gr.HTML(value=_WC)
                with gr.Column(scale=2):
                    vio_html    = gr.HTML(value=_WV)

    gr.HTML(SH("PROCESSED VIDEO — REVIEW &amp; DOWNLOAD"))
    with gr.Row():
        review_vid = gr.Video(label="Annotated output — click Play")
        dl_file    = gr.File(label="Download annotated video (.mp4)")

    gr.HTML("<div style='text-align:center;padding:10px 0;"
            "font-family:monospace;font-size:.56rem;color:#484f58;"
            "border-top:1px solid #1e2d40;margin-top:12px;'>"
            "<span style='color:#fb923c;font-weight:700;'>SAI VIGNESH</span>"
            " | VEHICLE SPEED DETECTION | INDIA ROADS | YOLOv8</div>")

    # ── Wire: video upload → pipeline ──────────────────────────────
    run_btn.click(
        fn=process_video,
        inputs=[video_in, model_dd, conf_sl, iou_sl,
                g_sl, y_sl, r_sl,
                line_cb, line_pct_sl, trail_cb],
        outputs=[live_img, kpi_html, counts_html, vio_html, review_vid, dl_file],
    )

    # ── Wire: webcam stream → live preview ─────────────────────────
    webcam_in.stream(
        fn=webcam_fn,
        inputs=[webcam_in, model_dd, conf_sl, iou_sl, g_sl, y_sl, r_sl],
        outputs=[live_img],
        time_limit=300,
        stream_every=0.25,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,    # public HTTPS link: https://xxx.gradio.live
        inbrowser=True,  # auto-opens browser
        show_error=True,
    )
