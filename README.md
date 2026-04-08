# Real-Time Vehicle Speed Detection & Monitoring System
## India Roads — Production Grade

---

## Architecture

```
vehicle_speed_system/
├── app.py                  # Streamlit dashboard (main entry point)
├── detection.py            # YOLOv8 vehicle detection
├── tracking.py             # ByteTrack multi-object tracking
├── speed_estimation.py     # Real-world speed calculation + annotation
├── requirements.txt        # Python dependencies
├── setup.sh                # One-command setup script
└── README.md               # This file
```

---

## System Design

```
Video/Camera Input
       │
       ▼
┌─────────────────┐
│  VehicleDetector│  ← YOLOv8 (configurable model size)
│  detection.py   │    Detects: Car, Bike, Bus, Truck, Bicycle
└────────┬────────┘
         │  List[Detection]
         ▼
┌─────────────────┐
│  VehicleTracker │  ← ByteTrack (supervision library)
│  tracking.py    │    Assigns persistent IDs, stores position history
└────────┬────────┘
         │  List[TrackedVehicle]
         ▼
┌─────────────────┐
│ SpeedEstimator  │  ← Pixel displacement → real-world speed
│speed_estimation │    EMA smoothing, zone classification
└────────┬────────┘
         │  {track_id: (speed_kmh, zone)}
         ▼
┌─────────────────┐
│ Streamlit app   │  ← Annotated frame, KPI cards, violation log
│    app.py       │    Export: CSV report + processed video
└─────────────────┘
```

---

## Quick Start

### Option 1: Automated Setup (Linux/Mac)
```bash
git clone <your-repo>
cd vehicle_speed_system
bash setup.sh
source venv/bin/activate
streamlit run app.py
```

### Option 2: Manual Setup (Windows + all platforms)
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2b. Install PyTorch (CPU only)
pip install torch torchvision

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Pre-download YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

# 5. Launch dashboard
streamlit run app.py
```

Open browser: **http://localhost:8501**

---

## Usage Guide

### Step 1: Select Input Source
| Source | How to use |
|--------|------------|
| Video File | Upload MP4, AVI, MOV, MKV |
| Webcam | Enter camera index (0 = default laptop camera) |
| IP Camera | Enter RTSP URL: `rtsp://192.168.x.x:554/stream` |
| Mobile (DroidCam) | Use HTTP URL: `http://192.168.x.x:4747/video` |

### Step 2: Configure Calibration (Critical for Accuracy)
1. Find two points in the video that are a **known real-world distance apart**  
   Examples: lane markings (3m), two parked cars, road dividers
2. Enter their pixel coordinates (P1 X, P1 Y, P2 X, P2 Y) in the sidebar
3. Enter the real-world distance between them in meters
4. The system auto-calculates pixels/meter ratio

**Reference distances:**
- Standard road lane width: **3.5 m**
- Standard car length: **4.2 – 4.7 m**
- Auto-rickshaw length: **3.0 m**
- Bus length: **10 – 12 m**

### Step 3: Set Speed Zones
- **Green**: Safe speed (e.g., 0-40 km/h for city roads)
- **Yellow**: Warning (e.g., 41-60 km/h)
- **Red**: Over-speed (above yellow max)

### Step 4: Start Processing
Click **START PROCESSING** — the system begins detecting, tracking, and measuring speed in real time.

---

## Model Selection Guide

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| yolov8n.pt | 6 MB | ~120 FPS (GPU) | Good | Webcam, real-time on low-end GPU |
| yolov8s.pt | 22 MB | ~70 FPS (GPU) | Better | **Default — best balance** |
| yolov8m.pt | 50 MB | ~40 FPS (GPU) | High | High accuracy, mid-range GPU |
| yolov8l.pt | 83 MB | ~25 FPS (GPU) | Higher | Best accuracy, good GPU |
| yolov8x.pt | 131 MB | ~15 FPS (GPU) | Highest | Offline processing only |

---

## Speed Calculation Method

### Calibration Mode (default)
```
pixel_distance = sqrt((cx2-cx1)² + (cy2-cy1)²)
real_distance_m = pixel_distance / pixels_per_meter
speed_m/s = real_distance_m / time_delta
speed_km/h = speed_m/s × 3.6
```

Speed is then smoothed using an Exponential Moving Average (alpha=0.35) to reduce jitter while remaining responsive.

### Perspective Transform Mode (advanced)
When you provide 4 road corner points, the system applies a homography transform to create a bird's-eye view, correcting for camera angle. This gives more accurate speed on cameras mounted at angles.

---

## Performance Tuning

### For maximum FPS:
- Use `yolov8n.pt` model
- Set confidence threshold to 0.5 (fewer detections = faster)
- Reduce input resolution: set `imgsz=416` in VehicleDetector

### For maximum accuracy:
- Use `yolov8l.pt` or `yolov8x.pt`
- Lower confidence to 0.25 (catch more vehicles)
- Enable GPU (CUDA)

### For Indian traffic (crowded scenes):
- Keep confidence at 0.30-0.40
- IOU threshold at 0.40-0.45 (prevents merge of adjacent vehicles)
- Use `yolov8s.pt` or `yolov8m.pt`

---

## IP Camera / Mobile Camera

### Android — DroidCam
1. Install DroidCam app on Android
2. Note IP address shown in app
3. Enter: `http://192.168.x.x:4747/video`

### iOS — EpocCam / Camo
Follow app instructions for RTSP/HTTP stream URL.

### RTSP IP Camera
```
rtsp://admin:password@192.168.1.100:554/stream1
rtsp://192.168.1.100:8554/unicast
```

---

## Output & Export

### Speed Violation Report (CSV)
Downloaded from sidebar after processing. Columns:
- `timestamp` — time of violation
- `track_id` — unique vehicle ID
- `class` — vehicle type
- `speed_kmh` — measured speed
- `zone` — RED/YELLOW

### Processed Video
Enable "Save Output Video" toggle before starting. Download from sidebar after stopping.

---

## Limitations & Notes

1. **Auto-rickshaw classification**: COCO-trained YOLOv8 may classify auto-rickshaws as "Car" or "Motorcycle". For Indian-specific classification, fine-tune on an Indian vehicle dataset (IITD-Vehicle dataset or custom labeled data).

2. **Speed accuracy**: Depends heavily on correct calibration. Camera parallax (angle) reduces accuracy for vehicles not near the reference line.

3. **Night mode**: YOLOv8 works reasonably in low light. For night use, fine-tune on nighttime data or use IR-camera streams.

4. **GPU strongly recommended**: CPU processing achieves ~3-8 FPS (acceptable for saved video, borderline for live streams).

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Detection | Ultralytics YOLO | 8.2+ |
| Tracking | Supervision ByteTrack | 0.20+ |
| Computer Vision | OpenCV | 4.8+ |
| Deep Learning | PyTorch | 2.0+ |
| Dashboard | Streamlit | 1.32+ |
| Data | Pandas + NumPy | latest |

---

## License
MIT — Free for personal and commercial use with attribution.
