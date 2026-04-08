"""
detection.py
============
Vehicle detection module using YOLOv8.
Handles model loading, inference, class filtering, and result packaging.

Supported vehicle classes (COCO + extended mapping):
  - Car, Motorcycle/Bike, Bus, Truck, Bicycle
  - Auto-rickshaw (mapped from car class in Indian context)
  - Van/Tempo (mapped from truck/car)
"""

import numpy as np
import torch
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# COCO class IDs for vehicles
# ─────────────────────────────────────────────
VEHICLE_CLASS_IDS = {
    1:  "Bicycle",
    2:  "Car",
    3:  "Motorcycle",
    5:  "Bus",
    7:  "Truck",
}

# Extended display labels with Indian context
DISPLAY_LABELS = {
    1:  "Bicycle",
    2:  "Car",
    3:  "Bike",
    5:  "Bus",
    7:  "Truck",
}

# Colors per class (BGR for OpenCV, RGB for display)
CLASS_COLORS_RGB = {
    1:  (100, 200, 255),   # Bicycle  - light blue
    2:  (50,  200, 100),   # Car      - green
    3:  (255, 160, 50),    # Bike     - orange
    5:  (220, 80,  220),   # Bus      - purple
    7:  (255, 80,  80),    # Truck    - red
    -1: (180, 180, 180),   # Unknown  - gray
}

# Speed zone colors (RGB)
ZONE_COLORS = {
    "green":  (34,  197, 94),
    "yellow": (234, 179, 8),
    "red":    (239, 68,  68),
}


@dataclass
class Detection:
    """Single vehicle detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float]
    color_rgb: Tuple[int, int, int]


@dataclass
class DetectionResult:
    """Frame-level detection output."""
    detections: List[Detection]
    frame_shape: Tuple[int, int, int]   # H, W, C
    model_inference_ms: float


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Parameters
    ----------
    model_path : str
        Path to YOLO weights file (e.g. 'yolov8n.pt', 'yolov8s.pt').
    confidence_threshold : float
        Minimum detection confidence (0-1).
    iou_threshold : float
        NMS IoU threshold.
    device : str
        'cuda', 'cpu', or 'mps' (Apple Silicon).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device or self._select_device()

        logger.info(f"Loading YOLO model: {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Warm up model
        self._warmup()
        logger.info("VehicleDetector ready.")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a single BGR frame.

        Returns DetectionResult with all vehicle detections.
        """
        import time
        t0 = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=list(VEHICLE_CLASS_IDS.keys()),
            verbose=False,
            device=self.device,
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        detections = self._parse_results(results)

        return DetectionResult(
            detections=detections,
            frame_shape=frame.shape,
            model_inference_ms=inference_ms,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """Run detection on a batch of frames."""
        import time
        t0 = time.perf_counter()

        results_list = self.model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=list(VEHICLE_CLASS_IDS.keys()),
            verbose=False,
            device=self.device,
        )

        inference_ms = (time.perf_counter() - t0) * 1000 / max(len(frames), 1)
        return [
            DetectionResult(
                detections=self._parse_results([r]),
                frame_shape=frames[i].shape,
                model_inference_ms=inference_ms,
            )
            for i, r in enumerate(results_list)
        ]

    def update_thresholds(self, conf: float, iou: float):
        """Dynamically update detection thresholds."""
        self.confidence_threshold = conf
        self.iou_threshold = iou

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _parse_results(self, results) -> List[Detection]:
        """Convert YOLO result objects to Detection dataclasses."""
        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes  = result.boxes.xyxy.cpu().numpy()    # [N, 4]
            confs  = result.boxes.conf.cpu().numpy()    # [N]
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                x1, y1, x2, y2 = bbox.astype(int)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                detections.append(Detection(
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=DISPLAY_LABELS.get(cls_id, "Vehicle"),
                    center=center,
                    color_rgb=CLASS_COLORS_RGB.get(cls_id, CLASS_COLORS_RGB[-1]),
                ))

        return detections

    def _warmup(self):
        """Run one dummy inference to initialize CUDA kernels."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            conf=0.9,
            verbose=False,
            device=self.device,
        )

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected. Using GPU acceleration.")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS detected.")
            return "mps"
        else:
            logger.info("No GPU found. Using CPU.")
            return "cpu"


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def get_zone_color(speed_kmh: float, green_max: float, yellow_max: float) -> str:
    """Return 'green', 'yellow', or 'red' based on speed thresholds."""
    if speed_kmh <= green_max:
        return "green"
    elif speed_kmh <= yellow_max:
        return "yellow"
    else:
        return "red"


def get_class_summary(detections: List[Detection]) -> dict:
    """Return count per vehicle class from a list of detections."""
    summary = {name: 0 for name in DISPLAY_LABELS.values()}
    for d in detections:
        label = d.class_name
        if label in summary:
            summary[label] += 1
    return summary


def list_available_models() -> List[str]:
    """Return common YOLOv8 model sizes."""
    return [
        "yolov8n.pt",   # Nano  - fastest, lowest accuracy
        "yolov8s.pt",   # Small - good balance
        "yolov8m.pt",   # Medium
        "yolov8l.pt",   # Large
        "yolov8x.pt",   # XLarge - highest accuracy, slowest
        "yolov9c.pt",   # YOLOv9 compact
        "yolov9e.pt",   # YOLOv9 extended
    ]
