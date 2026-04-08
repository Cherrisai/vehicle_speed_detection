"""
speed_estimation.py
===================
Real-world speed estimation for tracked vehicles.

Two modes are supported:

1. CALIBRATION MODE (recommended):
   - User marks two points on the road separated by a known distance (meters).
   - System derives pixels-per-meter ratio for that region.
   - Speed = pixel_displacement / ppm / time_delta * 3.6  (km/h)

2. PERSPECTIVE TRANSFORM MODE (advanced):
   - User selects 4 road corners defining a known real-world rectangle.
   - Homography maps bird's-eye view onto pixel space.
   - More accurate for wide-angle cameras and angled setups.

Speed smoothing uses an exponential moving average to eliminate jitter
while remaining responsive to genuine speed changes.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Min pixels a vehicle must move per frame to be considered "in motion"
MIN_MOTION_PIXELS = 2.0

# Max credible speed (km/h) on Indian roads for filtering outliers
MAX_CREDIBLE_SPEED_KMH = 200.0

# Smoothing factor for EMA (0 = no smoothing, 1 = infinite smoothing)
EMA_ALPHA = 0.35


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class CalibrationConfig:
    """
    Pixel-to-meter calibration derived from a reference segment.

    point1, point2: pixel coordinates of the two reference points on road.
    real_distance_m: known real-world distance between them (meters).
    """
    point1: Tuple[int, int] = (0, 0)
    point2: Tuple[int, int] = (100, 0)
    real_distance_m: float = 10.0

    @property
    def pixels_per_meter(self) -> float:
        dx = self.point2[0] - self.point1[0]
        dy = self.point2[1] - self.point1[1]
        pixel_dist = max(np.sqrt(dx**2 + dy**2), 1e-6)
        return pixel_dist / max(self.real_distance_m, 0.1)

    def is_valid(self) -> bool:
        dx = self.point2[0] - self.point1[0]
        dy = self.point2[1] - self.point1[1]
        return np.sqrt(dx**2 + dy**2) > 10 and self.real_distance_m > 0


@dataclass
class PerspectiveConfig:
    """
    4-point perspective transform configuration.

    src_points: 4 pixel coordinates on the original frame (road corners).
    dst_size: size of the bird's-eye output rectangle in meters (width, height).
    """
    src_points: Optional[np.ndarray] = None   # shape (4, 2)
    dst_width_m: float = 10.0
    dst_height_m: float = 20.0
    dst_px_per_meter: float = 20.0            # resolution of bird's-eye view

    @property
    def dst_points(self) -> np.ndarray:
        w = self.dst_width_m  * self.dst_px_per_meter
        h = self.dst_height_m * self.dst_px_per_meter
        return np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ], dtype=np.float32)

    def get_homography(self) -> Optional[np.ndarray]:
        if self.src_points is None or len(self.src_points) != 4:
            return None
        H, _ = cv2.findHomography(
            self.src_points.astype(np.float32),
            self.dst_points,
        )
        return H


# ─────────────────────────────────────────────
# Core estimator
# ─────────────────────────────────────────────

class SpeedEstimator:
    """
    Estimates real-world vehicle speed from pixel trajectory data.

    Parameters
    ----------
    calibration : CalibrationConfig
        Pixel-to-meter reference segment.
    perspective : PerspectiveConfig, optional
        4-point perspective transform (overrides calibration if set).
    green_max : float
        Upper bound for green (safe) speed zone (km/h).
    yellow_max : float
        Upper bound for yellow (warning) speed zone (km/h).
        Above this is red (over-speed).
    """

    def __init__(
        self,
        calibration: Optional[CalibrationConfig] = None,
        perspective: Optional[PerspectiveConfig] = None,
        green_max: float = 40.0,
        yellow_max: float = 70.0,
    ):
        self.calibration = calibration or CalibrationConfig()
        self.perspective = perspective
        self.green_max = green_max
        self.yellow_max = yellow_max

        # Per-track EMA state: track_id -> (ema_speed, last_timestamp, last_pos)
        self._track_state: Dict[int, dict] = {}

        # Cached homography matrix
        self._homography: Optional[np.ndarray] = None
        if perspective is not None:
            self._homography = perspective.get_homography()

        logger.info(
            f"SpeedEstimator ready. Zones: Green<=  {green_max}, "
            f"Yellow<={yellow_max}, Red=above"
        )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def estimate(
        self,
        track_id: int,
        position_history: deque,
    ) -> Tuple[float, str]:
        """
        Compute speed for a vehicle given its position history.

        Parameters
        ----------
        track_id : int
        position_history : deque
            Deque of (cx, cy, timestamp_sec) tuples.

        Returns
        -------
        (speed_kmh, zone) : Tuple[float, str]
            Smoothed speed in km/h and zone label ('green'/'yellow'/'red').
        """
        if len(position_history) < 2:
            return 0.0, "green"

        # Use last two positions for incremental speed
        prev_cx, prev_cy, prev_t = position_history[-2]
        curr_cx, curr_cy, curr_t = position_history[-1]

        dt = curr_t - prev_t
        if dt <= 0:
            return self._get_cached_speed(track_id)

        # Pixel displacement
        if self._homography is not None:
            speed_kmh = self._speed_with_perspective(
                prev_cx, prev_cy, curr_cx, curr_cy, dt
            )
        else:
            speed_kmh = self._speed_with_calibration(
                prev_cx, prev_cy, curr_cx, curr_cy, dt
            )

        # Filter out noise (tiny movements) and physical outliers
        pixel_dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
        if pixel_dist < MIN_MOTION_PIXELS:
            speed_kmh = 0.0

        speed_kmh = min(speed_kmh, MAX_CREDIBLE_SPEED_KMH)
        speed_kmh = max(speed_kmh, 0.0)

        # Exponential moving average smoothing
        speed_kmh = self._apply_ema(track_id, speed_kmh)

        zone = self._classify_zone(speed_kmh)
        return speed_kmh, zone

    def estimate_batch(
        self,
        vehicles,   # List[TrackedVehicle]
    ) -> Dict[int, Tuple[float, str]]:
        """Estimate speed for a list of tracked vehicles. Returns {track_id: (speed, zone)}."""
        results = {}
        for vehicle in vehicles:
            speed, zone = self.estimate(vehicle.track_id, vehicle.position_history)
            results[vehicle.track_id] = (speed, zone)
        return results

    def update_zones(self, green_max: float, yellow_max: float):
        """Dynamically update speed zone thresholds."""
        self.green_max = green_max
        self.yellow_max = yellow_max

    def update_calibration(self, config: CalibrationConfig):
        """Replace calibration config."""
        self.calibration = config

    def update_perspective(self, config: PerspectiveConfig):
        """Replace perspective config and recompute homography."""
        self.perspective = config
        self._homography = config.get_homography()

    def remove_track(self, track_id: int):
        """Clean up state for a track that has ended."""
        self._track_state.pop(track_id, None)

    def reset(self):
        """Clear all track states."""
        self._track_state.clear()

    # ─────────────────────────────────────────
    # Speed calculation methods
    # ─────────────────────────────────────────

    def _speed_with_calibration(
        self,
        px1: float, py1: float,
        px2: float, py2: float,
        dt: float,
    ) -> float:
        """
        Speed using linear pixel-per-meter calibration.

        pixel_dist (px) / ppm (px/m) = real_dist (m)
        speed (km/h) = real_dist / dt * 3.6
        """
        ppm = self.calibration.pixels_per_meter
        if ppm <= 0:
            return 0.0

        pixel_dist = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
        real_dist_m = pixel_dist / ppm
        speed_ms = real_dist_m / dt
        return speed_ms * 3.6

    def _speed_with_perspective(
        self,
        px1: float, py1: float,
        px2: float, py2: float,
        dt: float,
    ) -> float:
        """
        Speed using perspective-corrected bird's-eye coordinates.
        """
        if self._homography is None:
            return self._speed_with_calibration(px1, py1, px2, py2, dt)

        # Transform both points to bird's-eye plane
        p1 = self._transform_point(px1, py1)
        p2 = self._transform_point(px2, py2)

        if p1 is None or p2 is None:
            return 0.0

        # Distance in bird's-eye pixels
        bev_dist_px = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # Convert to meters
        ppm = self.perspective.dst_px_per_meter if self.perspective else 20.0
        real_dist_m = bev_dist_px / max(ppm, 1e-6)

        speed_ms = real_dist_m / dt
        return speed_ms * 3.6

    def _transform_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Apply homography to a single point."""
        try:
            pt = np.array([[[px, py]]], dtype=np.float32)
            result = cv2.perspectiveTransform(pt, self._homography)
            return float(result[0][0][0]), float(result[0][0][1])
        except Exception:
            return None

    # ─────────────────────────────────────────
    # EMA smoothing
    # ─────────────────────────────────────────

    def _apply_ema(self, track_id: int, raw_speed: float) -> float:
        if track_id not in self._track_state:
            self._track_state[track_id] = {"ema": raw_speed}
            return raw_speed

        prev_ema = self._track_state[track_id]["ema"]
        new_ema = EMA_ALPHA * raw_speed + (1 - EMA_ALPHA) * prev_ema
        self._track_state[track_id]["ema"] = new_ema
        return new_ema

    def _get_cached_speed(self, track_id: int) -> Tuple[float, str]:
        cached = self._track_state.get(track_id, {}).get("ema", 0.0)
        return cached, self._classify_zone(cached)

    def _classify_zone(self, speed_kmh: float) -> str:
        if speed_kmh <= self.green_max:
            return "green"
        elif speed_kmh <= self.yellow_max:
            return "yellow"
        else:
            return "red"


# ─────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────

ZONE_COLORS_BGR = {
    "green":  (34, 197, 94),     # BGR
    "yellow": (8, 179, 234),
    "red":    (68, 68, 239),
}

ZONE_COLORS_RGB = {
    "green":  (34, 197, 94),
    "yellow": (234, 179, 8),
    "red":    (239, 68,  68),
}


def draw_vehicle_annotations(
    frame: np.ndarray,
    vehicles,   # List[TrackedVehicle]
    draw_trail: bool = True,
    trail_length: int = 20,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, speed, and motion trails on frame.

    Returns annotated frame (BGR).
    """
    annotated = frame.copy()

    for vehicle in vehicles:
        if vehicle.current_bbox is None:
            continue

        x1, y1, x2, y2 = vehicle.current_bbox
        zone = vehicle.speed_zone
        speed = vehicle.current_speed_kmh

        # Zone color (BGR)
        zone_bgr = _rgb_to_bgr(ZONE_COLORS_RGB[zone])
        vehicle_bgr = _rgb_to_bgr(vehicle.color_rgb)

        # Draw trail
        if draw_trail and len(vehicle.position_history) >= 2:
            history = list(vehicle.position_history)[-trail_length:]
            for i in range(1, len(history)):
                alpha = i / len(history)
                trail_color = tuple(int(c * alpha) for c in zone_bgr)
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]),   int(history[i][1]))
                cv2.line(annotated, pt1, pt2, trail_color, 2)

        # Bounding box - colored by zone
        cv2.rectangle(annotated, (x1, y1), (x2, y2), zone_bgr, 2)

        # Label background
        label_top = f"ID:{vehicle.track_id} {vehicle.class_name}"
        label_bot = f"{speed:.1f} km/h"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness  = 1

        (tw, th), _ = cv2.getTextSize(label_top, font, font_scale, thickness)
        (bw, bh), _ = cv2.getTextSize(label_bot, font, font_scale, thickness)
        box_w = max(tw, bw) + 8
        box_h = th + bh + 12

        # Draw filled label background
        label_y = max(y1 - box_h - 2, 0)
        cv2.rectangle(
            annotated,
            (x1, label_y),
            (x1 + box_w, label_y + box_h),
            zone_bgr,
            -1,
        )

        # Text
        cv2.putText(
            annotated, label_top,
            (x1 + 4, label_y + th + 2),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
        cv2.putText(
            annotated, label_bot,
            (x1 + 4, label_y + th + bh + 8),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    return annotated


def draw_counting_line(
    frame: np.ndarray,
    line_y: Optional[int] = None,
    line_x: Optional[int] = None,
    color_bgr: Tuple = (255, 255, 0),
    label: str = "COUNT LINE",
) -> np.ndarray:
    """Draw virtual counting line on frame."""
    h, w = frame.shape[:2]
    if line_y is not None:
        cv2.line(frame, (0, line_y), (w, line_y), color_bgr, 2)
        cv2.putText(
            frame, label,
            (10, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA
        )
    if line_x is not None:
        cv2.line(frame, (line_x, 0), (line_x, h), color_bgr, 2)
        cv2.putText(
            frame, label,
            (line_x + 6, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA
        )
    return frame


def draw_calibration_line(
    frame: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    distance_m: float,
) -> np.ndarray:
    """Draw calibration reference segment on frame."""
    cv2.line(frame, p1, p2, (0, 255, 255), 2)
    cv2.circle(frame, p1, 5, (0, 255, 255), -1)
    cv2.circle(frame, p2, 5, (0, 255, 255), -1)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    cv2.putText(
        frame, f"{distance_m:.1f}m",
        (mid[0] + 6, mid[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA
    )
    return frame


def draw_roi(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],    # x, y, w, h
    color: Tuple = (0, 200, 255),
) -> np.ndarray:
    """Draw region of interest rectangle."""
    x, y, w, h = roi
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame, "ROI",
        (x + 4, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
    )
    return frame


def _rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])
