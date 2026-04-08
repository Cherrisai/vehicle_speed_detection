"""
tracking.py
===========
Multi-object vehicle tracking using ByteTrack (via supervision library).
Maintains track history, handles ID assignment, and feeds data to
speed estimation.

ByteTrack is preferred over DeepSORT because:
  - No re-ID model required (lighter weight)
  - Better handling of crowded scenes (Indian traffic)
  - Lower latency per frame
  - Maintained in the supervision library
"""

import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import logging

from detection import Detection, DetectionResult, DISPLAY_LABELS, CLASS_COLORS_RGB

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class TrackedVehicle:
    """
    Represents a single tracked vehicle with full history.
    """
    track_id: int
    class_id: int
    class_name: str
    color_rgb: Tuple[int, int, int]

    # Position history: list of (cx, cy, timestamp_sec)
    position_history: deque = field(default_factory=lambda: deque(maxlen=60))

    # Speed history (km/h): rolling buffer for smoothing
    speed_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Current state
    current_bbox: Optional[np.ndarray] = None
    current_speed_kmh: float = 0.0
    speed_zone: str = "green"
    confidence: float = 0.0

    # Line crossing tracking
    crossed_line: bool = False
    cross_direction: Optional[str] = None   # 'up' | 'down' | 'left' | 'right'

    # Lifecycle
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    is_active: bool = True
    frames_since_seen: int = 0

    @property
    def smoothed_speed(self) -> float:
        """Return exponentially smoothed speed."""
        if not self.speed_history:
            return 0.0
        weights = np.exp(np.linspace(-1, 0, len(self.speed_history)))
        weights /= weights.sum()
        return float(np.dot(weights, list(self.speed_history)))

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if not self.position_history:
            return None
        return self.position_history[-1][0], self.position_history[-1][1]


class VehicleTracker:
    """
    Multi-object tracker wrapping ByteTrack from the supervision library.

    Parameters
    ----------
    track_thresh : float
        Detection confidence threshold to initiate a new track.
    track_buffer : int
        Frames a track is kept alive after last detection.
    match_thresh : float
        IoU threshold for track-to-detection matching.
    frame_rate : int
        Video frame rate (used by ByteTrack internally).
    max_history : int
        Max position history frames kept per track.
    """

    def __init__(
        self,
        track_thresh: float = 0.35,
        track_buffer: int = 30,
        match_thresh: float = 0.85,
        frame_rate: int = 30,
        max_history: int = 60,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.max_history = max_history

        # Initialize ByteTracker
        self.byte_tracker = sv.ByteTracker(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )

        # Track store: track_id -> TrackedVehicle
        self.tracks: Dict[int, TrackedVehicle] = {}

        # Global stats
        self.total_unique_vehicles: int = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.crossed_line_counts: Dict[str, int] = defaultdict(int)
        self.frame_count: int = 0

        logger.info("VehicleTracker (ByteTrack) initialized.")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def update(
        self,
        detection_result: DetectionResult,
        timestamp_sec: float,
    ) -> List[TrackedVehicle]:
        """
        Update tracker with new frame detections.

        Parameters
        ----------
        detection_result : DetectionResult
            Output from VehicleDetector.detect().
        timestamp_sec : float
            Current frame timestamp in seconds.

        Returns
        -------
        List[TrackedVehicle]
            All currently active tracked vehicles.
        """
        self.frame_count += 1

        if not detection_result.detections:
            self._mark_all_unseen()
            return self._get_active_tracks()

        # Build supervision Detections object
        sv_detections = self._to_sv_detections(detection_result)

        # Run ByteTrack update
        tracked_sv = self.byte_tracker.update_with_detections(sv_detections)

        if len(tracked_sv) == 0:
            self._mark_all_unseen()
            return self._get_active_tracks()

        # Build a lookup from detection index to original Detection object
        # (sv tracked detections maintain source indices via tracker_id)
        det_map = {i: d for i, d in enumerate(detection_result.detections)}

        # Process tracked detections
        active_ids = set()
        for i in range(len(tracked_sv)):
            tracker_id = int(tracked_sv.tracker_id[i])
            bbox = tracked_sv.xyxy[i].astype(int)
            class_id = int(tracked_sv.class_id[i])
            conf = float(tracked_sv.confidence[i])

            active_ids.add(tracker_id)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if tracker_id not in self.tracks:
                # New track
                vehicle = TrackedVehicle(
                    track_id=tracker_id,
                    class_id=class_id,
                    class_name=DISPLAY_LABELS.get(class_id, "Vehicle"),
                    color_rgb=CLASS_COLORS_RGB.get(class_id, (180, 180, 180)),
                    first_seen_frame=self.frame_count,
                )
                self.tracks[tracker_id] = vehicle
                self.total_unique_vehicles += 1
                self.class_counts[vehicle.class_name] += 1
                logger.debug(f"New track: ID={tracker_id} class={vehicle.class_name}")
            else:
                vehicle = self.tracks[tracker_id]

            # Update vehicle state
            vehicle.current_bbox = bbox
            vehicle.confidence = conf
            vehicle.last_seen_frame = self.frame_count
            vehicle.frames_since_seen = 0
            vehicle.is_active = True
            vehicle.position_history.append((cx, cy, timestamp_sec))

        # Mark tracks not seen this frame
        for tid, vehicle in self.tracks.items():
            if tid not in active_ids:
                vehicle.frames_since_seen += 1
                if vehicle.frames_since_seen > self.track_buffer:
                    vehicle.is_active = False

        return self._get_active_tracks()

    def update_speed(
        self,
        track_id: int,
        speed_kmh: float,
        zone: str,
    ):
        """Called by speed estimator to push speed into a track."""
        if track_id in self.tracks:
            v = self.tracks[track_id]
            v.speed_history.append(speed_kmh)
            v.current_speed_kmh = v.smoothed_speed
            v.speed_zone = zone

    def check_line_crossing(
        self,
        line_y: Optional[int],
        line_x: Optional[int] = None,
        direction: str = "horizontal",
    ):
        """
        Detect which vehicles cross a virtual counting line this frame.

        Parameters
        ----------
        line_y : int, optional
            Y-coordinate of horizontal counting line.
        line_x : int, optional
            X-coordinate of vertical counting line.
        direction : str
            'horizontal' or 'vertical'.
        """
        for vehicle in self.tracks.values():
            if not vehicle.is_active or len(vehicle.position_history) < 2:
                continue
            if vehicle.crossed_line:
                continue

            prev = vehicle.position_history[-2]
            curr = vehicle.position_history[-1]

            if direction == "horizontal" and line_y is not None:
                if (prev[1] < line_y <= curr[1]) or (prev[1] > line_y >= curr[1]):
                    vehicle.crossed_line = True
                    vehicle.cross_direction = "down" if curr[1] > prev[1] else "up"
                    self.crossed_line_counts[vehicle.class_name] += 1

            elif direction == "vertical" and line_x is not None:
                if (prev[0] < line_x <= curr[0]) or (prev[0] > line_x >= curr[0]):
                    vehicle.crossed_line = True
                    vehicle.cross_direction = "right" if curr[0] > prev[0] else "left"
                    self.crossed_line_counts[vehicle.class_name] += 1

    def reset(self):
        """Full reset of tracker state."""
        self.byte_tracker = sv.ByteTracker(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate,
        )
        self.tracks.clear()
        self.total_unique_vehicles = 0
        self.class_counts.clear()
        self.crossed_line_counts.clear()
        self.frame_count = 0
        logger.info("Tracker reset.")

    def get_stats(self) -> dict:
        """Return current tracking statistics."""
        active = self._get_active_tracks()
        speeds = [v.current_speed_kmh for v in active if v.current_speed_kmh > 0]

        fastest = None
        if speeds:
            fastest_vehicle = max(active, key=lambda v: v.current_speed_kmh)
            fastest = {
                "speed": fastest_vehicle.current_speed_kmh,
                "class": fastest_vehicle.class_name,
                "track_id": fastest_vehicle.track_id,
            }

        return {
            "active_count": len(active),
            "total_unique": self.total_unique_vehicles,
            "class_counts": dict(self.class_counts),
            "crossed_counts": dict(self.crossed_line_counts),
            "current_max_speed": max(speeds, default=0.0),
            "fastest_ever": fastest,
            "avg_speed": np.mean(speeds) if speeds else 0.0,
        }

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _to_sv_detections(self, result: DetectionResult) -> sv.Detections:
        """Convert DetectionResult to supervision Detections format."""
        if not result.detections:
            return sv.Detections.empty()

        bboxes   = np.array([d.bbox for d in result.detections], dtype=float)
        confs    = np.array([d.confidence for d in result.detections], dtype=float)
        cls_ids  = np.array([d.class_id for d in result.detections], dtype=int)

        return sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=cls_ids,
        )

    def _get_active_tracks(self) -> List[TrackedVehicle]:
        return [v for v in self.tracks.values() if v.is_active]

    def _mark_all_unseen(self):
        for vehicle in self.tracks.values():
            vehicle.frames_since_seen += 1
            if vehicle.frames_since_seen > self.track_buffer:
                vehicle.is_active = False
