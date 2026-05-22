# models/pose_model.py — YOLOv8 pose estimation wrapper

from __future__ import annotations

from typing import Optional
import numpy as np
from ultralytics import YOLO

import config

# COCO keypoint order used by YOLOv8 pose models
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


class PoseEstimator:
    """Wraps a YOLOv8 pose model and returns labelled keypoints per frame."""

    def __init__(self, model_path: str = config.MODEL_NAME) -> None:
        self.model = YOLO(model_path)

    def predict(self, frame: np.ndarray):
        """Run inference on a single BGR frame. Returns raw YOLO results."""
        results = self.model(frame, verbose=False)
        return results

    def extract_keypoints(
        self, results
    ) -> Optional[dict[str, tuple[float, float]]]:
        """
        Parse the first detected person's keypoints into a named dict.

        Returns None when no person is detected or confidence is too low.
        Each value is an (x, y) pixel tuple; low-confidence points become None.
        """
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                return None

            # Take the person with the highest detection confidence
            kp_data = result.keypoints.data[0]  # shape (17, 3): x, y, conf

            keypoints: dict[str, Optional[tuple[float, float]]] = {}
            for idx, name in enumerate(KEYPOINT_NAMES):
                x, y, conf = float(kp_data[idx][0]), float(kp_data[idx][1]), float(kp_data[idx][2])
                if conf >= config.CONFIDENCE_THRESHOLD:
                    keypoints[name] = (x, y)
                else:
                    keypoints[name] = None   # mark as unreliable

            return keypoints

        return None
