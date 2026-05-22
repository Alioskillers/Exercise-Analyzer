# processing/video_reader.py — frame-by-frame video reading

from __future__ import annotations

from typing import Generator
import cv2
import numpy as np


class VideoReader:
    """Yields BGR frames from a video file one at a time."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self._cap: cv2.VideoCapture | None = None

    # ── context manager so the capture is always released ──────────────────
    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        # Honor rotation metadata when the backend supports it.
        # This helps preserve the intended aspect ratio on phone-recorded videos
        # that are stored with orientation tags instead of physically rotated frames.
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto is not None:
            try:
                self._cap.set(orientation_auto, 1)
            except Exception:
                pass
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()

    # ── public helpers ──────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame) pairs until the video ends."""
        if self._cap is None:
            raise RuntimeError("Use VideoReader inside a 'with' block.")

        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
