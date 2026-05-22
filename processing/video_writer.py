# processing/video_writer.py — writes annotated frames to an MP4 file

from __future__ import annotations

import cv2
import numpy as np


class VideoWriter:
    """Wraps cv2.VideoWriter with a context manager interface."""

    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]) -> None:
        """
        Args:
            output_path: destination .mp4 path
            fps:         frames-per-second (match source video)
            frame_size:  (width, height) of each frame
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    # ── context manager ────────────────────────────────────────────────────
    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
        print(f"[✓] Video saved to: {self.output_path}")
