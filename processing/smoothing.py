# processing/smoothing.py — Exponential Moving Average keypoint smoother

from __future__ import annotations

from typing import Optional
import config


class KeypointSmoother:
    """
    Applies per-keypoint Exponential Moving Average (EMA):

        S_t = alpha * X_t + (1 - alpha) * S_(t-1)

    A lower alpha yields heavier smoothing; higher alpha tracks faster motion.
    """

    def __init__(self, alpha: float = config.EMA_ALPHA) -> None:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: dict[str, Optional[tuple[float, float]]] = {}

    def smooth(
        self,
        current_keypoints: dict[str, Optional[tuple[float, float]]],
    ) -> dict[str, Optional[tuple[float, float]]]:
        """
        Blend current detections with the running EMA state.

        Keypoints that are None this frame keep the last known smoothed value
        (temporal carry-forward), so downstream code has a better chance of
        computing valid angles even when a keypoint flickers.
        """
        smoothed: dict[str, Optional[tuple[float, float]]] = {}

        for name, current in current_keypoints.items():
            prev = self._state.get(name)

            if current is None:
                # carry forward the last known position unchanged
                smoothed[name] = prev
            elif prev is None:
                # first valid observation — initialise with raw detection
                smoothed[name] = current
            else:
                # EMA blend
                sx = self.alpha * current[0] + (1 - self.alpha) * prev[0]
                sy = self.alpha * current[1] + (1 - self.alpha) * prev[1]
                smoothed[name] = (sx, sy)

        self._state = smoothed
        return smoothed
