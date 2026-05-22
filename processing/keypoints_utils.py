# processing/keypoints_utils.py — helpers for working with keypoint dicts

from __future__ import annotations

import math
from typing import Optional
import math


def get_point(
    keypoints: dict[str, Optional[tuple[float, float]]],
    name: str,
) -> Optional[tuple[float, float]]:
    """Return a keypoint by name, or None if missing / low-confidence."""
    return keypoints.get(name)


def all_present(
    keypoints: dict[str, Optional[tuple[float, float]]],
    *names: str,
) -> bool:
    """Return True only when every requested keypoint has a valid position."""
    return all(keypoints.get(n) is not None for n in names)


def midpoint(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Return the pixel midpoint between two (x, y) points."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def body_orientation(
    keypoints: dict[str, Optional[tuple[float, float]]],
) -> Optional[str]:
    """
    Classify the person's gross body orientation as 'vertical' or 'horizontal'.

    Uses the vector from the hip midpoint to the shoulder midpoint.
    When this vector is more vertical than horizontal the person is upright
    (standing / squatting); when more horizontal they are prone (push-up).

    Returns None when the required keypoints are unavailable.
    """
    hips = _pair_or_single(keypoints, "left_hip", "right_hip")
    shoulders = _pair_or_single(keypoints, "left_shoulder", "right_shoulder")

    if hips is None or shoulders is None:
        return None

    dx = shoulders[0] - hips[0]
    dy = shoulders[1] - hips[1]   # positive y is downward in image coords

    torso_angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return "vertical" if torso_angle < 45 else "horizontal"


def _pair_or_single(
    keypoints: dict[str, Optional[tuple[float, float]]],
    left: str,
    right: str,
) -> Optional[tuple[float, float]]:
    """Return the midpoint of two keypoints, or whichever single one exists."""
    l = keypoints.get(left)
    r = keypoints.get(right)
    if l is not None and r is not None:
        return midpoint(l, r)
    return l if l is not None else r
