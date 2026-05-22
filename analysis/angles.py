# analysis/angles.py — joint angle computation

from __future__ import annotations

import math
from typing import Optional


def calculate_angle(
    A: tuple[float, float],
    B: tuple[float, float],
    C: tuple[float, float],
) -> float:
    """
    Return the angle (degrees) at vertex B formed by rays B→A and B→C.

    Uses the dot-product formula:
        cos(θ) = (BA · BC) / (|BA| * |BC|)

    Returns 0.0 when either ray has zero length (degenerate points).
    """
    # vectors from B
    ba = (A[0] - B[0], A[1] - B[1])
    bc = (C[0] - B[0], C[1] - B[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    # clamp to [-1, 1] to guard against floating-point rounding past ±1
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def average_angle(angle_a: Optional[float], angle_b: Optional[float]) -> Optional[float]:
    """Average two angles; returns None when either is None."""
    if angle_a is None or angle_b is None:
        return None
    return (angle_a + angle_b) / 2
