# analysis/squat_analyzer.py — squat repetition counting and form assessment

from __future__ import annotations

from typing import Optional

import config
from analysis.angles import calculate_angle
from processing.keypoints_utils import all_present


class SquatAnalyzer:
    """
    Detects squat repetitions and evaluates form using knee angles.

    State machine:
        "up"   — person is standing (knee angle ≥ SQUAT_UP_ANGLE)
        "down" — person has bent knees below SQUAT_DOWN_ANGLE (any squat attempt)

    A rep is counted on every down → up transition, regardless of depth.
    A rep is marked "good form" only when the knee angle also dropped below
    SQUAT_GOOD_FORM_ANGLE (full depth) during that rep.
    """

    def __init__(self) -> None:
        self.state: str = "up"
        self.rep_count: int = 0
        self.good_form_count: int = 0
        self.current_angle: Optional[float] = None   # exposed for visualisation
        self._reached_full_depth: bool = False        # crossed SQUAT_GOOD_FORM_ANGLE this rep

    def update(
        self,
        keypoints: dict[str, Optional[tuple[float, float]]],
    ) -> None:
        """Process one frame's smoothed keypoints and update rep state."""
        knee_angle = self._compute_knee_angle(keypoints)
        self.current_angle = knee_angle
        if knee_angle is None:
            return

        # track whether full depth was achieved during this rep
        if knee_angle < config.SQUAT_GOOD_FORM_ANGLE:
            self._reached_full_depth = True

        # ── state transitions ──────────────────────────────────────────────
        # Enter "down" on any meaningful bend (SQUAT_DOWN_ANGLE = 120°)
        if self.state == "up" and knee_angle < config.SQUAT_DOWN_ANGLE:
            self.state = "down"

        # Count the rep when the person stands back up
        elif self.state == "down" and knee_angle > config.SQUAT_UP_ANGLE:
            self.state = "up"
            self.rep_count += 1
            if self._reached_full_depth:
                self.good_form_count += 1
            self._reached_full_depth = False   # reset for next rep

    # ── private helpers ────────────────────────────────────────────────────
    def _compute_knee_angle(
        self,
        keypoints: dict[str, Optional[tuple[float, float]]],
    ) -> Optional[float]:
        """Average left and right knee angles (hip–knee–ankle), single-side fallback."""
        left_angle = right_angle = None

        if all_present(keypoints, "left_hip", "left_knee", "left_ankle"):
            left_angle = calculate_angle(
                keypoints["left_hip"],   # type: ignore[arg-type]
                keypoints["left_knee"],  # type: ignore[arg-type]
                keypoints["left_ankle"], # type: ignore[arg-type]
            )

        if all_present(keypoints, "right_hip", "right_knee", "right_ankle"):
            right_angle = calculate_angle(
                keypoints["right_hip"],   # type: ignore[arg-type]
                keypoints["right_knee"],  # type: ignore[arg-type]
                keypoints["right_ankle"], # type: ignore[arg-type]
            )

        if left_angle is not None and right_angle is not None:
            return (left_angle + right_angle) / 2
        return left_angle if left_angle is not None else right_angle

    # ── summary ────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "total_reps": self.rep_count,
            "good_form_reps": self.good_form_count,
        }
