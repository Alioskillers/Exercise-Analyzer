# analysis/pushup_analyzer.py — push-up repetition counting and form assessment

from __future__ import annotations

from typing import Optional

import config
from analysis.angles import calculate_angle
from processing.keypoints_utils import all_present


class PushUpAnalyzer:
    """
    Detects push-up repetitions and evaluates form using elbow angles and
    body-line alignment (shoulder–hip–ankle).

    State machine:
        "up"   — arms extended (elbow angle ≥ PUSHUP_UP_ANGLE)
        "down" — chest near floor (elbow angle ≤ PUSHUP_DOWN_ANGLE)

    A rep is counted on the down → up transition.
    Good form requires elbow angle < PUSHUP_DOWN_ANGLE AND a straight body
    line (shoulder–hip–ankle within BODY_LINE_TOLERANCE degrees of 180°).
    """

    # consecutive frames that must agree before a state flip is accepted
    _CONFIRM_FRAMES = 1

    def __init__(self) -> None:
        self.state: str = "up"
        self.rep_count: int = 0
        self.good_form_count: int = 0
        self.current_angle: Optional[float] = None    # exposed for visualisation
        self.current_body_straight: bool = False      # exposed for visualisation
        self._reached_bottom: bool = False
        self._body_straight_at_bottom: bool = False
        self._confirm_counter: int = 0   # frames consistently in new state

    def update(
        self,
        keypoints: dict[str, Optional[tuple[float, float]]],
    ) -> None:
        """Process one frame's smoothed keypoints and update rep state."""
        elbow_angle = self._compute_elbow_angle(keypoints)
        self.current_angle = elbow_angle
        self.current_body_straight = self._check_body_line(keypoints)
        if elbow_angle is None:
            return

        # ── bottom tracking ────────────────────────────────────────────────
        if elbow_angle < config.PUSHUP_DOWN_ANGLE:
            self._reached_bottom = True
            self._body_straight_at_bottom = self.current_body_straight

        # ── state transitions with frame-confirmation hysteresis ──────────
        # Require _CONFIRM_FRAMES consecutive frames in the new zone before
        # flipping state; prevents a single noisy detection from counting a rep.
        if self.state == "up":
            if elbow_angle < config.PUSHUP_DOWN_ANGLE:
                self._confirm_counter += 1
                if self._confirm_counter >= self._CONFIRM_FRAMES:
                    self.state = "down"
                    self._confirm_counter = 0
            else:
                self._confirm_counter = 0

        else:  # state == "down"
            if elbow_angle > config.PUSHUP_UP_ANGLE:
                self._confirm_counter += 1
                if self._confirm_counter >= self._CONFIRM_FRAMES:
                    self.state = "up"
                    self.rep_count += 1
                    if self._reached_bottom and self._body_straight_at_bottom:
                        self.good_form_count += 1
                    self._reached_bottom = False
                    self._body_straight_at_bottom = False
                    self._confirm_counter = 0
            else:
                self._confirm_counter = 0

    # ── private helpers ────────────────────────────────────────────────────
    def _compute_elbow_angle(
        self,
        keypoints: dict[str, Optional[tuple[float, float]]],
    ) -> Optional[float]:
        """
        Return elbow angle (shoulder–elbow–wrist).

        Averages both sides when both are visible; falls back to whichever
        single side is available.  This prevents None returns when one arm
        is occluded (common in side-view recordings).
        """
        left_angle = right_angle = None

        if all_present(keypoints, "left_shoulder", "left_elbow", "left_wrist"):
            left_angle = calculate_angle(
                keypoints["left_shoulder"],  # type: ignore[arg-type]
                keypoints["left_elbow"],     # type: ignore[arg-type]
                keypoints["left_wrist"],     # type: ignore[arg-type]
            )

        if all_present(keypoints, "right_shoulder", "right_elbow", "right_wrist"):
            right_angle = calculate_angle(
                keypoints["right_shoulder"],  # type: ignore[arg-type]
                keypoints["right_elbow"],     # type: ignore[arg-type]
                keypoints["right_wrist"],     # type: ignore[arg-type]
            )

        # use both when available, otherwise fall back to whichever side exists
        if left_angle is not None and right_angle is not None:
            return (left_angle + right_angle) / 2
        return left_angle if left_angle is not None else right_angle

    def _check_body_line(
        self,
        keypoints: dict[str, Optional[tuple[float, float]]],
    ) -> bool:
        """
        Check that shoulder–hip–ankle forms roughly a straight line.

        A straight plank position gives ≈ 180° at the hip; deviations beyond
        BODY_LINE_TOLERANCE indicate sagging hips or piked hips.
        """
        left_ok = right_ok = False

        if all_present(keypoints, "left_shoulder", "left_hip", "left_ankle"):
            angle = calculate_angle(
                keypoints["left_shoulder"],  # type: ignore[arg-type]
                keypoints["left_hip"],       # type: ignore[arg-type]
                keypoints["left_ankle"],     # type: ignore[arg-type]
            )
            left_ok = abs(180 - angle) <= config.BODY_LINE_TOLERANCE

        if all_present(keypoints, "right_shoulder", "right_hip", "right_ankle"):
            angle = calculate_angle(
                keypoints["right_shoulder"],  # type: ignore[arg-type]
                keypoints["right_hip"],       # type: ignore[arg-type]
                keypoints["right_ankle"],     # type: ignore[arg-type]
            )
            right_ok = abs(180 - angle) <= config.BODY_LINE_TOLERANCE

        # accept if at least one side is measurable and within tolerance
        return left_ok or right_ok

    # ── summary ────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "total_reps": self.rep_count,
            "good_form_reps": self.good_form_count,
        }
