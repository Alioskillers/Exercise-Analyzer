# utils/helpers.py — frame annotation utilities

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

import config

# ── colour palette ─────────────────────────────────────────────────────────────
GREEN  = (0, 230, 0)
YELLOW = (0, 220, 220)
RED    = (0, 60, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
CYAN   = (230, 230, 0)
BLUE   = (200, 150, 0)
MAGENTA = (200, 0, 200)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: dict[str, Optional[tuple[float, float]]],
    color: tuple[int, int, int] = GREEN,
    point_radius: int = 5,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw joint dots and limb lines onto a copy of *frame*."""
    vis = frame.copy()
    def _segment_color(s, e):
        # assign colors by body region heuristics
        if "shoulder" in s or "hip" in s or "shoulder" in e or "hip" in e:
            return CYAN
        if "elbow" in s or "wrist" in s or "elbow" in e or "wrist" in e:
            # left vs right tint
            if s.startswith("left_") or e.startswith("left_"):
                return GREEN
            return YELLOW
        if "knee" in s or "ankle" in s or "knee" in e or "ankle" in e:
            if s.startswith("left_") or e.startswith("left_"):
                return RED
            return BLUE
        if "nose" in s or "nose" in e:
            return MAGENTA
        return color

    # draw limb lines first for better layering, each with its region color
    for start_name, end_name in config.SKELETON_CONNECTIONS:
        a = keypoints.get(start_name)
        b = keypoints.get(end_name)
        if a is not None and b is not None:
            seg_color = _segment_color(start_name, end_name)
            cv2.line(
                vis,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                seg_color,
                line_thickness,
            )

    # draw joint dots with a per-joint color to make tracking clearer
    def _joint_color(name):
        if "left_" in name:
            if "hand" in name or "wrist" in name or "elbow" in name:
                return GREEN
            if "knee" in name or "ankle" in name or "hip" in name:
                return RED
            return CYAN
        if "right_" in name:
            if "hand" in name or "wrist" in name or "elbow" in name:
                return YELLOW
            if "knee" in name or "ankle" in name or "hip" in name:
                return BLUE
            return CYAN
        # neutral joints
        return MAGENTA

    for name, pt in keypoints.items():
        if pt is not None:
            jcol = _joint_color(name)
            cv2.circle(vis, (int(pt[0]), int(pt[1])), point_radius, jcol, -1)

    return vis


def _label_angle(
    frame: np.ndarray,
    point: tuple[float, float],
    angle: float,
    color: tuple[int, int, int] = YELLOW,
) -> None:
    """Draw an angle value (degrees) near a joint — mutates frame in-place."""
    x, y = int(point[0]), int(point[1])
    cv2.putText(
        frame,
        f"{angle:.0f}deg",
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BLACK,
        3,   # thick black outline for readability
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{angle:.0f}deg",
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def _hud_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    alpha: float = 0.55,
) -> None:
    """Draw a semi-transparent dark rectangle as HUD background — mutates frame."""
    # Create overlay on a copy, blend result back into frame via a temporary buffer
    # to avoid the in-place aliasing issue where src and dst share memory.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    np.copyto(frame, blended)


def annotate_squat_frame(
    frame: np.ndarray,
    keypoints: dict[str, Optional[tuple[float, float]]],
    rep_count: int,
    good_form_count: int,
    knee_angle: Optional[float],
    state: str,
) -> np.ndarray:
    """
    Return an annotated copy of *frame* for the squats output video.

    Draws:
    - Green skeleton overlay
    - Knee angle label at the knee joint
    - HUD in the top-left corner showing reps, good-form count, phase, and angle
    """
    vis = draw_skeleton(frame, keypoints)

    # ── angle label at each visible knee ──────────────────────────────────
    if knee_angle is not None:
        for side in ("left_knee", "right_knee"):
            pt = keypoints.get(side)
            if pt is not None:
                _label_angle(vis, pt, knee_angle)

    # ── HUD panel — draw background FIRST, then text on top ───────────────
    hud_height = 148 if knee_angle is not None else 122
    _hud_bar(vis, 8, 8, 265, hud_height)

    font        = cv2.FONT_HERSHEY_SIMPLEX
    state_color = GREEN if state == "up" else RED

    cv2.putText(vis, "SQUATS",                         (16, 34),  font, 0.75, CYAN,        2, cv2.LINE_AA)
    cv2.putText(vis, f"Reps      : {rep_count}",       (16, 60),  font, 0.6,  WHITE,       1, cv2.LINE_AA)
    cv2.putText(vis, f"Good form : {good_form_count}", (16, 82),  font, 0.6,  GREEN,       1, cv2.LINE_AA)
    cv2.putText(vis, f"Phase     : {state.upper()}",   (16, 104), font, 0.6,  state_color, 1, cv2.LINE_AA)
    if knee_angle is not None:
        cv2.putText(vis, f"Knee angle: {knee_angle:.0f}deg", (16, 126), font, 0.6, YELLOW, 1, cv2.LINE_AA)

    # large rep counter at top-right — hard to miss regardless of HUD rendering
    h, w = vis.shape[:2]
    label = str(rep_count)
    (tw, th), _ = cv2.getTextSize(label, font, 3.5, 6)
    tx = w - tw - 20
    cv2.putText(vis, label, (tx, 90),     font, 3.5, BLACK, 10, cv2.LINE_AA)  # shadow
    cv2.putText(vis, label, (tx, 90),     font, 3.5, CYAN,   6, cv2.LINE_AA)  # foreground

    return vis


def annotate_pushup_frame(
    frame: np.ndarray,
    keypoints: dict[str, Optional[tuple[float, float]]],
    rep_count: int,
    good_form_count: int,
    elbow_angle: Optional[float],
    body_straight: bool,
    state: str,
) -> np.ndarray:
    """
    Return an annotated copy of *frame* for the push-ups output video.

    Draws:
    - Green skeleton overlay
    - Elbow angle label at each visible elbow joint
    - HUD with reps, good-form count, body-line status, phase, and angle
    """
    vis = draw_skeleton(frame, keypoints)

    # ── angle label at each visible elbow ─────────────────────────────────
    if elbow_angle is not None:
        for side in ("left_elbow", "right_elbow"):
            pt = keypoints.get(side)
            if pt is not None:
                _label_angle(vis, pt, elbow_angle)

    # ── HUD panel — draw background FIRST, then text on top ───────────────
    hud_height = 170 if elbow_angle is not None else 144
    _hud_bar(vis, 8, 8, 275, hud_height)

    font        = cv2.FONT_HERSHEY_SIMPLEX
    state_color = GREEN if state == "up" else RED
    body_color  = GREEN if body_straight else RED
    body_label  = "STRAIGHT" if body_straight else "BENT"

    cv2.putText(vis, "PUSH-UPS",                        (16, 34),  font, 0.75, CYAN,        2, cv2.LINE_AA)
    cv2.putText(vis, f"Reps      : {rep_count}",        (16, 60),  font, 0.6,  WHITE,       1, cv2.LINE_AA)
    cv2.putText(vis, f"Good form : {good_form_count}",  (16, 82),  font, 0.6,  GREEN,       1, cv2.LINE_AA)
    cv2.putText(vis, f"Body line : {body_label}",       (16, 104), font, 0.6,  body_color,  1, cv2.LINE_AA)
    cv2.putText(vis, f"Phase     : {state.upper()}",    (16, 126), font, 0.6,  state_color, 1, cv2.LINE_AA)
    if elbow_angle is not None:
        cv2.putText(vis, f"Elbow angle: {elbow_angle:.0f}deg", (16, 148), font, 0.6, YELLOW, 1, cv2.LINE_AA)

    # large rep counter at top-right
    h, w = vis.shape[:2]
    label = str(rep_count)
    (tw, th), _ = cv2.getTextSize(label, font, 3.5, 6)
    tx = w - tw - 20
    cv2.putText(vis, label, (tx, 90), font, 3.5, BLACK, 10, cv2.LINE_AA)
    cv2.putText(vis, label, (tx, 90), font, 3.5, CYAN,   6, cv2.LINE_AA)

    return vis
