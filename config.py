# config.py — central configuration for the Exercise Analysis System

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "yolov8s-pose.pt"   # YOLOv8 small pose model
CONFIDENCE_THRESHOLD = 0.5       # minimum keypoint confidence to use

# ── Smoothing ─────────────────────────────────────────────────────────────────
EMA_ALPHA = 0.3   # EMA smoothing factor (0 = heavy smooth, 1 = no smooth)

# ── Squat thresholds (knee angle, degrees) ────────────────────────────────────
SQUAT_UP_ANGLE       = 160   # knee angle to register returning to standing
SQUAT_DOWN_ANGLE     = 120   # knee angle to register any squat attempt (counts as a rep)
SQUAT_GOOD_FORM_ANGLE = 90   # knee angle required for full depth (good form)

# ── Push-up thresholds (elbow angle, degrees) ────────────────────────────────
# Generous ranges accommodate real-world form and side-view camera angles.
# Entering "down" phase:  elbow angle drops below PUSHUP_DOWN_ANGLE
# Entering "up"  phase:  elbow angle rises above PUSHUP_UP_ANGLE
PUSHUP_UP_ANGLE   = 140    # elbow angle at top position  (was 160 — too strict)
PUSHUP_DOWN_ANGLE = 110    # elbow angle at bottom position (was 90 — too strict)

# ── Body alignment (push-up form check) ──────────────────────────────────────
BODY_LINE_TOLERANCE = 20   # max degrees deviation from straight body line

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_PATH         = "output/results.json"
OUTPUT_VIDEO_SQUATS = "output/result_squats.mp4"
OUTPUT_VIDEO_PUSHUP = "output/result_pushup.mp4"

# ── Debug / visualization ─────────────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    # torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # face
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]
