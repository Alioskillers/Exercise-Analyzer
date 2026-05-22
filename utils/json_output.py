# utils/json_output.py — structured JSON result generation

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def build_result(squat_summary: dict, pushup_summary: dict) -> dict:
    """Assemble the top-level result document with a unique video ID."""
    return {
        "video_id": str(uuid.uuid4()),
        "summary": {
            "squats": squat_summary,
            "pushups": pushup_summary,
        },
    }


def build_batch_result(video_runs: list[dict]) -> dict:
    """Assemble a batch result document for multiple processed videos."""
    squat_total = sum(run["summary"]["squats"]["total_reps"] for run in video_runs)
    squat_good = sum(run["summary"]["squats"]["good_form_reps"] for run in video_runs)
    pushup_total = sum(run["summary"]["pushups"]["total_reps"] for run in video_runs)
    pushup_good = sum(run["summary"]["pushups"]["good_form_reps"] for run in video_runs)

    return {
        "video_id": str(uuid.uuid4()),
        "mode": "batch",
        "video_runs": video_runs,
        "summary": {
            "squats": {
                "total_reps": squat_total,
                "good_form_reps": squat_good,
            },
            "pushups": {
                "total_reps": pushup_total,
                "good_form_reps": pushup_good,
            },
        },
    }


def save_result(result: dict, output_path: str) -> None:
    """Write the result dict to a JSON file, creating parent dirs as needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[✓] Results saved to: {os.path.abspath(output_path)}")
