# main.py — entry point for the Exercise Analysis System

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

import config
from models.pose_model import PoseEstimator
from processing.keypoints_utils import body_orientation
from processing.video_reader import VideoReader
from processing.video_writer import VideoWriter
from processing.smoothing import KeypointSmoother
from analysis.squat_analyzer import SquatAnalyzer
from analysis.pushup_analyzer import PushUpAnalyzer
from utils.json_output import build_batch_result, build_result, save_result
from utils.helpers import annotate_squat_frame, annotate_pushup_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise Analysis System — squats & push-ups via YOLOv8 pose"
    )
    parser.add_argument("--video",      nargs="+", required=True,        help="One or more input video files")
    parser.add_argument("--output",     default=config.OUTPUT_PATH,       help="JSON results path")
    parser.add_argument("--out-squats", default=config.OUTPUT_VIDEO_SQUATS, help="Squats output video")
    parser.add_argument("--out-pushup", default=config.OUTPUT_VIDEO_PUSHUP, help="Push-ups output video")
    parser.add_argument("--debug",      action="store_true",              help="Play back annotated output videos after processing")
    parser.add_argument("--live",       action="store_true",              help="Show annotated frames live while processing")
    return parser.parse_args()


def _derive_output_path(base_output: str, video_path: str, suffix: str) -> str:
    """Create a per-video output path when running in batch mode."""
    base = Path(base_output)
    stem = Path(video_path).stem
    return str(base.with_name(f"{base.stem}_{stem}_{suffix}{base.suffix}"))


def process_video(video_path: str, out_squats: str, out_pushup: str, live: bool = False) -> dict:
    """
    Run pose estimation and rep counting on the input video.
    Writes two annotated output MP4 files and returns a per-video summary.
    """
    Path(out_squats).parent.mkdir(parents=True, exist_ok=True)
    Path(out_pushup).parent.mkdir(parents=True, exist_ok=True)

    estimator = PoseEstimator(config.MODEL_NAME)
    smoother  = KeypointSmoother(alpha=config.EMA_ALPHA)
    squats    = SquatAnalyzer()
    pushups   = PushUpAnalyzer()

    import itertools

    with VideoReader(video_path) as reader:
        fps   = reader.fps or 30.0
        total = reader.total_frames

        frame_iter = reader.frames()
        try:
            _, probe = next(frame_iter)
        except StopIteration:
            print("[ERROR] Video has no frames.", file=sys.stderr)
            sys.exit(1)

        h, w = probe.shape[:2]

        with VideoWriter(out_squats, fps, (w, h)) as squat_writer, \
             VideoWriter(out_pushup, fps, (w, h)) as pushup_writer:

            for frame_idx, frame in itertools.chain([(0, probe)], frame_iter):
                if frame_idx % 50 == 0:
                    pct = (frame_idx / total * 100) if total > 0 else 0
                    print(f"    frame {frame_idx}/{total}  ({pct:.0f}%)", end="\r")

                results   = estimator.predict(frame)
                keypoints = estimator.extract_keypoints(results)

                if keypoints is None:
                    squat_writer.write(frame)
                    pushup_writer.write(frame)
                    continue

                keypoints   = smoother.smooth(keypoints)
                orientation = body_orientation(keypoints)

                if orientation == "vertical":
                    squats.update(keypoints)
                elif orientation == "horizontal":
                    pushups.update(keypoints)

                # Only annotate the relevant output video for the detected orientation.
                # This prevents the squat HUD/box appearing in the push-up video and
                # vice versa.
                squat_frame = frame.copy()
                pushup_frame = frame.copy()

                if orientation == "vertical":
                    squat_frame = annotate_squat_frame(
                        frame, keypoints,
                        rep_count=squats.rep_count,
                        good_form_count=squats.good_form_count,
                        knee_angle=squats.current_angle,
                        state=squats.state,
                    )
                else:
                    # leave squat_frame as the raw frame (no squat HUD)
                    pass

                if orientation == "horizontal":
                    pushup_frame = annotate_pushup_frame(
                        frame, keypoints,
                        rep_count=pushups.rep_count,
                        good_form_count=pushups.good_form_count,
                        elbow_angle=pushups.current_angle,
                        body_straight=pushups.current_body_straight,
                        state=pushups.state,
                    )
                else:
                    # leave pushup_frame as the raw frame (no push-up HUD)
                    pass

                squat_writer.write(squat_frame)
                pushup_writer.write(pushup_frame)

                # Optionally show a live side-by-side view while processing
                if live:
                    try:
                        display_h = min(h, 720)
                        scale = display_h / h if h > 0 else 1.0
                        display_w = int(w * scale)

                        # Show only a single frame window — the one relevant to the
                        # detected orientation. This avoids running two instances
                        # of the same video side-by-side.
                        if orientation == "vertical":
                            show_frame = squat_frame
                        elif orientation == "horizontal":
                            show_frame = pushup_frame
                        else:
                            show_frame = frame

                        show_resized = cv2.resize(show_frame, (display_w, display_h))
                        cv2.imshow("Live Analysis", show_resized)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("[!] Live display stopped by user.")
                            break
                    except Exception:
                        # don't let the display interrupt processing on unexpected errors
                        pass

    print()
    # NOTE: do not destroy live windows here so batch mode can keep the
    # display open across multiple videos. Window lifecycle is managed by
    # `main()` when `--live` is used.

    return {
        "video": video_path,
        "outputs": {
            "squats_video": out_squats,
            "pushups_video": out_pushup,
        },
        "summary": {
            "squats": squats.summary(),
            "pushups": pushups.summary(),
        },
    }


def play_debug(squat_path: str, pushup_path: str) -> None:
    """
    Play back both annotated videos in a single side-by-side window.
    Both videos were produced by the same counting pipeline, so the on-screen
    rep counters are guaranteed to match the JSON results.
    Press Q to quit early.
    """
    cap_s = cv2.VideoCapture(squat_path)
    cap_p = cv2.VideoCapture(pushup_path)

    if not cap_s.isOpened() or not cap_p.isOpened():
        print("[!] Could not open output videos for playback.", file=sys.stderr)
        return

    fps   = cap_s.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))
    src_w = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # scale each half to at most 720p height so the combined window fits most screens
    display_h = min(src_h, 720)
    scale     = display_h / src_h
    display_w = int(src_w * scale)

    print(f"\n[*] Playing back annotated videos — press Q to quit …")
    cv2.namedWindow("Exercise Analyzer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Exercise Analyzer", display_w * 2, display_h)

    while True:
        ok_s, frame_s = cap_s.read()
        ok_p, frame_p = cap_p.read()

        if not ok_s or not ok_p:
            break

        # resize each frame to a common display size
        frame_s = cv2.resize(frame_s, (display_w, display_h))
        frame_p = cv2.resize(frame_p, (display_w, display_h))

        # label dividers so the user knows which side is which
        cv2.putText(frame_s, "SQUATS",   (10, display_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(frame_p, "PUSH-UPS", (10, display_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 230), 2, cv2.LINE_AA)

        combined = cv2.hconcat([frame_s, frame_p])
        cv2.imshow("Exercise Analyzer", combined)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("[!] Playback stopped by user.")
            break

    cap_s.release()
    cap_p.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    video_paths = [str(Path(path)) for path in args.video]

    print("[*] Loading pose model …")
    print(f"[*] Processing {len(video_paths)} video(s)")

    # If requested, create the live analysis window once so it persists
    # across multiple videos in batch mode.
    if args.live:
        try:
            cv2.namedWindow("Live Analysis", cv2.WINDOW_NORMAL)
        except Exception:
            pass

    try:
        if len(video_paths) == 1:
            video = video_paths[0]
            print(f"[*] Processing video: {video}")
            run = process_video(
                video,
                args.out_squats,
                args.out_pushup,
                live=args.live,
            )
            result = build_result(run["summary"]["squats"], run["summary"]["pushups"])
        else:
            video_runs = []
            for index, video in enumerate(video_paths, start=1):
                print(f"[*] Processing video {index}/{len(video_paths)}: {video}")
                run = process_video(
                    video,
                    _derive_output_path(args.out_squats, video, "squats"),
                    _derive_output_path(args.out_pushup, video, "pushups"),
                    live=args.live,
                )
                video_runs.append(run)
            result = build_batch_result(video_runs)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    save_result(result, args.output)

    print("\n===== RESULTS =====")
    print(f"  Squats  — total: {result['summary']['squats']['total_reps']}"
          f"  |  good form: {result['summary']['squats']['good_form_reps']}")
    print(f"  Push-ups— total: {result['summary']['pushups']['total_reps']}"
          f"  |  good form: {result['summary']['pushups']['good_form_reps']}")
    print("===================")

    # debug: play back the saved annotated videos — same frames that produced
    # the JSON result, so the counter on screen is always accurate.
    if args.debug and len(video_paths) == 1:
        play_debug(args.out_squats, args.out_pushup)
    elif args.debug:
        print("[*] Debug playback is skipped in batch mode.")

    # destroy any live windows after the whole batch completes
    if args.live:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
