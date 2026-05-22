"""processing/video_padder.py

Pad portrait videos with black bars on the left/right to produce a
landscape-oriented MP4 without rotating frames.

Usage: python -m processing.video_padder <input> <output> [--aspect 16:9]
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
from processing.video_reader import VideoReader
from processing.video_writer import VideoWriter


def parse_aspect(aspect: str) -> float:
    if ":" in aspect:
        a, b = aspect.split(":", 1)
        return float(a) / float(b)
    return float(aspect)


def pad_to_aspect(input_path: str, output_path: str, target_aspect: float = 16 / 9) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with VideoReader(input_path) as reader:
        fps = reader.fps or 30.0
        frames = reader.frames()

        # Read first frame to determine sizes and create writer
        try:
            idx, first = next(frames)
        except StopIteration:
            raise RuntimeError("Input video contains no frames")

        if not isinstance(first, np.ndarray):
            first = np.asarray(first)

        h, w = first.shape[:2]
        target_w = int(round(h * target_aspect))

        # Never shrink horizontally; if the source is already wider, keep its width
        if target_w <= w:
            target_w = w

        left = (target_w - w) // 2
        right = target_w - w - left

        # Prepare black canvas and place first frame
        def make_padded(frame: np.ndarray) -> np.ndarray:
            canvas = np.zeros((h, target_w, 3), dtype=frame.dtype)
            canvas[:, left:left + w] = frame
            return canvas

        with VideoWriter(output_path, fps, (target_w, h)) as writer:
            writer.write(make_padded(first))

            for idx, frame in frames:
                if not isinstance(frame, np.ndarray):
                    frame = np.asarray(frame)
                writer.write(make_padded(frame))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Pad video to landscape by adding black side bars")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--aspect", default="16:9", help="target aspect ratio (e.g. 16:9 or 1.777)")
    args = p.parse_args(argv)

    aspect = parse_aspect(args.aspect)
    print(f"[*] Padding {args.input} -> {args.output} with aspect {aspect:.3f}")
    pad_to_aspect(args.input, args.output, aspect)
    print("[✓] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
