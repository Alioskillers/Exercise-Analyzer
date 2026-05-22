# processing/video_rotator.py — rotate portrait videos to landscape

from __future__ import annotations

import cv2
import numpy as np
import sys
from pathlib import Path
from processing.video_reader import VideoReader
from processing.video_writer import VideoWriter


def rotate_video_90_clockwise(input_path: str, output_path: str) -> None:
    """
    Rotate a portrait video 90 degrees clockwise to landscape.
    
    Args:
        input_path: path to input video (portrait orientation)
        output_path: path to save rotated video (landscape orientation)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with VideoReader(input_path) as reader:
        fps = reader.fps or 30.0
        total = reader.total_frames
        
        # Collect all frames first to get dimensions
        all_frames = []
        for frame_idx, frame in reader.frames():
            all_frames.append(frame)
            if len(all_frames) == 1:
                # Get dimensions from first frame
                h, w = frame.shape[:2]
                new_w, new_h = h, w  # Swap dimensions after 90° rotation
                print(f"[*] Original size: {w}x{h}, rotated size: {new_w}x{new_h}")
        
        # Now write all frames with rotation
        with VideoWriter(output_path, fps, (new_w, new_h)) as writer:
            for frame_idx, frame in enumerate(all_frames):
                if frame_idx % 50 == 0:
                    pct = (frame_idx / len(all_frames) * 100) if all_frames else 0
                    print(f"    Writing frame {frame_idx}/{len(all_frames)}  ({pct:.0f}%)", end="\r")
                
                # Ensure frame is numpy array
                if not isinstance(frame, np.ndarray):
                    frame = np.asarray(frame)
                
                # Rotate 90 degrees clockwise: transpose + flip
                rotated = cv2.transpose(frame)
                rotated = cv2.flip(rotated, 1)  # flip horizontally
                writer.write(rotated)
        
        print()  # newline after progress


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m processing.video_rotator <input_video> <output_video>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    
    print(f"[*] Rotating {input_video} to landscape...")
    rotate_video_90_clockwise(input_video, output_video)
    print(f"[?] Done! Landscape video saved to: {output_video}")
