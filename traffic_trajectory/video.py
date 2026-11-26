from collections.abc import Callable
from math import isfinite
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .geometry import Point
from .scenario import Track
from .tracker import PointTracker

type FrameDetector = Callable[[MatLike], list[Point]]


def foreground_detector(background_path: Path, min_area_px: int) -> FrameDetector:
    """Detect foreground blobs in a fixed scene with a clean reference frame."""
    background = cv2.imdecode(np.fromfile(background_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError(f"cannot decode background image: {background_path}")
    kernel = np.ones((3, 3), dtype=np.uint8)

    def detect(frame: MatLike) -> list[Point]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != background.shape:
            raise ValueError("background image dimensions must match the video")
        difference = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(mask)
        points = [
            (x + (width - 1) / 2, float(y + height - 1))
            for x, y, width, height, area in boxes[1:]
            if area >= min_area_px
        ]
        return sorted(points)

    return detect


def track_video(
    path: Path, detect: FrameDetector, max_gap_s: float, max_distance_px: float
) -> tuple[Track, ...]:
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    tracker = PointTracker(max_gap_s, max_distance_px)
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("the video must report a positive frame rate")

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tracker.update(frame_index / fps, detect(frame))
            frame_index += 1
        if frame_index == 0:
            raise ValueError("video has no decodable frames")
    finally:
        capture.release()

    return tracker.tracks()
