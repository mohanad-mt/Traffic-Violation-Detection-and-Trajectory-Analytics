from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import isfinite
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .detector import Event
from .scenario import Scenario, Track
from .streaming import StreamingEvents
from .tracker import FrameDetections, make_tracker

type FrameDetector = Callable[[MatLike], FrameDetections]


def foreground_detector(background_path: Path, min_area_px: int) -> FrameDetector:
    """Detect foreground blobs in a fixed scene with a clean reference frame."""
    background = cv2.imdecode(np.fromfile(background_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError(f"cannot decode background image: {background_path}")
    kernel = np.ones((3, 3), dtype=np.uint8)

    def detect(frame: MatLike) -> FrameDetections:
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
        return FrameDetections(sorted(points))

    return detect


@contextmanager
def open_video(path: Path) -> Iterator[tuple[cv2.VideoCapture, float]]:
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("the video must report a positive frame rate")
        yield capture, fps
    finally:
        capture.release()


def decoded_frames(capture: cv2.VideoCapture, *, strict_length: bool = False) -> Iterator[MatLike]:
    read = 0
    expected = capture.get(cv2.CAP_PROP_FRAME_COUNT) if strict_length else 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        read += 1
        yield frame
    if read == 0:
        raise ValueError("video has no decodable frames")
    if expected > 0 and read < expected:
        raise ValueError(f"video ended after {read} frames, expected {expected:g}")


def track_video(
    path: Path,
    detect: FrameDetector,
    max_gap_s: float,
    max_distance_px: float,
    *,
    tracker_name: str = "point",
    confidence: float = 0.35,
    strict_length: bool = False,
) -> tuple[Track, ...]:
    with open_video(path) as (capture, fps):
        tracker = make_tracker(tracker_name, fps, max_gap_s, max_distance_px, confidence)
        for frame_index, frame in enumerate(decoded_frames(capture, strict_length=strict_length)):
            tracker.update_frame(frame_index / fps, detect(frame))
    return tracker.tracks()


def stream_video_events(
    path: Path,
    detect: FrameDetector,
    scenario: Scenario,
    max_distance_px: float,
    *,
    tracker_name: str = "point",
    confidence: float = 0.35,
    strict_length: bool = False,
) -> Iterator[Event]:
    state = StreamingEvents(scenario)
    with open_video(path) as (capture, fps):
        tracker = make_tracker(
            tracker_name,
            fps,
            scenario.thresholds.max_gap_s,
            max_distance_px,
            confidence,
            keep_history=False,
        )
        for index, frame in enumerate(decoded_frames(capture, strict_length=strict_length)):
            time = index / fps
            yield from state.update(time, tracker.update_frame(time, detect(frame)))
