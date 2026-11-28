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
type TimedFrame = tuple[float, MatLike]


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


def _presentation_frames(container, stream, strict_length: bool) -> Iterator[TimedFrame]:
    origin = None
    previous = -1.0
    read = 0
    for frame in container.decode(stream):
        if frame.pts is None or frame.time_base is None:
            raise ValueError(f"frame {read} has no presentation timestamp")
        pts = frame.pts * frame.time_base
        if origin is None:
            origin = pts
        timestamp_s = float(pts - origin)
        if not isfinite(timestamp_s) or timestamp_s <= previous:
            raise ValueError(f"frame {read} presentation timestamp is not strictly increasing")
        yield timestamp_s, frame.to_ndarray(format="bgr24")
        previous = timestamp_s
        read += 1
    if read == 0:
        raise ValueError("video has no decodable frames")
    if strict_length and stream.frames > 0 and read < stream.frames:
        raise ValueError(f"video ended after {read} frames, expected {stream.frames}")


@contextmanager
def timed_video(
    path: Path, *, timestamp_source: str = "fps", strict_length: bool = False
) -> Iterator[tuple[float, Iterator[TimedFrame]]]:
    """Yield nominal FPS and BGR frames timestamped relative to the first frame."""
    if timestamp_source == "fps":
        with open_video(path) as (capture, fps):
            frames = decoded_frames(capture, strict_length=strict_length)
            yield fps, ((index / fps, frame) for index, frame in enumerate(frames))
        return
    if timestamp_source != "pts":
        raise ValueError(f"unknown timestamp source: {timestamp_source}")
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")
    try:
        import av
    except ImportError as error:
        raise ValueError(
            "presentation timestamps require PyAV (requirements-video.txt)"
        ) from error
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"no video stream in {path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            if not isfinite(fps) or fps <= 0:
                raise ValueError("the video must report a positive nominal frame rate")
            yield fps, _presentation_frames(container, stream, strict_length)
    except av.FFmpegError as error:
        raise ValueError(f"cannot decode video {path}: {error}") from error


def track_video(
    path: Path,
    detect: FrameDetector,
    max_gap_s: float,
    max_distance_px: float,
    *,
    tracker_name: str = "point",
    confidence: float = 0.35,
    strict_length: bool = False,
    timestamp_source: str = "fps",
) -> tuple[Track, ...]:
    with timed_video(path, timestamp_source=timestamp_source, strict_length=strict_length) as (
        fps,
        frames,
    ):
        tracker = make_tracker(tracker_name, fps, max_gap_s, max_distance_px, confidence)
        for timestamp_s, frame in frames:
            tracker.update_frame(timestamp_s, detect(frame))
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
    timestamp_source: str = "fps",
) -> Iterator[Event]:
    state = StreamingEvents(scenario)
    with timed_video(path, timestamp_source=timestamp_source, strict_length=strict_length) as (
        fps,
        frames,
    ):
        tracker = make_tracker(
            tracker_name,
            fps,
            scenario.thresholds.max_gap_s,
            max_distance_px,
            confidence,
            keep_history=False,
        )
        for time, frame in frames:
            yield from state.update(time, tracker.update_frame(time, detect(frame)))
