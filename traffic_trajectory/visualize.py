"""Render video and previously saved --include-tracks results as an annotated MP4."""

import argparse
import json
from collections import defaultdict, deque
from contextlib import contextmanager
from fractions import Fraction
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from zlib import crc32

import cv2
import numpy as np
from cv2.typing import MatLike

from .geometry import Point
from .scenario import Observation, Scenario, _parse_track, parse_scenario
from .video import timed_video

COLORS = ((252, 194, 90), (100, 225, 250), (96, 220, 138), (224, 157, 245), (151, 193, 255))


def _pixel(point: Point) -> tuple[int, int]:
    return round(point[0]), round(point[1])


def _text(frame: MatLike, value: str, point: tuple[int, int]) -> None:
    cv2.putText(frame, value, point, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        frame, value, point, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1, cv2.LINE_AA
    )


def _geometry(frame: MatLike, scenario: Scenario, timestamp_s: float) -> None:
    for zone in scenario.zones:
        lane = zone.lane
        vertices = [start for start, _ in lane.polygon.edges]
        cv2.polylines(
            frame,
            [np.array([_pixel(v) for v in vertices], np.int32)],
            True,
            (245, 213, 91),
            1,
            cv2.LINE_AA,
        )
        x, y = (sum(axis) / len(vertices) for axis in zip(*vertices, strict=True))
        dx, dy = lane.forward
        cv2.arrowedLine(
            frame,
            _pixel((x, y)),
            _pixel((x + 25 * dx, y + 25 * dy)),
            (245, 213, 91),
            1,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        if zone.stop_line:
            line = zone.stop_line
            red = any(start <= timestamp_s < end for start, end in line.red_intervals_s)
            first, second = map(_pixel, line.segment)
            color = (55, 65, 245) if red else (197, 197, 197)
            cv2.line(frame, first, second, color, 2, cv2.LINE_AA)
            _text(frame, "red" if red else "outside red interval", (first[0], first[1] - 5))


class Overlay:
    def __init__(self, predictions: dict, time_scale: float, title: str) -> None:
        self.title = title
        self.time_scale = time_scale
        if "tracks" not in predictions:
            raise ValueError("rendering requires predictions generated with --include-tracks")
        self.observations = defaultdict(list)
        for index, raw in enumerate(predictions["tracks"]):
            track = _parse_track(raw, f"tracks[{index}]")
            for sample in track.observations:
                self.observations[round(sample.timestamp_s * time_scale)].append(
                    (track.id, sample)
                )
        self.events = predictions["events"]
        self.tails: dict[str, deque[tuple[float, Point]]] = {}

    def draw(self, frame: MatLike, t: float, scenario: Scenario) -> MatLike:
        frame = frame.copy()
        _geometry(frame, scenario, t)
        for track_id, sample in self.observations.get(round(t * self.time_scale), []):
            self._draw_sample(frame, track_id, sample, t)
        self.tails = {key: tail for key, tail in self.tails.items() if t - tail[-1][0] <= 2}
        _text(frame, f"{self.title}  {t:.2f}s", (8, 17))
        recent = [event for event in self.events if 0 <= t - event["confirmed_s"] <= 3]
        for row, event in enumerate(recent[-3:]):
            kind = event["type"].removesuffix("_candidate").replace("_", " ")
            message = (
                f"{kind}  {event['track_id']}  "
                f"{event['start_s']:.2f}s onset / {event['confirmed_s']:.2f}s confirmed"
            )
            _text(frame, message, (8, frame.shape[0] - 10 - 15 * row))
        return frame

    def _draw_sample(self, frame: MatLike, key: str, sample: Observation, t: float) -> None:
        color = COLORS[crc32(key.encode()) % len(COLORS)]
        tail = self.tails.setdefault(key, deque())
        while tail and t - tail[0][0] > 2:
            tail.popleft()
        tail.append((t, sample.position))
        if len(tail) > 1:
            cv2.polylines(
                frame,
                [np.array([_pixel(p) for _, p in tail], np.int32)],
                False,
                color,
                1,
                cv2.LINE_AA,
            )
        anchor = _pixel(sample.position)
        cv2.circle(frame, anchor, 2, color, -1, cv2.LINE_AA)
        box = sample.box
        text = key.removeprefix("vehicle-")
        if box is not None:
            anchor = _pixel((box.x, box.y))
            second = _pixel((box.x + box.width, box.y + box.height))
            cv2.rectangle(frame, anchor, second, color, 1, cv2.LINE_AA)
            text = f"{text} {box.category or 'vehicle'} {box.score:.2f}"
        _text(frame, text, (max(0, anchor[0]), max(12, anchor[1] - 3)))


@contextmanager
def _video_writer(path: Path, fps: float, size: tuple[int, int], source: str):
    if source == "fps":
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        try:
            if not writer.isOpened():
                raise ValueError("could not open an MP4 video writer")
            yield lambda image, _time: writer.write(image)
        finally:
            writer.release()
        return

    import av

    try:
        with av.open(str(path), mode="w") as output:
            stream = output.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
            stream.width, stream.height = size
            stream.pix_fmt = "yuv420p" if all(length % 2 == 0 for length in size) else "yuv444p"
            time_base = Fraction(1, 1_000_000)
            stream.time_base = stream.codec_context.time_base = time_base

            def write(image: MatLike, time: float) -> None:
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts = round(time / time_base)
                frame.time_base = time_base
                output.mux(stream.encode(frame))

            yield write
            output.mux(stream.encode())
    except av.FFmpegError as error:
        raise ValueError(f"could not encode the timestamped overlay: {error}") from error


def render(
    video: Path, scenario: Scenario, predictions: list[tuple[str, dict]], destination: Path
) -> None:
    if destination.resolve() == video.resolve() or destination.suffix.lower() != ".mp4":
        raise ValueError("choose a separate .mp4 output path")
    sources = {data.get("timestamp_source", "fps") for _, data in predictions}
    if len(sources) != 1:
        raise ValueError("comparison predictions must use the same timestamp source")
    source = sources.pop()
    with NamedTemporaryFile(dir=destination.parent, suffix=".mp4", delete=False) as temp:
        writing = Path(temp.name)
    try:
        with timed_video(video, timestamp_source=source, strict_length=True) as (fps, frames):
            time_scale = 1_000_000 if source == "pts" else fps
            overlays = [Overlay(data, time_scale, title) for title, data in predictions]
            first = next(frames)
            height, width = first[1].shape[:2]
            with _video_writer(writing, fps, (width * len(overlays), height), source) as write:
                for time, frame in chain((first,), frames):
                    images = [overlay.draw(frame, time, scenario) for overlay in overlays]
                    write(images[0] if len(images) == 1 else np.hstack(images), time)
        writing.replace(destination)
    finally:
        writing.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Pipeline")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--compare-title", default="Comparison")
    args = parser.parse_args()

    def load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    try:
        scene = parse_scenario({**load(args.scene), "tracks": []})
        predictions = [(args.title, load(args.predictions))]
        if args.compare:
            predictions.append((args.compare_title, load(args.compare)))
        render(args.video, scene, predictions, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
