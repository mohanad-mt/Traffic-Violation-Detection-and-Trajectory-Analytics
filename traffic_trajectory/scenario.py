from dataclasses import dataclass, fields
from itertools import pairwise
from math import hypot, isfinite
from typing import Any

from .geometry import ConvexPolygon, Point


@dataclass(frozen=True)
class Lane:
    id: str
    polygon: ConvexPolygon
    forward: Point


@dataclass(frozen=True)
class Observation:
    timestamp_s: float
    position: Point


@dataclass(frozen=True)
class Track:
    id: str
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Thresholds:
    min_reverse_distance_px: float = 35.0
    min_duration_s: float = 1.0
    max_gap_s: float = 0.75
    min_step_px: float = 1.0
    min_heading_cosine: float = 0.7
    max_pause_s: float = 0.25
    stopped_duration_s: float = 3.0
    stopped_radius_px: float = 4.0
    red_line_margin_px: float = 2.0


@dataclass(frozen=True)
class StopLine:
    segment: tuple[Point, Point]
    red_intervals_s: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Scenario:
    lane: Lane
    thresholds: Thresholds
    tracks: tuple[Track, ...]
    stop_line: StopLine | None = None


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _id(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _number(value: object, path: str) -> float:
    if type(value) in (int, float):
        try:
            number = float(value)
        except OverflowError:
            pass
        else:
            if isfinite(number):
                return number
    raise ValueError(f"{path} must be a finite number")


def _point(value: object, path: str) -> Point:
    coordinates = _array(value, path)
    if len(coordinates) != 2:
        raise ValueError(f"{path} must contain two coordinates")
    return _number(coordinates[0], f"{path}[0]"), _number(coordinates[1], f"{path}[1]")


def _parse_lane(value: object) -> Lane:
    lane = _object(value, "lane")
    lane_id = _id(lane.get("id"), "lane.id")
    vertices = tuple(
        _point(point, f"lane.polygon[{index}]")
        for index, point in enumerate(_array(lane.get("polygon"), "lane.polygon"))
    )
    x, y = _point(lane.get("forward"), "lane.forward")
    # Scale before taking the norm to allow any finite, nonzero direction vector.
    scale = max(abs(x), abs(y))
    if scale == 0:
        raise ValueError("lane.forward must be nonzero")
    x, y = x / scale, y / scale
    length = hypot(x, y)
    return Lane(lane_id, ConvexPolygon.from_vertices(vertices), (x / length, y / length))


def _parse_thresholds(value: object) -> Thresholds:
    config = _object(value, "thresholds")
    defaults = Thresholds()
    names = {field.name for field in fields(Thresholds)}
    unknown = config.keys() - names
    if unknown:
        raise ValueError(f"unknown thresholds: {', '.join(sorted(unknown))}")

    values = {
        name: _number(config.get(name, getattr(defaults, name)), f"thresholds.{name}")
        for name in sorted(names)
    }
    for name, number in values.items():
        if name in {"max_pause_s", "red_line_margin_px"}:
            if number < 0:
                raise ValueError(f"thresholds.{name} must be nonnegative")
        elif number <= 0:
            raise ValueError(f"thresholds.{name} must be positive")
    if values["min_heading_cosine"] > 1:
        raise ValueError("thresholds.min_heading_cosine must be at most 1")
    return Thresholds(**values)


def _parse_track(value: object, path: str) -> Track:
    track = _object(value, path)
    track_id = _id(track.get("id"), f"{path}.id")
    observations = []
    for index, raw in enumerate(_array(track.get("observations"), f"{path}.observations")):
        observation_path = f"{path}.observations[{index}]"
        observation = _object(raw, observation_path)
        timestamp = _number(observation.get("timestamp_s"), f"{observation_path}.timestamp_s")
        if timestamp < 0:
            raise ValueError(f"{observation_path}.timestamp_s must be nonnegative")
        point = _point(observation.get("position"), f"{observation_path}.position")
        observations.append(Observation(timestamp, point))

    if any(right.timestamp_s <= left.timestamp_s for left, right in pairwise(observations)):
        raise ValueError(f"{path}.observations timestamps must be strictly increasing")
    return Track(track_id, tuple(observations))


def _parse_stop_line(value: object, lane: Lane) -> StopLine:
    config = _object(value, "stop_line")
    segment = _array(config.get("segment"), "stop_line.segment")
    if len(segment) != 2:
        raise ValueError("stop_line.segment must contain two endpoints")
    start = _point(segment[0], "stop_line.segment[0]")
    end = _point(segment[1], "stop_line.segment[1]")
    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx * lane.forward[1] - dy * lane.forward[0] == 0:
        raise ValueError("stop_line.segment must cross the lane direction")

    intervals = tuple(
        _point(interval, f"stop_line.red_intervals_s[{index}]")
        for index, interval in enumerate(
            _array(config.get("red_intervals_s"), "stop_line.red_intervals_s")
        )
    )
    if any(not 0 <= begin < finish for begin, finish in intervals):
        raise ValueError("red intervals must have nonnegative start times and positive duration")
    if any(previous[1] > current[0] for previous, current in pairwise(intervals)):
        raise ValueError("red intervals must be ordered and non-overlapping")
    return StopLine((start, end), intervals)


def parse_scenario(value: object) -> Scenario:
    document = _object(value, "scenario")
    lane = _parse_lane(document.get("lane"))
    thresholds = _parse_thresholds(document.get("thresholds", {}))
    tracks = tuple(
        _parse_track(track, f"tracks[{index}]")
        for index, track in enumerate(_array(document.get("tracks"), "tracks"))
    )
    if len({track.id for track in tracks}) != len(tracks):
        raise ValueError("track IDs must be unique")
    stop_line = _parse_stop_line(document["stop_line"], lane) if "stop_line" in document else None
    return Scenario(lane, thresholds, tracks, stop_line)
