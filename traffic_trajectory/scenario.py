from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from math import hypot, isfinite
from typing import Any

from .calibration import GroundPlane
from .geometry import ConvexPolygon, Point, VehicleBox


@dataclass(frozen=True)
class Lane:
    id: str
    polygon: ConvexPolygon
    forward: Point


@dataclass(frozen=True)
class Observation:
    timestamp_s: float
    position: Point
    box: VehicleBox | None = None


@dataclass(frozen=True)
class Track:
    id: str
    observations: tuple[Observation, ...]


def track_record(track: Track) -> dict:
    observations = []
    for sample in track.observations:
        row = {"timestamp_s": sample.timestamp_s, "position": list(sample.position)}
        if sample.box is not None:
            row["box"] = asdict(sample.box)
        observations.append(row)
    return {"id": track.id, "observations": observations}


@dataclass(frozen=True)
class Thresholds:
    min_reverse_distance_px: float = 35.0
    min_duration_s: float = 1.0
    max_gap_s: float = 0.75
    min_step_px: float = 1.0
    min_reverse_speed_px_s: float = 0.0
    min_heading_cosine: float = 0.7
    max_pause_s: float = 0.25
    stopped_duration_s: float = 3.0
    stopped_radius_px: float = 4.0
    stopped_realert_gap_s: float = 2.0
    red_line_margin_px: float = 2.0
    max_stopped_drift_px_s: float = 0.0


@dataclass(frozen=True)
class StopLine:
    segment: tuple[Point, Point]
    red_intervals_s: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class LaneZone:
    lane: Lane
    stop_line: StopLine | None = None
    stop_exclusions: tuple[ConvexPolygon, ...] = ()


@dataclass(frozen=True)
class Scenario:
    zones: tuple[LaneZone, ...]
    thresholds: Thresholds
    tracks: tuple[Track, ...]
    ground_plane: GroundPlane | None = None

    @property
    def lane(self) -> Lane:
        return self.zones[0].lane

    @property
    def stop_line(self) -> StopLine | None:
        return self.zones[0].stop_line


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


def _parse_polygon(value: object, path: str) -> ConvexPolygon:
    vertices = tuple(
        _point(point, f"{path}[{index}]") for index, point in enumerate(_array(value, path))
    )
    return ConvexPolygon.from_vertices(vertices)


def _parse_lane(value: object, path: str) -> Lane:
    lane = _object(value, path)
    lane_id = _id(lane.get("id"), f"{path}.id")
    polygon = _parse_polygon(lane.get("polygon"), f"{path}.polygon")
    x, y = _point(lane.get("forward"), f"{path}.forward")
    # Scale before taking the norm to allow any finite, nonzero direction vector.
    scale = max(abs(x), abs(y))
    if scale == 0:
        raise ValueError("lane.forward must be nonzero")
    x, y = x / scale, y / scale
    length = hypot(x, y)
    return Lane(lane_id, polygon, (x / length, y / length))


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
        if name in {
            "max_pause_s",
            "max_stopped_drift_px_s",
            "red_line_margin_px",
            "stopped_realert_gap_s",
            "min_reverse_speed_px_s",
            "min_step_px",
        }:
            if number < 0:
                raise ValueError(f"thresholds.{name} must be nonnegative")
        elif number <= 0:
            raise ValueError(f"thresholds.{name} must be positive")
    if values["min_heading_cosine"] > 1:
        raise ValueError("thresholds.min_heading_cosine must be at most 1")
    if values["min_step_px"] == 0 and values["min_reverse_speed_px_s"] == 0:
        raise ValueError("set a positive minimum step or reverse speed")
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
        raw_box = observation.get("box")
        box = _parse_box(raw_box, f"{observation_path}.box") if raw_box is not None else None
        observations.append(Observation(timestamp, point, box))

    if any(right.timestamp_s <= left.timestamp_s for left, right in pairwise(observations)):
        raise ValueError(f"{path}.observations timestamps must be strictly increasing")
    return Track(track_id, tuple(observations))


def _parse_box(value: object, path: str) -> VehicleBox:
    box = _object(value, path)
    x, y, width, height, score = (
        _number(box.get(key), f"{path}.{key}") for key in ("x", "y", "width", "height", "score")
    )
    if width <= 0 or height <= 0 or not 0 <= score <= 1:
        raise ValueError(f"{path} must have positive sides and a score in [0, 1]")
    category = box.get("category")
    return VehicleBox(
        x,
        y,
        width,
        height,
        score,
        _id(category, f"{path}.category") if category is not None else None,
    )


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
    if "lanes" in document:
        if "lane" in document or "stop_line" in document or "stop_exclusions" in document:
            raise ValueError("define either lane with top-level rules, or lanes with lane rules")
        zones = []
        for index, value in enumerate(_array(document["lanes"], "lanes")):
            path = f"lanes[{index}]"
            config = _object(value, path)
            zones.append(_parse_zone(_parse_lane(config, path), config, path))
        if not zones or len({zone.lane.id for zone in zones}) != len(zones):
            raise ValueError("lanes must contain at least one lane with unique IDs")
    else:
        zones = [_parse_zone(_parse_lane(document.get("lane"), "lane"), document, "scenario")]
    thresholds = _parse_thresholds(document.get("thresholds", {}))
    tracks = tuple(
        _parse_track(track, f"tracks[{index}]")
        for index, track in enumerate(_array(document.get("tracks"), "tracks"))
    )
    if len({track.id for track in tracks}) != len(tracks):
        raise ValueError("track IDs must be unique")
    ground_plane = (
        _parse_ground_plane(document["ground_plane"]) if "ground_plane" in document else None
    )
    return Scenario(tuple(zones), thresholds, tracks, ground_plane)


def _parse_ground_plane(value: object) -> GroundPlane:
    ground = _object(value, "ground_plane")
    path = "ground_plane.image_to_world"
    matrix = tuple(
        tuple(_number(v, f"{path}[{i}][{j}]") for j, v in enumerate(_array(row, f"{path}[{i}]")))
        for i, row in enumerate(_array(ground.get("image_to_world"), path))
    )
    return GroundPlane(matrix, _id(ground.get("unit"), "ground_plane.unit"))


def _parse_zone(lane: Lane, config: dict, path: str) -> LaneZone:
    stop_line = _parse_stop_line(config["stop_line"], lane) if "stop_line" in config else None
    exclusion_path = f"{path}.stop_exclusions"
    exclusions = tuple(
        _parse_polygon(raw, f"{exclusion_path}[{index}]")
        for index, raw in enumerate(_array(config.get("stop_exclusions", []), exclusion_path))
    )
    return LaneZone(lane, stop_line, exclusions)
