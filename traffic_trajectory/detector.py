from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from math import dist, hypot
from typing import Literal

from .scenario import Lane, Observation, Scenario, StopLine, Thresholds, Track


@dataclass(frozen=True)
class WrongWayEvent:
    track_id: str
    lane_id: str
    start_s: float
    confirmed_s: float
    reverse_distance_px: float
    type: Literal["wrong_way_candidate"] = "wrong_way_candidate"


@dataclass(frozen=True)
class StoppedVehicleEvent:
    track_id: str
    lane_id: str
    start_s: float
    confirmed_s: float
    type: Literal["stopped_vehicle_candidate"] = "stopped_vehicle_candidate"


@dataclass(frozen=True)
class RedLightEvent:
    track_id: str
    lane_id: str
    start_s: float
    confirmed_s: float
    type: Literal["red_light_crossing_candidate"] = "red_light_crossing_candidate"


type Event = WrongWayEvent | StoppedVehicleEvent | RedLightEvent


def _detect_track(track: Track, lane: Lane, thresholds: Thresholds) -> Iterator[WrongWayEvent]:
    start = None
    last_reverse_s = 0.0
    emitted = False
    samples = (
        (observation, lane.polygon.contains(observation.position))
        for observation in track.observations
    )
    for (previous, previous_in_lane), (current, current_in_lane) in pairwise(samples):
        dx = current.position[0] - previous.position[0]
        dy = current.position[1] - previous.position[1]
        step = hypot(dx, dy)
        reverse_step = -(dx * lane.forward[0] + dy * lane.forward[1])
        connected = (
            previous_in_lane
            and current_in_lane
            and current.timestamp_s - previous.timestamp_s <= thresholds.max_gap_s
        )
        if (
            connected
            and start is not None
            and step < thresholds.min_step_px
            and current.timestamp_s - last_reverse_s <= thresholds.max_pause_s
        ):
            continue
        qualifies = (
            connected
            and step >= thresholds.min_step_px
            and reverse_step >= step * thresholds.min_heading_cosine
        )
        if not qualifies:
            start = None
            emitted = False
            continue

        if start is None:
            start = previous
        last_reverse_s = current.timestamp_s
        reverse_distance = (start.position[0] - current.position[0]) * lane.forward[0] + (
            start.position[1] - current.position[1]
        ) * lane.forward[1]
        if (
            not emitted
            and reverse_distance >= thresholds.min_reverse_distance_px
            and current.timestamp_s - start.timestamp_s >= thresholds.min_duration_s
        ):
            yield WrongWayEvent(
                track.id, lane.id, start.timestamp_s, current.timestamp_s, reverse_distance
            )
            emitted = True


def detect_wrong_way(scenario: Scenario) -> list[WrongWayEvent]:
    events = (
        event
        for track in scenario.tracks
        for event in _detect_track(track, scenario.lane, scenario.thresholds)
    )
    return sorted(events, key=lambda event: (event.confirmed_s, event.track_id))


@dataclass(frozen=True)
class _StopRun:
    track_id: str
    anchor: Observation
    last: Observation
    confirmed_s: float | None
    interrupted: bool


def _stationary_runs(track: Track, lane: Lane, thresholds: Thresholds) -> Iterator[_StopRun]:
    anchor = None
    last = None
    confirmed_s = None
    for sample in track.observations:
        in_lane = lane.polygon.contains(sample.position)
        gap = last is not None and sample.timestamp_s - last.timestamp_s > thresholds.max_gap_s
        if anchor is not None and (
            gap
            or not in_lane
            or dist(anchor.position, sample.position) > thresholds.stopped_radius_px
        ):
            yield _StopRun(track.id, anchor, last, confirmed_s, gap)
            anchor = None
            confirmed_s = None
        if not in_lane:
            continue
        if anchor is None:
            anchor = sample
        last = sample
        if (
            confirmed_s is None
            and last.timestamp_s - anchor.timestamp_s >= thresholds.stopped_duration_s
        ):
            confirmed_s = last.timestamp_s

    if anchor is not None:
        yield _StopRun(track.id, anchor, last, confirmed_s, True)


def _detect_stopped(scenario: Scenario) -> Iterator[StoppedVehicleEvent]:
    lane, thresholds = scenario.lane, scenario.thresholds
    runs = sorted(
        (run for track in scenario.tracks for run in _stationary_runs(track, lane, thresholds)),
        key=lambda run: (run.anchor.timestamp_s, run.track_id),
    )
    reported = []
    bridge_duration = min(0.5, thresholds.stopped_duration_s / 2)
    for run in runs:
        start_s = run.anchor.timestamp_s
        reported = [
            (anchor, prior)
            for anchor, prior in reported
            if start_s - prior.last.timestamp_s <= thresholds.stopped_realert_gap_s
        ]
        if run.last.timestamp_s - start_s < bridge_duration:
            continue

        # Bridge a lost track only when there is one plausible stationary predecessor.
        matches = [
            (anchor, prior)
            for anchor, prior in reported
            if prior.last.timestamp_s < start_s
            and dist(prior.last.position, run.anchor.position) <= thresholds.stopped_radius_px
            and dist(anchor.position, run.last.position) <= thresholds.stopped_radius_px
        ]
        if len(matches) == 1:
            anchor, prior = matches[0]
            reported.remove((anchor, prior))
        elif run.confirmed_s is not None:
            yield StoppedVehicleEvent(run.track_id, lane.id, start_s, run.confirmed_s)
            anchor = run.anchor
        else:
            continue
        if run.interrupted:
            reported.append((anchor, run))


def _detect_red_light(
    track: Track, lane: Lane, thresholds: Thresholds, line: StopLine
) -> Iterator[RedLightEvent]:
    (x, y), (end_x, end_y) = line.segment
    dx, dy = end_x - x, end_y - y
    length = hypot(dx, dy)
    length_sq = length * length
    orientation = 1 if dx * lane.forward[1] - dy * lane.forward[0] > 0 else -1
    normal_x, normal_y = orientation * -dy / length, orientation * dx / length
    margin = thresholds.red_line_margin_px
    samples = (
        (sample, normal_x * (sample.position[0] - x) + normal_y * (sample.position[1] - y))
        for sample in track.observations
    )
    approached = False
    red_crossing_s = None
    for (previous, before), (current, after) in pairwise(samples):
        gap = current.timestamp_s - previous.timestamp_s
        if gap > thresholds.max_gap_s:
            approached = False
            red_crossing_s = None
            continue
        if before < 0:
            if not lane.polygon.contains(previous.position):
                approached = False
            elif before <= -margin:
                approached = True
        if after <= 0:
            red_crossing_s = None
        if not approached or after <= 0:
            continue

        if before <= 0:
            fraction = -before / (after - before)
            px, py = previous.position
            qx, qy = current.position
            step_x, step_y = qx - px, qy - py
            crossing = (px + fraction * step_x, py + fraction * step_y)
            along_line = ((crossing[0] - x) * dx + (crossing[1] - y) * dy) / length_sq
            crossed_s = previous.timestamp_s + fraction * gap
            if (
                0 <= along_line <= 1
                and step_x * lane.forward[0] + step_y * lane.forward[1] > 0
                and lane.polygon.contains(crossing)
                and any(start <= crossed_s < end for start, end in line.red_intervals_s)
            ):
                red_crossing_s = crossed_s

        if after >= margin:
            if red_crossing_s is not None:
                yield RedLightEvent(track.id, lane.id, red_crossing_s, current.timestamp_s)
            approached = False
            red_crossing_s = None


def detect_events(scenario: Scenario) -> list[Event]:
    events: list[Event] = list(detect_wrong_way(scenario))
    events.extend(_detect_stopped(scenario))
    for track in scenario.tracks:
        if scenario.stop_line is not None:
            events.extend(
                _detect_red_light(track, scenario.lane, scenario.thresholds, scenario.stop_line)
            )
    return sorted(events, key=lambda event: (event.confirmed_s, event.track_id, event.type))
