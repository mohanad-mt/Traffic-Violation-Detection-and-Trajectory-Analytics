from collections.abc import Iterator
from dataclasses import dataclass
from math import dist, hypot
from typing import Literal

from .scenario import Lane, LaneZone, Observation, Scenario, StopLine, Thresholds, Track


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


class WrongWayState:
    def __init__(self, track_id: str, lane: Lane, thresholds: Thresholds) -> None:
        self.track_id = track_id
        self.lane = lane
        self.thresholds = thresholds
        self.previous: Observation | None = None
        self.previous_in_lane = False
        self.start: Observation | None = None
        self.last_reverse_s = 0.0
        self.emitted = False

    def update(self, current: Observation) -> WrongWayEvent | None:
        lane, thresholds = self.lane, self.thresholds
        current_in_lane = lane.polygon.contains(current.position)
        previous, self.previous = self.previous, current
        previous_in_lane, self.previous_in_lane = self.previous_in_lane, current_in_lane
        if previous is None:
            return None
        dx = current.position[0] - previous.position[0]
        dy = current.position[1] - previous.position[1]
        step = hypot(dx, dy)
        reverse_step = -(dx * lane.forward[0] + dy * lane.forward[1])
        gap = current.timestamp_s - previous.timestamp_s
        connected = previous_in_lane and current_in_lane and gap <= thresholds.max_gap_s
        minimum_step = max(thresholds.min_step_px, thresholds.min_reverse_speed_px_s * gap)
        if (
            connected
            and self.start is not None
            and step < minimum_step
            and current.timestamp_s - self.last_reverse_s <= thresholds.max_pause_s
        ):
            return None
        qualifies = (
            connected
            and step >= minimum_step
            and reverse_step >= thresholds.min_reverse_speed_px_s * gap
            and reverse_step >= step * thresholds.min_heading_cosine
        )
        if not qualifies:
            self.start = None
            self.emitted = False
            return None

        if self.start is None:
            self.start = previous
        start = self.start
        self.last_reverse_s = current.timestamp_s
        reverse_distance = (start.position[0] - current.position[0]) * lane.forward[0] + (
            start.position[1] - current.position[1]
        ) * lane.forward[1]
        if (
            not self.emitted
            and reverse_distance >= thresholds.min_reverse_distance_px
            and current.timestamp_s - start.timestamp_s >= thresholds.min_duration_s
        ):
            self.emitted = True
            return WrongWayEvent(
                self.track_id, lane.id, start.timestamp_s, current.timestamp_s, reverse_distance
            )
        return None


def _detect_track(track: Track, lane: Lane, thresholds: Thresholds) -> Iterator[WrongWayEvent]:
    state = WrongWayState(track.id, lane, thresholds)
    for sample in track.observations:
        if (event := state.update(sample)) is not None:
            yield event


def detect_wrong_way(scenario: Scenario) -> list[WrongWayEvent]:
    events = (
        event
        for track in scenario.tracks
        for zone in scenario.zones
        for event in _detect_track(track, zone.lane, scenario.thresholds)
    )
    return sorted(events, key=lambda event: (event.confirmed_s, event.track_id))


@dataclass(frozen=True)
class _StopRun:
    track_id: str
    anchor: Observation
    last: Observation
    confirmed_s: float | None
    interrupted: bool


def _stationary_runs(track: Track, zone: LaneZone, thresholds: Thresholds) -> Iterator[_StopRun]:
    anchor = None
    last = None
    confirmed_s = None
    for sample in track.observations:
        in_lane = zone.lane.polygon.contains(sample.position) and not any(
            polygon.contains(sample.position) for polygon in zone.stop_exclusions
        )
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


def _detect_stopped(scenario: Scenario, zone: LaneZone) -> Iterator[StoppedVehicleEvent]:
    lane, thresholds = zone.lane, scenario.thresholds
    runs = sorted(
        (run for track in scenario.tracks for run in _stationary_runs(track, zone, thresholds)),
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


class RedLightState:
    def __init__(self, track_id: str, lane: Lane, thresholds: Thresholds, line: StopLine) -> None:
        self.track_id = track_id
        self.lane = lane
        self.thresholds = thresholds
        self.line = line
        self.previous: tuple[Observation, float] | None = None
        self.approached = False
        self.red_crossing_s: float | None = None
        (x, y), (end_x, end_y) = line.segment
        dx, dy = end_x - x, end_y - y
        length = hypot(dx, dy)
        length_sq = length * length
        orientation = 1 if dx * lane.forward[1] - dy * lane.forward[0] > 0 else -1
        normal_x, normal_y = orientation * -dy / length, orientation * dx / length
        self.geometry = (x, y, dx, dy, length_sq)
        self.normal = (normal_x, normal_y)

    def update(self, current: Observation) -> RedLightEvent | None:
        lane, line, thresholds = self.lane, self.line, self.thresholds
        margin = thresholds.red_line_margin_px
        x, y, dx, dy, length_sq = self.geometry
        normal_x, normal_y = self.normal
        after = normal_x * (current.position[0] - x) + normal_y * (current.position[1] - y)
        last, self.previous = self.previous, (current, after)
        if last is None:
            return None
        previous, before = last
        gap = current.timestamp_s - previous.timestamp_s
        if gap > thresholds.max_gap_s:
            self.approached = False
            self.red_crossing_s = None
            return None
        if before < 0:
            if not lane.polygon.contains(previous.position):
                self.approached = False
            elif before <= -margin:
                self.approached = True
        if after <= 0:
            self.red_crossing_s = None
        if not self.approached or after <= 0:
            return None

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
                self.red_crossing_s = crossed_s

        if after < margin:
            return None
        crossing_s, self.red_crossing_s = self.red_crossing_s, None
        self.approached = False
        if crossing_s is not None:
            return RedLightEvent(self.track_id, lane.id, crossing_s, current.timestamp_s)
        return None


def _detect_red_light(
    track: Track, lane: Lane, thresholds: Thresholds, line: StopLine
) -> Iterator[RedLightEvent]:
    state = RedLightState(track.id, lane, thresholds, line)
    for sample in track.observations:
        if (event := state.update(sample)) is not None:
            yield event


def detect_events(scenario: Scenario) -> list[Event]:
    events: list[Event] = list(detect_wrong_way(scenario))
    for zone in scenario.zones:
        events.extend(_detect_stopped(scenario, zone))
        if zone.stop_line is not None:
            for track in scenario.tracks:
                events.extend(
                    _detect_red_light(track, zone.lane, scenario.thresholds, zone.stop_line)
                )
    return sorted(events, key=lambda event: (event.confirmed_s, event.track_id, event.type))
