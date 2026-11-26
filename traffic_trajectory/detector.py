from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from math import hypot
from typing import Literal

from .scenario import Lane, Scenario, Thresholds, Track


@dataclass(frozen=True)
class WrongWayEvent:
    track_id: str
    lane_id: str
    start_s: float
    confirmed_s: float
    reverse_distance_px: float
    type: Literal["wrong_way_candidate"] = "wrong_way_candidate"


def _detect_track(track: Track, lane: Lane, thresholds: Thresholds) -> Iterator[WrongWayEvent]:
    start_s = None
    reverse_distance = 0.0
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
        qualifies = (
            previous_in_lane
            and current_in_lane
            and current.timestamp_s - previous.timestamp_s <= thresholds.max_gap_s
            and step >= thresholds.min_step_px
            and reverse_step >= step * thresholds.min_heading_cosine
        )
        if not qualifies:
            start_s = None
            reverse_distance = 0.0
            emitted = False
            continue

        if start_s is None:
            start_s = previous.timestamp_s
        reverse_distance += reverse_step
        if (
            not emitted
            and reverse_distance >= thresholds.min_reverse_distance_px
            and current.timestamp_s - start_s >= thresholds.min_duration_s
        ):
            yield WrongWayEvent(track.id, lane.id, start_s, current.timestamp_s, reverse_distance)
            emitted = True


def detect_wrong_way(scenario: Scenario) -> list[WrongWayEvent]:
    events = (
        event
        for track in scenario.tracks
        for event in _detect_track(track, scenario.lane, scenario.thresholds)
    )
    return sorted(events, key=lambda event: (event.confirmed_s, event.track_id))
