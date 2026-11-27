from collections import defaultdict
from dataclasses import dataclass
from math import dist

from .detector import (
    Event,
    RedLightState,
    StoppedVehicleEvent,
    WrongWayState,
    stop_window_is_ready,
)
from .scenario import LaneZone, Observation, Scenario, Thresholds


@dataclass
class StationaryState:
    anchor: Observation | None = None
    last: Observation | None = None
    confirmed: bool = False

    def update(self, sample: Observation, zone: LaneZone, thresholds: Thresholds) -> bool:
        inside = zone.lane.polygon.contains(sample.position) and not any(
            polygon.contains(sample.position) for polygon in zone.stop_exclusions
        )
        if not inside:
            self.anchor = self.last = None
            self.confirmed = False
            return False
        if (
            self.anchor is None
            or dist(self.anchor.position, sample.position) > thresholds.stopped_radius_px
        ):
            self.anchor = sample
            self.confirmed = False
        self.last = sample
        if not self.confirmed and stop_window_is_ready(self.anchor, sample, thresholds):
            self.confirmed = True
            return True
        return False


@dataclass
class TrackState:
    last_seen: float
    wrong_way: WrongWayState
    red_light: RedLightState | None
    stationary: StationaryState


class StreamingEvents:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.active: dict[tuple[str, str], TrackState] = {}
        self.recent_stops: dict[str, list[tuple[Observation, Observation]]] = defaultdict(list)

    def _expire(self, timestamp_s: float) -> None:
        thresholds = self.scenario.thresholds
        expired = [
            key
            for key, state in self.active.items()
            if timestamp_s - state.last_seen > thresholds.max_gap_s
        ]
        for key in expired:
            state = self.active.pop(key).stationary
            if state.confirmed:
                self.recent_stops[key[1]].append((state.anchor, state.last))
        retain_s = thresholds.stopped_realert_gap_s + thresholds.stopped_duration_s
        for lane_id, stops in self.recent_stops.items():
            self.recent_stops[lane_id] = [
                (anchor, last)
                for anchor, last in stops
                if timestamp_s - last.timestamp_s <= retain_s
            ]

    def update(
        self, timestamp_s: float, observations: list[tuple[str, Observation]]
    ) -> list[Event]:
        self._expire(timestamp_s)
        thresholds = self.scenario.thresholds
        events: list[Event] = []
        for track_id, sample in observations:
            for zone in self.scenario.zones:
                lane = zone.lane
                key = (track_id, lane.id)
                if key not in self.active:
                    self.active[key] = TrackState(
                        sample.timestamp_s,
                        WrongWayState(track_id, lane, thresholds),
                        RedLightState(track_id, lane, thresholds, zone.stop_line)
                        if zone.stop_line
                        else None,
                        StationaryState(),
                    )
                state = self.active[key]
                state.last_seen = timestamp_s
                if (wrong := state.wrong_way.update(sample)) is not None:
                    events.append(wrong)
                if state.red_light and (crossing := state.red_light.update(sample)) is not None:
                    events.append(crossing)
                if state.stationary.update(sample, zone, thresholds):
                    anchor = state.stationary.anchor
                    prior = [
                        pending
                        for pending in self.recent_stops[lane.id]
                        if 0
                        < anchor.timestamp_s - pending[1].timestamp_s
                        <= thresholds.stopped_realert_gap_s
                        and dist(pending[0].position, sample.position)
                        <= thresholds.stopped_radius_px
                        and dist(pending[1].position, anchor.position)
                        <= thresholds.stopped_radius_px
                    ]
                    if len(prior) == 1:
                        self.recent_stops[lane.id].remove(prior[0])
                        state.stationary.anchor = prior[0][0]
                    else:
                        events.append(
                            StoppedVehicleEvent(track_id, lane.id, anchor.timestamp_s, timestamp_s)
                        )
        return sorted(events, key=lambda event: (event.track_id, event.type, event.lane_id))
