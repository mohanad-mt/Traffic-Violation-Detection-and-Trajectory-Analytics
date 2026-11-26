from dataclasses import dataclass, field
from math import dist

from .geometry import Point
from .scenario import Observation, Track


@dataclass(frozen=True)
class FrameDetections:
    strong: list[Point]
    weak: list[Point] = field(default_factory=list)


class PointTracker:
    """Associate frame detections with constant-velocity track predictions."""

    def __init__(self, max_gap_s: float, max_distance_px: float) -> None:
        self.max_gap_s = max_gap_s
        self.max_distance_px = max_distance_px
        self.histories: list[list[Observation]] = []
        self.active: set[int] = set()

    def _predict(self, track: list[Observation], timestamp_s: float) -> Point:
        last = track[-1]
        if len(track) == 1:
            return last.position
        previous = track[-2]
        ratio = (timestamp_s - last.timestamp_s) / (last.timestamp_s - previous.timestamp_s)
        x, y = last.position
        px, py = previous.position
        return x + ratio * (x - px), y + ratio * (y - py)

    def _match(
        self,
        timestamp_s: float,
        detections: list[Point],
        tracks: set[int],
        distance_gate: float,
    ) -> tuple[set[int], set[int]]:
        candidates = []
        for index in tracks:
            predicted = self._predict(self.histories[index], timestamp_s)
            for detection_index, position in enumerate(detections):
                distance = dist(predicted, position)
                if distance <= distance_gate:
                    candidates.append((distance, index, detection_index))

        matched_tracks = set()
        matched_detections = set()
        for _, index, detection_index in sorted(candidates):
            if index in matched_tracks or detection_index in matched_detections:
                continue
            self.histories[index].append(Observation(timestamp_s, detections[detection_index]))
            matched_tracks.add(index)
            matched_detections.add(detection_index)
        return matched_tracks, matched_detections

    def update(
        self,
        timestamp_s: float,
        detections: list[Point],
        *,
        weak_detections: list[Point] | None = None,
    ) -> None:
        self.active = {
            index
            for index in self.active
            if timestamp_s - self.histories[index][-1].timestamp_s <= self.max_gap_s
        }
        matched_tracks, matched_detections = self._match(
            timestamp_s, detections, self.active, self.max_distance_px
        )

        if weak_detections:
            self._match(
                timestamp_s,
                weak_detections,
                self.active - matched_tracks,
                self.max_distance_px / 2,
            )

        for detection_index, position in enumerate(detections):
            if detection_index not in matched_detections:
                self.active.add(len(self.histories))
                self.histories.append([Observation(timestamp_s, position)])

    def tracks(self) -> tuple[Track, ...]:
        return tuple(
            Track(f"vehicle-{index}", tuple(history))
            for index, history in enumerate(self.histories, start=1)
        )
