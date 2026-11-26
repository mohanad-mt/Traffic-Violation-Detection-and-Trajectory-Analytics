from math import dist

from .geometry import Point
from .scenario import Observation, Track


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

    def update(self, timestamp_s: float, detections: list[Point]) -> None:
        self.active = {
            index
            for index in self.active
            if timestamp_s - self.histories[index][-1].timestamp_s <= self.max_gap_s
        }
        candidates = []
        for index in self.active:
            predicted = self._predict(self.histories[index], timestamp_s)
            for detection_index, position in enumerate(detections):
                distance = dist(predicted, position)
                if distance <= self.max_distance_px:
                    candidates.append((distance, index, detection_index))

        matched_tracks = set()
        matched_detections = set()
        for _, index, detection_index in sorted(candidates):
            if index in matched_tracks or detection_index in matched_detections:
                continue
            self.histories[index].append(Observation(timestamp_s, detections[detection_index]))
            matched_tracks.add(index)
            matched_detections.add(detection_index)

        for detection_index, position in enumerate(detections):
            if detection_index not in matched_detections:
                self.active.add(len(self.histories))
                self.histories.append([Observation(timestamp_s, position)])

    def tracks(self) -> tuple[Track, ...]:
        return tuple(
            Track(f"vehicle-{index}", tuple(history))
            for index, history in enumerate(self.histories, start=1)
        )
