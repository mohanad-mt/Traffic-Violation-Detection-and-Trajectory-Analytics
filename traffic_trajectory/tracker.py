from dataclasses import dataclass, field
from math import dist
from typing import Protocol

from .geometry import Point, VehicleBox
from .scenario import Observation, Track


@dataclass(frozen=True)
class FrameDetections:
    strong: list[Point]
    weak: list[Point] = field(default_factory=list)
    boxes: tuple[VehicleBox, ...] = ()
    confidence: float = 0.0

    @classmethod
    def from_boxes(cls, boxes: list[VehicleBox], confidence: float) -> "FrameDetections":
        return cls(
            strong=sorted(box.position for box in boxes if box.score > confidence),
            weak=sorted(box.position for box in boxes if box.score <= confidence),
            boxes=tuple(boxes),
            confidence=confidence,
        )


class Tracker(Protocol):
    def update_frame(
        self, timestamp_s: float, detections: FrameDetections
    ) -> list[tuple[str, Observation]]: ...

    def tracks(self) -> tuple[Track, ...]: ...


class PointTracker:
    """Associate frame detections with constant-velocity track predictions."""

    def __init__(
        self, max_gap_s: float, max_distance_px: float, *, keep_history: bool = True
    ) -> None:
        self.max_gap_s = max_gap_s
        self.max_distance_px = max_distance_px
        self.keep_history = keep_history
        self.histories: dict[int, list[Observation]] = {}
        self.next_id = 0
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
        boxes: list[VehicleBox] | None,
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
            position = detections[detection_index]
            box = boxes[detection_index] if boxes is not None else None
            self.histories[index].append(Observation(timestamp_s, position, box))
            matched_tracks.add(index)
            matched_detections.add(detection_index)
        return matched_tracks, matched_detections

    def update(
        self,
        timestamp_s: float,
        detections: list[Point],
        *,
        weak_detections: list[Point] | None = None,
        boxes: list[VehicleBox] | None = None,
        weak_boxes: list[VehicleBox] | None = None,
    ) -> None:
        current = {
            index
            for index in self.active
            if timestamp_s - self.histories[index][-1].timestamp_s <= self.max_gap_s
        }
        if not self.keep_history:
            for index in self.active - current:
                del self.histories[index]
        self.active = current
        matched_tracks, matched_detections = self._match(
            timestamp_s, detections, self.active, self.max_distance_px, boxes
        )

        if weak_detections:
            self._match(
                timestamp_s,
                weak_detections,
                self.active - matched_tracks,
                self.max_distance_px / 2,
                weak_boxes,
            )

        for detection_index, position in enumerate(detections):
            if detection_index not in matched_detections:
                box = boxes[detection_index] if boxes is not None else None
                self.active.add(self.next_id)
                self.histories[self.next_id] = [Observation(timestamp_s, position, box)]
                self.next_id += 1
        if not self.keep_history:
            for index in self.active:
                del self.histories[index][:-2]

    def update_frame(
        self, timestamp_s: float, detections: FrameDetections
    ) -> list[tuple[str, Observation]]:
        if detections.boxes:
            ordered = sorted(detections.boxes, key=lambda box: box.position)
            strong = [box for box in ordered if box.score > detections.confidence]
            weak = [box for box in ordered if box.score <= detections.confidence]
            self.update(
                timestamp_s,
                [box.position for box in strong],
                weak_detections=[box.position for box in weak],
                boxes=strong,
                weak_boxes=weak,
            )
        else:
            self.update(timestamp_s, detections.strong, weak_detections=detections.weak)
        return [
            (f"vehicle-{index + 1}", self.histories[index][-1])
            for index in sorted(self.active)
            if self.histories[index][-1].timestamp_s == timestamp_s
        ]

    def tracks(self) -> tuple[Track, ...]:
        return tuple(
            Track(f"vehicle-{index + 1}", tuple(history))
            for index, history in self.histories.items()
        )


def make_tracker(
    name: str,
    fps: float,
    max_gap_s: float,
    max_distance_px: float,
    confidence: float,
    *,
    keep_history: bool = True,
) -> Tracker:
    if name == "point":
        return PointTracker(max_gap_s, max_distance_px, keep_history=keep_history)
    from .box_tracking import BoxTracker

    return BoxTracker(name, fps, max_gap_s, confidence, keep_history=keep_history)
