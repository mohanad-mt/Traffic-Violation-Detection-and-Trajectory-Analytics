from collections import defaultdict

import numpy as np

from .scenario import Observation, Track
from .tracker import FrameDetections


class BoxTracker:
    def __init__(
        self,
        name: str,
        fps: float,
        max_gap_s: float,
        confidence: float,
        *,
        keep_history: bool = True,
    ) -> None:
        try:
            import supervision as sv
            from trackers import ByteTrackTracker, OCSORTTracker
        except ImportError as error:
            raise ValueError("box tracking requires requirements-trackers.txt") from error

        options = {
            "lost_track_buffer": round(max_gap_s * 30),
            "frame_rate": fps,
            "high_conf_det_threshold": confidence,
        }
        if name == "bytetrack":
            self.engine = ByteTrackTracker(track_activation_threshold=confidence, **options)
        elif name == "ocsort":
            self.engine = OCSORTTracker(**options)
        else:
            raise ValueError(f"unknown tracker: {name}")
        self.detections_type = sv.Detections
        self.keep_history = keep_history
        self.histories = defaultdict(list)

    def update_frame(
        self, timestamp_s: float, detections: FrameDetections
    ) -> list[tuple[str, Observation]]:
        boxes = detections.boxes
        tracked = self.engine.update(
            self.detections_type(
                xyxy=np.asarray(
                    [(b.x, b.y, b.x + b.width, b.y + b.height) for b in boxes], dtype=float
                ).reshape(-1, 4),
                confidence=np.asarray([b.score for b in boxes], dtype=float),
                data={"source_index": np.arange(len(boxes))},
            ),
            timestamp=timestamp_s,
        )
        observations = []
        if len(tracked):
            for track_id, index in zip(
                tracked.tracker_id, tracked.data["source_index"], strict=True
            ):
                if track_id < 0:
                    continue
                key = f"vehicle-{track_id}"
                box = boxes[index]
                sample = Observation(timestamp_s, box.position, box)
                if self.keep_history:
                    self.histories[key].append(sample)
                observations.append((key, sample))
        return observations

    def tracks(self) -> tuple[Track, ...]:
        return tuple(Track(key, tuple(history)) for key, history in self.histories.items())
