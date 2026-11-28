from collections.abc import Iterator

from .calibration import GroundPlane
from .geometry import Point
from .scenario import Track


def project_track(track: Track, ground: GroundPlane, max_gap_s: float) -> Iterator[dict]:
    previous_point: Point | None = None
    previous_s = 0.0
    for sample in track.observations:
        point = ground.project(sample.position)
        dt = sample.timestamp_s - previous_s
        velocity = None
        if previous_point is not None and point is not None and 0 < dt <= max_gap_s:
            velocity = ((point[0] - previous_point[0]) / dt, (point[1] - previous_point[1]) / dt)
        yield {"timestamp_s": sample.timestamp_s, "position": point, "velocity": velocity}
        previous_point, previous_s = point, sample.timestamp_s
