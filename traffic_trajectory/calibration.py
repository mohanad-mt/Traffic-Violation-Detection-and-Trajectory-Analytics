from dataclasses import dataclass
from math import isfinite

from .geometry import Point


@dataclass(frozen=True)
class GroundPlane:
    image_to_world: tuple[tuple[float, ...], ...]
    unit: str

    def __post_init__(self) -> None:
        if len(self.image_to_world) != 3 or any(len(row) != 3 for row in self.image_to_world):
            raise ValueError("ground_plane.image_to_world must be a 3 by 3 matrix")
        if not all(isfinite(value) for row in self.image_to_world for value in row):
            raise ValueError("homography coefficients must be finite")
        (a, b, c), (d, e, f), (g, h, i) = self.image_to_world
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if not isfinite(determinant) or determinant == 0:
            raise ValueError("ground-plane homography must be invertible")

    def project(self, point: Point) -> Point | None:
        (a, b, c), (d, e, f), (g, h, i) = self.image_to_world
        x, y = point
        scale = g * x + h * y + i
        if not isfinite(scale) or scale == 0:
            return None
        world = ((a * x + b * y + c) / scale, (d * x + e * y + f) / scale)
        return world if all(isfinite(value) for value in world) else None
