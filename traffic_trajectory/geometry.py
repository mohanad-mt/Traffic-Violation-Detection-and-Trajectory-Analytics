from dataclasses import dataclass
from itertools import pairwise

Point = tuple[float, float]
Edge = tuple[Point, Point]


def _side(start: Point, end: Point, point: Point) -> float:
    """Return the signed cross product of an edge and the offset to the point."""
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )


@dataclass(frozen=True)
class ConvexPolygon:
    edges: tuple[Edge, ...]
    orientation: int

    @classmethod
    def from_vertices(cls, vertices: tuple[Point, ...]) -> "ConvexPolygon":
        if len(vertices) < 3 or len(set(vertices)) != len(vertices):
            raise ValueError("lane.polygon must have at least three distinct vertices")

        edges = tuple(pairwise((*vertices, vertices[0])))
        orientation = 1 if _side(*vertices[:3]) > 0 else -1
        for start, end in edges:
            for vertex in vertices:
                if vertex not in (start, end) and not orientation * _side(start, end, vertex) > 0:
                    raise ValueError(
                        "lane.polygon vertices must follow a strictly convex boundary"
                    )
        return cls(edges, orientation)

    def contains(self, point: Point) -> bool:
        return all(self.orientation * _side(start, end, point) >= 0 for start, end in self.edges)
