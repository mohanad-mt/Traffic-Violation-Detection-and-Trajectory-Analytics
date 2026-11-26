from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .tracker import FrameDetections

# OpenCV Zoo's COCO YOLOX-S export. Grid cells are ordered by stride, then row.
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
CLASS_COUNT = 80
VEHICLE_CLASSES = (2, 3, 5, 7)  # car, motorcycle, bus, truck
NMS_IOU = 0.45


@dataclass(frozen=True)
class VehicleBox:
    x: float
    y: float
    width: float
    height: float
    score: float

    @property
    def position(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height


def _suppress(boxes: list[VehicleBox], confidence: float) -> list[VehicleBox]:
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(
        [(box.x, box.y, box.width, box.height) for box in boxes],
        [box.score for box in boxes],
        confidence,
        NMS_IOU,
    )
    return [boxes[index] for index in keep]


def _head_cells() -> tuple[MatLike, MatLike]:
    grid = []
    stride_per_cell = []
    for stride in STRIDES:
        width = INPUT_SIZE // stride
        cells = np.arange(width * width)
        grid.append(np.column_stack((cells % width, cells // width)))
        stride_per_cell.append(np.full(len(cells), stride))
    return np.concatenate(grid), np.concatenate(stride_per_cell)


_GRID, _CELL_STRIDE = _head_cells()


def _vehicle_boxes(
    prediction: MatLike,
    scale: float,
    width: int,
    height: int,
    cutoff: float,
) -> list[VehicleBox]:
    if prediction.shape != (1, len(_GRID), 5 + CLASS_COUNT):
        raise ValueError("expected the OpenCV Zoo YOLOX-S raw COCO output (1, 8400, 85)")

    rows = prediction[0]
    class_probabilities = rows[:, 5:]
    best_class = class_probabilities.argmax(axis=1)
    scores = rows[:, 4] * class_probabilities.max(axis=1)
    selected = (scores > cutoff) & np.isfinite(scores) & np.isin(best_class, VEHICLE_CLASSES)
    if not selected.any():
        return []

    raw = rows[selected, :4].astype(np.float64)
    strides = _CELL_STRIDE[selected, None]
    centers = (raw[:, :2] + _GRID[selected]) * strides / scale
    with np.errstate(over="ignore", invalid="ignore"):
        sizes = np.exp(raw[:, 2:]) * strides / scale
    lower = np.clip(centers - sizes / 2, 0, [width - 1, height - 1])
    upper = np.clip(centers + sizes / 2, 0, [width - 1, height - 1])
    valid = np.isfinite(raw).all(axis=1) & np.isfinite(sizes).all(axis=1)
    valid &= (upper > lower).all(axis=1)
    if not valid.any():
        return []

    xy = lower[valid]
    wh = upper[valid] - xy
    scores = scores[selected][valid]
    return [
        VehicleBox(float(x), float(y), float(w), float(h), float(score))
        for (x, y), (w, h), score in zip(xy, wh, scores, strict=True)
    ]


def _tile_starts(length: int, tile_size: int) -> list[int]:
    last = max(0, length - tile_size)
    stride = max(1, tile_size * 3 // 4)
    return [*range(0, last, stride), last]


def _interior_box(
    box: VehicleBox,
    window: tuple[int, int, int, int],
    roi: tuple[int, int, int, int],
) -> bool:
    left, top, right, bottom = window
    return (
        (left == roi[0] or box.x > left + 1)
        and (top == roi[1] or box.y > top + 1)
        and (right == roi[2] or box.x + box.width < right - 2)
        and (bottom == roi[3] or box.y + box.height < bottom - 2)
    )


class YoloXDetector:
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        continuation_confidence: float | None = None,
        roi: tuple[int, int, int, int] | None = None,
        *,
        tile_size: int | None = None,
    ) -> None:
        if not model_path.is_file():
            raise ValueError(f"YOLOX model file does not exist: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.confidence = confidence
        self.continuation_confidence = (
            confidence if continuation_confidence is None else continuation_confidence
        )
        self.roi = roi
        if tile_size is not None and tile_size <= 0:
            raise ValueError("detection tile size must be positive")
        self.tile_size = tile_size

    def _window_boxes(self, frame: MatLike, window: tuple[int, int, int, int]) -> list[VehicleBox]:
        left, top, right, bottom = window
        frame = frame[top:bottom, left:right]
        width, height = right - left, bottom - top
        scale = INPUT_SIZE / max(width, height)
        resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        self.net.setInput(cv2.dnn.blobFromImage(padded, swapRB=True))
        boxes = _vehicle_boxes(
            self.net.forward(), scale, width, height, self.continuation_confidence
        )
        return [
            replace(box, x=box.x + left, y=box.y + top)
            for box in _suppress(boxes, self.continuation_confidence)
        ]

    def predict_boxes(self, frame: MatLike) -> list[VehicleBox]:
        frame_height, frame_width = frame.shape[:2]
        roi = self.roi or (0, 0, frame_width, frame_height)
        left, top, right, bottom = roi
        if not (0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height):
            raise ValueError("detection ROI must be a nonempty rectangle within the video frame")
        boxes = self._window_boxes(frame, roi)
        size = self.tile_size
        if size is None or max(right - left, bottom - top) <= size:
            return boxes

        # Keep the full view for vehicles that span more than one tile.
        for offset_y in _tile_starts(bottom - top, size):
            for offset_x in _tile_starts(right - left, size):
                x, y = left + offset_x, top + offset_y
                window = (x, y, min(x + size, right), min(y + size, bottom))
                boxes.extend(
                    box
                    for box in self._window_boxes(frame, window)
                    if _interior_box(box, window, roi)
                )
        return _suppress(boxes, self.continuation_confidence)

    def __call__(self, frame: MatLike) -> FrameDetections:
        boxes = self.predict_boxes(frame)
        return FrameDetections(
            sorted(box.position for box in boxes if box.score > self.confidence),
            sorted(box.position for box in boxes if box.score <= self.confidence),
        )
