from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .geometry import Point

# OpenCV Zoo's COCO YOLOX-S export. Grid cells are ordered by stride, then row.
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
CLASS_COUNT = 80
VEHICLE_CLASSES = (2, 3, 5, 7)  # car, motorcycle, bus, truck


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


def _vehicle_points(
    prediction: MatLike, scale: float, width: int, height: int, confidence: float
) -> list[Point]:
    if prediction.shape != (1, len(_GRID), 5 + CLASS_COUNT):
        raise ValueError("expected the OpenCV Zoo YOLOX-S raw COCO output (1, 8400, 85)")

    rows = prediction[0]
    class_probabilities = rows[:, 5:]
    best_class = class_probabilities.argmax(axis=1)
    scores = rows[:, 4] * class_probabilities.max(axis=1)
    selected = (scores > confidence) & np.isin(best_class, VEHICLE_CLASSES)
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
    boxes = np.column_stack((xy, wh))
    survivors = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores[selected][valid].tolist(), confidence, 0.45
    )
    return sorted(
        (float(boxes[i, 0] + boxes[i, 2] / 2), float(boxes[i, 1] + boxes[i, 3])) for i in survivors
    )


class YoloXDetector:
    def __init__(self, model_path: Path, confidence: float) -> None:
        if not model_path.is_file():
            raise ValueError(f"YOLOX model file does not exist: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.confidence = confidence

    def __call__(self, frame: MatLike) -> list[Point]:
        height, width = frame.shape[:2]
        scale = INPUT_SIZE / max(width, height)
        resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        self.net.setInput(cv2.dnn.blobFromImage(padded, swapRB=True))
        return _vehicle_points(self.net.forward(), scale, width, height, self.confidence)
