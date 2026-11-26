from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .detection import BoxDetector, _suppress
from .detection import VehicleBox as VehicleBox

# OpenCV Zoo's COCO YOLOX-S export. Grid cells are ordered by stride, then row.
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
CLASS_COUNT = 80
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


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
    selected = (
        (scores > cutoff) & np.isfinite(scores) & np.isin(best_class, tuple(VEHICLE_CLASSES))
    )
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
    classes = best_class[selected][valid]
    return [
        VehicleBox(float(x), float(y), float(w), float(h), float(score), VEHICLE_CLASSES[cls])
        for (x, y), (w, h), score, cls in zip(xy, wh, scores, classes, strict=True)
    ]


class YoloXDetector(BoxDetector):
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        continuation_confidence: float | None = None,
        roi: tuple[int, int, int, int] | None = None,
        *,
        tile_size: int | None = None,
        tile_confidence: float | None = None,
        device: str = "cpu",
        input_color: str = "rgb",
    ) -> None:
        super().__init__(confidence, continuation_confidence, roi, tile_size, tile_confidence)
        if not model_path.is_file():
            raise ValueError(f"YOLOX model file does not exist: {model_path}")
        if device == "cuda":
            from .inference import CudaOnnxNet

            self.net = CudaOnnxNet(model_path)
        elif device == "cpu":
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
        else:
            raise ValueError("YOLOX device must be cpu or cuda")
        if input_color not in {"rgb", "bgr"}:
            raise ValueError("YOLOX input color must be rgb or bgr")
        self.swap_rb = input_color == "rgb"

    def _crop_boxes(self, frame: MatLike) -> list[VehicleBox]:
        height, width = frame.shape[:2]
        scale = INPUT_SIZE / max(width, height)
        resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        self.net.setInput(cv2.dnn.blobFromImage(padded, swapRB=self.swap_rb))
        boxes = _vehicle_boxes(
            self.net.forward(), scale, width, height, self.continuation_confidence
        )
        return _suppress(boxes, self.continuation_confidence)
