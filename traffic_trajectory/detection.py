from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

import cv2
from cv2.typing import MatLike

from .geometry import VehicleBox as VehicleBox
from .tracker import FrameDetections

NMS_IOU = 0.45
DETECTORS = ("yolox", "rtdetr", "rfdetr", "rfdetr-large")


def _suppress(
    boxes: list[VehicleBox], confidence: float, overlap: float = NMS_IOU
) -> list[VehicleBox]:
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(
        [(box.x, box.y, box.width, box.height) for box in boxes],
        [box.score for box in boxes],
        confidence,
        overlap,
    )
    return [boxes[index] for index in keep]


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


class BoxDetector(ABC):
    def __init__(
        self,
        confidence: float,
        continuation_confidence: float | None,
        roi: tuple[int, int, int, int] | None,
        tile_size: int | None,
        tile_confidence: float | None,
    ) -> None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if continuation_confidence is not None and not 0 < continuation_confidence <= confidence:
            raise ValueError("continuation confidence must be positive and at most confidence")
        if tile_size is not None and tile_size <= 0:
            raise ValueError("detection tile size must be positive")
        if tile_confidence is not None:
            if tile_size is None:
                raise ValueError("tile confidence requires a detection tile size")
            if not 0 < tile_confidence < 1:
                raise ValueError("tile confidence must be between 0 and 1")
        self.confidence = confidence
        self.continuation_confidence = (
            confidence if continuation_confidence is None else continuation_confidence
        )
        self.roi = roi
        self.tile_size = tile_size
        self.tile_confidence = tile_confidence

    @abstractmethod
    def _crop_boxes(self, frame: MatLike) -> list[VehicleBox]: ...

    def _window_boxes(self, frame: MatLike, window: tuple[int, int, int, int]) -> list[VehicleBox]:
        left, top, right, bottom = window
        return [
            replace(box, x=box.x + left, y=box.y + top)
            for box in self._crop_boxes(frame[top:bottom, left:right])
        ]

    def _frame_views(self, frame: MatLike) -> tuple[list[VehicleBox], list[VehicleBox] | None]:
        frame_height, frame_width = frame.shape[:2]
        roi = self.roi or (0, 0, frame_width, frame_height)
        left, top, right, bottom = roi
        if not (0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height):
            raise ValueError("detection ROI must be a nonempty rectangle within the video frame")
        boxes = self._window_boxes(frame, roi)
        size = self.tile_size
        if size is None or max(right - left, bottom - top) <= size:
            return boxes, None

        tile_boxes = []
        for offset_y in _tile_starts(bottom - top, size):
            for offset_x in _tile_starts(right - left, size):
                x, y = left + offset_x, top + offset_y
                window = (x, y, min(x + size, right), min(y + size, bottom))
                tile_boxes.extend(
                    box
                    for box in self._window_boxes(frame, window)
                    if _interior_box(box, window, roi)
                )
        return boxes, tile_boxes

    def predict_boxes(self, frame: MatLike) -> list[VehicleBox]:
        boxes, tile_boxes = self._frame_views(frame)
        if tile_boxes is None:
            return boxes

        # Keep the full view for vehicles that span more than one tile.
        if self.tile_confidence is not None:
            tile_boxes = [box for box in tile_boxes if box.score > self.tile_confidence]
        return _suppress([*boxes, *tile_boxes], self.continuation_confidence)

    def __call__(self, frame: MatLike) -> FrameDetections:
        return FrameDetections.from_boxes(self.predict_boxes(frame), self.confidence)


def make_detector(
    name: str,
    model: Path,
    confidence: float,
    continuation_confidence: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    *,
    tile_size: int | None = None,
    tile_confidence: float | None = None,
    device: str = "cpu",
    input_color: str = "rgb",
) -> BoxDetector:
    options = {
        "tile_size": tile_size,
        "tile_confidence": tile_confidence,
        "device": device,
    }
    if name == "rtdetr":
        from .rtdetr import RTDetrDetector

        return RTDetrDetector(model, confidence, continuation_confidence, roi, **options)
    if name in ("rfdetr", "rfdetr-large"):
        from .rfdetr import RFDetrDetector

        return RFDetrDetector(
            model,
            confidence,
            continuation_confidence,
            roi,
            large=name == "rfdetr-large",
            **options,
        )
    if name == "yolox":
        from .yolox import YoloXDetector

        return YoloXDetector(
            model, confidence, continuation_confidence, roi, input_color=input_color, **options
        )
    raise ValueError(f"unknown detector: {name}")
