import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np
from cv2.typing import MatLike

from traffic_trajectory.yolox import VehicleBox, YoloXDetector

VEHICLE_IDS = {0, 3, 4, 6}  # car, bus, truck, motorcycle in Foggy Driving
SMALL_AREA = 32**2
MATCH_IOU = 0.5


def _labels(text: str) -> list[VehicleBox]:
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cls, xmin, ymin, xmax, ymax = map(int, line.split())
        if cls in VEHICLE_IDS:
            # The archive uses one-based inclusive coordinates.
            boxes.append(VehicleBox(xmin - 1, ymin - 1, xmax - xmin + 1, ymax - ymin + 1, 1))
    return boxes


def _area(box: VehicleBox) -> float:
    return box.width * box.height


def _iou(a: VehicleBox, b: VehicleBox) -> float:
    width = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    height = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    intersection = max(0.0, width) * max(0.0, height)
    return intersection / (_area(a) + _area(b) - intersection)


def _matched_labels(predictions: list[VehicleBox], labels: list[VehicleBox]) -> list[VehicleBox]:
    remaining = labels.copy()
    matched = []
    for box in sorted(predictions, key=lambda item: -item.score):
        if not remaining:
            break
        best = max(range(len(remaining)), key=lambda index: _iou(box, remaining[index]))
        if _iou(box, remaining[best]) >= MATCH_IOU:
            matched.append(remaining.pop(best))
    return matched


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def examples(archive: Path, split: str) -> Iterator[tuple[MatLike, list[VehicleBox]]]:
    expected_images = {"test": 33, "test_extra": 68}[split]
    with ZipFile(archive) as dataset:
        files = sorted(
            name
            for name in dataset.namelist()
            if f"/leftImg8bit/{split}/" in name and name.endswith("_leftImg8bit.png")
        )
        if len(files) != expected_images:
            raise ValueError(f"expected {expected_images} images in Foggy Driving's {split} split")
        for name in files:
            labels = _labels(
                dataset.read(
                    name.replace("/leftImg8bit/", "/bboxGt/").removesuffix("_leftImg8bit.png")
                    + ".txt"
                ).decode("utf-8")
            )
            frame = cv2.imdecode(
                np.frombuffer(dataset.read(name), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError(f"cannot decode {name}")
            yield frame, labels


@dataclass
class BoxCounts:
    images: int = 0
    labels: int = 0
    predictions: int = 0
    matches: int = 0
    small_labels: int = 0
    small_matches: int = 0

    def add(self, boxes: list[VehicleBox], labels: list[VehicleBox]) -> None:
        hits = _matched_labels(boxes, labels)
        self.images += 1
        self.labels += len(labels)
        self.predictions += len(boxes)
        self.matches += len(hits)
        self.small_labels += sum(_area(label) < SMALL_AREA for label in labels)
        self.small_matches += sum(_area(label) < SMALL_AREA for label in hits)

    @property
    def precision(self) -> float | None:
        return _rate(self.matches, self.predictions)

    def report(self) -> dict:
        return {
            "images": self.images,
            "iou": MATCH_IOU,
            "labels": self.labels,
            "true_positives": self.matches,
            "false_positives": self.predictions - self.matches,
            "false_negatives": self.labels - self.matches,
            "precision": self.precision,
            "recall": _rate(self.matches, self.labels),
            "small_labels": self.small_labels,
            "small_matches": self.small_matches,
            "small_recall": _rate(self.small_matches, self.small_labels),
        }


def evaluate(archive: Path, detector: YoloXDetector) -> dict:
    counts = BoxCounts()
    seconds = 0.0
    for frame, labels in examples(archive, "test"):
        start = perf_counter()
        boxes = detector.predict_boxes(frame)
        seconds += perf_counter() - start
        counts.add(boxes, labels)
    return {**counts.report(), "mean_inference_ms": 1000 * seconds / counts.images}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score YOLOX boxes on Foggy Driving's fine split."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-confidence", type=float)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    cv2.setNumThreads(4)
    try:
        detector = YoloXDetector(
            args.model,
            args.confidence,
            tile_size=args.tile_size,
            tile_confidence=args.tile_confidence,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "confidence": args.confidence,
                "tile_size": args.tile_size,
                "tile_confidence": args.tile_confidence,
                **evaluate(args.archive, detector),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
