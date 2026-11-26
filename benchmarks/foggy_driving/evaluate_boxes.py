import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np

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


def evaluate(archive: Path, detector: YoloXDetector) -> dict:
    images = expected = predicted = matched = small = small_matched = 0
    seconds = 0.0
    with ZipFile(archive) as dataset:
        files = sorted(
            name
            for name in dataset.namelist()
            if "/leftImg8bit/test/" in name and name.endswith("_leftImg8bit.png")
        )
        if len(files) != 33:
            raise ValueError("expected 33 images in the Foggy Driving fine test split")
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
            start = perf_counter()
            boxes = detector.predict_boxes(frame)
            seconds += perf_counter() - start
            hits = _matched_labels(boxes, labels)
            images += 1
            expected += len(labels)
            predicted += len(boxes)
            matched += len(hits)
            small += sum(_area(label) < SMALL_AREA for label in labels)
            small_matched += sum(_area(label) < SMALL_AREA for label in hits)

    return {
        "images": images,
        "iou": MATCH_IOU,
        "labels": expected,
        "true_positives": matched,
        "false_positives": predicted - matched,
        "false_negatives": expected - matched,
        "precision": _rate(matched, predicted),
        "recall": _rate(matched, expected),
        "small_labels": small,
        "small_matches": small_matched,
        "small_recall": _rate(small_matched, small),
        "mean_inference_ms": 1000 * seconds / images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score YOLOX boxes on Foggy Driving's fine split."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--tile-size", type=int)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    if args.tile_size is not None and args.tile_size <= 0:
        parser.error("--tile-size must be positive")
    cv2.setNumThreads(4)
    detector = YoloXDetector(args.model, args.confidence, tile_size=args.tile_size)
    print(
        json.dumps(
            {
                "confidence": args.confidence,
                "tile_size": args.tile_size,
                **evaluate(args.archive, detector),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
