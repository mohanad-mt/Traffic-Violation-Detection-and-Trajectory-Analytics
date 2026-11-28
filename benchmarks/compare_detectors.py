"""Run a detector once per image and report COCO AP and operating-point counts."""

import argparse
import io
import json
import platform
import subprocess
from contextlib import redirect_stdout
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean, quantiles
from time import perf_counter

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from benchmarks.assets import sha256
from benchmarks.dawn import fog_examples
from benchmarks.foggy_driving.evaluate_boxes import BoxCounts, examples
from traffic_trajectory.detection import DETECTORS, VehicleBox, make_detector

CATEGORIES = {name: index for index, name in enumerate(("car", "truck", "bus", "motorcycle"), 1)}
COCO_METRICS = (
    "map50_95",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar100",
    "ar_small",
    "ar_medium",
    "ar_large",
)
OPERATING_POINTS = (0.01, 0.03, 0.08, 0.20, 0.35, 0.50, 0.65, 0.75, 0.9)


def coco_scores(
    labels: list[list[VehicleBox]], predictions: list[list[VehicleBox]], *, class_aware: bool
) -> dict:
    images = [{"id": image} for image in range(1, len(labels) + 1)]
    categories = (
        [{"id": cid, "name": name} for name, cid in CATEGORIES.items()]
        if class_aware
        else [{"id": 1, "name": "vehicle"}]
    )

    def rows(groups):
        for image, boxes in enumerate(groups, 1):
            for box in boxes:
                yield {
                    "image_id": image,
                    "category_id": CATEGORIES[box.category] if class_aware else 1,
                    "bbox": [box.x, box.y, box.width, box.height],
                    "area": box.width * box.height,
                    "iscrowd": 0,
                    "score": box.score,
                }

    actual, predicted = COCO(), COCO()
    for index, (dataset, groups) in enumerate(((actual, labels), (predicted, predictions))):
        dataset.dataset = {
            "images": images,
            "categories": categories,
            "annotations": [{"id": row_id, **row} for row_id, row in enumerate(rows(groups), 1)],
        }
    with redirect_stdout(io.StringIO()):
        actual.createIndex()
        predicted.createIndex()
        evaluator = COCOeval(actual, predicted, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        key: float(value) if value >= 0 else None
        for key, value in zip(COCO_METRICS, evaluator.stats, strict=True)
    }


def run(args: argparse.Namespace) -> dict:
    detector = make_detector(
        args.detector,
        args.model,
        OPERATING_POINTS[0],
        tile_size=args.tile_size,
        tile_confidence=args.tile_confidence,
        device=args.device,
        input_color=args.input_color,
    )
    frames = (
        fog_examples(args.archive)
        if args.dataset == "dawn-fog"
        else examples(args.archive, args.split)
    )
    cv2.setNumThreads(4)
    labels, predictions, latencies = [], [], []
    for frame, actual in frames:
        if not labels:
            for _ in range(5):
                detector.predict_boxes(frame)
        started = perf_counter()
        predictions.append(detector.predict_boxes(frame))
        latencies.append((perf_counter() - started) * 1000)
        labels.append(actual)

    operating_points = []
    for cutoff in OPERATING_POINTS:
        counts = BoxCounts()
        for boxes, actual in zip(predictions, labels, strict=True):
            counts.add([box for box in boxes if box.score > cutoff], actual)
        report = counts.report()
        denominator = counts.labels + counts.predictions
        operating_points.append(
            {
                "confidence": cutoff,
                **report,
                "f1": 2 * counts.matches / denominator if denominator else None,
            }
        )
    if args.cache is not None:
        with args.cache.open("w", encoding="utf-8") as out:
            json.dump([[asdict(box) for box in boxes] for boxes in predictions], out)
    weights = args.model / "model.safetensors" if args.model.is_dir() else args.model
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "id": args.id,
        "git_commit": revision or None,
        "dataset": args.dataset,
        "split": "fog (all 300)" if args.dataset == "dawn-fog" else args.split,
        "archive_sha256": sha256(args.archive),
        "model": args.detector,
        "weights_sha256": sha256(weights),
        "configuration": {
            "device": args.device,
            "input_color": args.input_color,
            "tile_size": args.tile_size,
            "tile_confidence": args.tile_confidence,
            "min_cached_score": OPERATING_POINTS[0],
            "opencv_threads": 4,
        },
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "pycocotools": version("pycocotools"),
            "platform": platform.platform(),
        },
        "images": len(labels),
        "labels": sum(map(len, labels)),
        "coco_class_agnostic": coco_scores(labels, predictions, class_aware=False),
        "coco_class_aware": coco_scores(labels, predictions, class_aware=True),
        "operating_points": operating_points,
        "mean_detector_ms": mean(latencies),
        "p95_detector_ms": quantiles(latencies, n=20, method="inclusive")[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", help="Experiment ID")
    parser.add_argument("archive", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--dataset", choices=("foggy-driving", "dawn-fog"), required=True)
    parser.add_argument("--split", choices=("test", "test_extra"), default="test")
    parser.add_argument("--detector", choices=DETECTORS, default="yolox")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input-color", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-confidence", type=float)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
