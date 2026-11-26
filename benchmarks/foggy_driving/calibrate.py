"""Sweep crop filtering on coarse annotations, then score the frozen choices."""

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import cv2

from benchmarks.foggy_driving.evaluate_boxes import BoxCounts, _area, examples
from traffic_trajectory.detection import NMS_IOU, BoxDetector, VehicleBox, _suppress
from traffic_trajectory.yolox import YoloXDetector


@dataclass(frozen=True)
class Settings:
    full_confidence: float = 0.35
    tile_confidence: float | None = None
    tile_max_area: int | None = None
    nms_iou: float = NMS_IOU


@dataclass(frozen=True)
class ImageBoxes:
    full: list[VehicleBox]
    tiles: list[VehicleBox] | None
    labels: list[VehicleBox]


def score(images: list[ImageBoxes], settings: Settings) -> BoxCounts:
    counts = BoxCounts()
    for image in images:
        boxes = [box for box in image.full if box.score > settings.full_confidence]
        if settings.tile_confidence is not None and image.tiles is not None:
            boxes.extend(
                box
                for box in image.tiles
                if box.score > settings.tile_confidence
                and (settings.tile_max_area is None or _area(box) <= settings.tile_max_area)
            )
            boxes = _suppress(boxes, 0.0, settings.nms_iou)
        counts.add(boxes, image.labels)
    return counts


def calibrate(archive: Path, detector: BoxDetector) -> dict:
    calibration = [
        ImageBoxes(*detector._frame_views(frame), labels)
        for frame, labels in examples(archive, "test_extra")
    ]
    baseline = Settings()
    equal_cutoffs = Settings(tile_confidence=0.35)
    precision_floor = score(calibration, baseline).precision
    if precision_floor is None:
        raise ValueError("the baseline yielded no predictions on the calibration split")

    # Inference runs once per view. Higher cutoffs only remove boxes in score order.
    grid = (
        Settings(*values)
        for values in product(
            (0.35, 0.40, 0.45, 0.50),
            (0.35, 0.45, 0.55, 0.65, 0.75),
            (32**2, 64**2, 96**2, None),
            (0.35, NMS_IOU),
        )
    )
    trials = [(settings, score(calibration, settings)) for settings in grid]
    eligible = sorted(
        (
            (settings, counts)
            for settings, counts in trials
            if counts.precision is not None and counts.precision >= precision_floor
        ),
        key=lambda trial: (trial[1].matches, trial[1].small_matches, trial[1].precision),
        reverse=True,
    )
    choices = {"full_frame": baseline, "equal_cutoffs": equal_cutoffs}
    if eligible:
        choices["best_grid"] = eligible[0][0]
    cutoff_only = next(
        (
            settings
            for settings, _ in eligible
            if settings.full_confidence == baseline.full_confidence
            and settings.tile_max_area is None
            and settings.nms_iou == NMS_IOU
        ),
        None,
    )
    if cutoff_only is not None:
        choices["crop_cutoff_only"] = cutoff_only

    evaluation = [
        ImageBoxes(*detector._frame_views(frame), labels)
        for frame, labels in examples(archive, "test")
    ]
    return {
        "precision_floor": precision_floor,
        "grid_trials": len(trials),
        "eligible_trials": len(eligible),
        "tile_size": detector.tile_size,
        "choices": {
            name: {
                "settings": asdict(settings),
                "calibration": score(calibration, settings).report(),
                "evaluation": score(evaluation, settings).report(),
            }
            for name, settings in choices.items()
        },
        "calibration_grid": [
            {"settings": asdict(settings), **counts.report()} for settings, counts in trials
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--cuda", action="store_true", help="Replay the pinned graph on CUDA")
    args = parser.parse_args()

    cv2.setNumThreads(4)
    try:
        detector = YoloXDetector(args.model, 0.34, tile_size=640)
        if args.cuda:
            from benchmarks.foggy_driving.cuda_probe import load_cuda_graph

            detector.net = load_cuda_graph(args.model)
        print(json.dumps(calibrate(args.archive, detector), indent=2))
    except (ImportError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
