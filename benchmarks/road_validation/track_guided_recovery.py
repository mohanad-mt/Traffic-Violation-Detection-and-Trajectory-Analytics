"""Measure whether a prior-frame proposal helps recover a missed RT-DETR detection."""

import argparse
import json
import platform
import subprocess
from math import dist
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
from cv2.typing import MatLike

from benchmarks.assets import sha256
from benchmarks.road_validation.compare import DATA, Case, cases, read_json, replay
from traffic_trajectory.detection import BoxDetector, _suppress
from traffic_trajectory.geometry import ConvexPolygon, Point, VehicleBox
from traffic_trajectory.rtdetr import RTDetrDetector
from traffic_trajectory.scenario import parse_scenario
from traffic_trajectory.video import decoded_frames, open_video

FULL_FLOOR = 0.03
SEED_FLOOR = 0.06
STRONG = 0.20
CROP_SIDES = (64, 96)
CROP_BUDGET = 4
GATE_PX = 16
TRACKERS = ("point", "ocsort")


def _window(anchor: Point, side: int, frame: MatLike) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    side_x, side_y = min(side, width), min(side, height)
    x = max(0, min(width - side_x, round(anchor[0] - side_x / 2)))
    y = max(0, min(height - side_y, round(anchor[1] - side_y / 2)))
    return x, y, x + side_x, y + side_y


def _recover(
    detector: BoxDetector,
    frame: MatLike,
    seeds: list[VehicleBox],
    current: list[VehicleBox],
    lane: ConvexPolygon,
    side: int,
) -> tuple[list[VehicleBox], int]:
    strong = [box for box in current if box.score > STRONG]
    added = []
    centers = []
    for seed in seeds:
        anchor = (seed.position[0], seed.y + seed.height / 2)
        if max(seed.width, seed.height) > side / 2:
            continue
        if any(dist(seed.position, box.position) <= GATE_PX for box in strong):
            continue
        if any(dist(anchor, center) <= side / 4 for center in centers):
            continue
        centers.append(anchor)
        candidates = [
            box
            for box in detector._window_boxes(frame, _window(anchor, side, frame))
            if box.score > STRONG
            and lane.contains(box.position)
            and dist(seed.position, box.position) <= GATE_PX
            and 0.5 * seed.width <= box.width <= 2 * seed.width
            and 0.5 * seed.height <= box.height <= 2 * seed.height
        ]
        if candidates:
            added.append(max(candidates, key=lambda box: box.score))
        if len(centers) == CROP_BUDGET:
            break
    return [*current, *_suppress(added, STRONG)], len(centers)


def _reference_samples() -> list[dict]:
    return read_json(DATA / "fog.manual-trace.json")["tracks"][0]["observations"]


def _reference_hits(frames: list[list[VehicleBox]], fps: float, cutoff: float) -> list[bool]:
    return [
        any(
            box.score > cutoff and dist(tuple(sample["position"]), box.position) <= 12
            for box in frames[round(sample["timestamp_s"] * fps)]
        )
        for sample in _reference_samples()
    ]


def _reference_crop_check(detector: BoxDetector, video: Path) -> dict:
    """Measure the detector with a known search center, outside the alert pipeline."""
    with open_video(video) as (capture, fps):
        samples = {
            round(sample["timestamp_s"] * fps): tuple(sample["position"])
            for sample in _reference_samples()
        }
        scores = {side: [] for side in CROP_SIDES}
        for index, frame in enumerate(decoded_frames(capture, strict_length=True)):
            if index not in samples:
                continue
            anchor = samples[index]
            for side in CROP_SIDES:
                boxes = detector._window_boxes(frame, _window(anchor, side, frame))
                match = max(
                    (box.score for box in boxes if dist(anchor, box.position) <= 12),
                    default=None,
                )
                scores[side].append(match)
    return scores


def run_case(detector: BoxDetector, case: Case) -> dict:
    lane = parse_scenario({**read_json(case.scene), "tracks": []}).lane.polygon
    full = []
    variants = [
        (source, side) for source in ("current_frame", "prior_frame") for side in CROP_SIDES
    ]
    rescued = {variant: [] for variant in variants}
    extra_ms = {variant: [] for variant in variants}
    calls = {variant: 0 for variant in variants}
    full_ms = []
    seeds = []
    with open_video(case.video) as (capture, fps):
        for frame in decoded_frames(capture, strict_length=True):
            if not full:
                for _ in range(5):
                    detector.predict_boxes(frame)
            start = perf_counter()
            current = detector.predict_boxes(frame)
            full_ms.append((perf_counter() - start) * 1000)
            full.append(current)
            current_seeds = _suppress(
                [box for box in current if lane.contains(box.position)], SEED_FLOOR
            )
            for source, side in variants:
                variant = source, side
                start = perf_counter()
                guides = seeds if source == "prior_frame" else current_seeds
                combined, count = _recover(detector, frame, guides, current, lane, side)
                extra_ms[variant].append((perf_counter() - start) * 1000)
                rescued[variant].append(combined)
                calls[variant] += count
            seeds = current_seeds

    result = {
        "fps": fps,
        "frames": len(full),
        "video_sha256": sha256(case.video),
        "mean_full_ms": mean(full_ms),
        "runs": {},
    }
    runs = [
        ("full_0.20", full, STRONG, None),
        ("full_0.03", full, FULL_FLOOR, None),
        *(
            (f"{src}_crop_{side}", rescued[src, side], STRONG, (src, side))
            for src, side in variants
        ),
    ]
    for name, boxes, cutoff, variant in runs:
        result["runs"][name] = {
            "reference_hits": _reference_hits(boxes, fps, cutoff) if case.name == "fog" else None,
            "crop_calls": calls[variant] if variant is not None else 0,
            "extra_ms_per_frame": mean(extra_ms[variant]) if variant is not None else 0,
            "added_boxes": sum(len(b) - len(a) for a, b in zip(full, boxes, strict=True)),
            "alerts": {
                tracker: replay(case, fps, boxes, tracker, cutoff, FULL_FLOOR)
                for tracker in TRACKERS
            },
        }
    if case.name == "fog":
        result["reference_centered_crop_scores"] = _reference_crop_check(detector, case.video)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("road_folder", type=Path)
    parser.add_argument("night_video", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    detector = RTDetrDetector(args.model, FULL_FLOOR, device=args.device)
    report = {
        "code_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_sha256": sha256(args.model / "model.safetensors"),
        "device": args.device,
        "hardware": detector.torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "platform": platform.platform(),
        "settings": {
            "full_floor": FULL_FLOOR,
            "seed_floor": SEED_FLOOR,
            "strong": STRONG,
            "crop_sides": CROP_SIDES,
            "crop_budget": CROP_BUDGET,
            "gate_px": GATE_PX,
        },
        "cases": {
            case.name: run_case(detector, case)
            for case in cases(args.road_folder, args.night_video)
            if case.name in ("fog", "red", "night-negative")
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
