"""Compare a background-based stall detector on AI City 2021 training clips."""

import argparse
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from cv2.typing import MatLike

from benchmarks.assets import sha256
from benchmarks.road_validation.aicity21_stop import CLIPS, EVENT_TYPE, STRIDE, TOLERANCE_S
from traffic_trajectory.detection import BoxDetector, make_detector
from traffic_trajectory.evaluate import evaluate
from traffic_trajectory.geometry import VehicleBox
from traffic_trajectory.scenario import parse_scenario
from traffic_trajectory.video import timed_video

DETECTION_STRIDE = 5
BACKGROUND_HISTORY = 20
BACKGROUND_VARIANCE = 16
CONFIDENCE = 0.20
TUBE_IOU = 0.50
MAX_MISSED_SECONDS = 3.0
MIN_TUBE_SECONDS = 30.0
TRACEBACK_SECONDS = 30.0
APPEARANCE_PSNR = 20.0
MIN_SUPPORT = 0.60


@dataclass
class Tube:
    anchor: VehicleBox
    samples: dict[int, VehicleBox] = field(default_factory=dict)


def _overlap(a: VehicleBox, b: VehicleBox) -> float:
    width = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    height = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    intersection = width * height
    return intersection / (a.width * a.height + b.width * b.height - intersection)


def _background_views(frames: Iterable[MatLike]) -> Iterator[tuple[int, MatLike]]:
    model = cv2.createBackgroundSubtractorMOG2(BACKGROUND_HISTORY, BACKGROUND_VARIANCE, False)
    for index, frame in enumerate(frames):
        model.apply(frame)
        if index % DETECTION_STRIDE == 0:
            yield index, model.getBackgroundImage()


def _background_boxes(
    frames: list[MatLike], detector: BoxDetector, scene: dict
) -> list[dict[int, list[VehicleBox]]]:
    lanes = [zone.lane for zone in parse_scenario(scene).zones]
    passes = []
    for reverse in (True, False):
        detections = {}
        source = reversed(frames) if reverse else iter(frames)
        for index, view in _background_views(source):
            frame_index = len(frames) - 1 - index if reverse else index
            detections[frame_index] = [
                box
                for box in detector.predict_boxes(view)
                if any(lane.polygon.contains(box.position) for lane in lanes)
            ]
        passes.append(detections)
    return passes


def _make_tubes(detections: dict[int, list[VehicleBox]], times: list[float]) -> Iterator[Tube]:
    active: list[Tube] = []
    for index, boxes in sorted(detections.items()):
        surviving = []
        for tube in active:
            last = next(reversed(tube.samples))
            if times[index] - times[last] > MAX_MISSED_SECONDS:
                yield tube
            else:
                surviving.append(tube)
        active = surviving
        assigned = set()
        for box in sorted(boxes, key=lambda item: item.score, reverse=True):
            overlap, match = max(
                (
                    (_overlap(box, tube.anchor), slot)
                    for slot, tube in enumerate(active)
                    if slot not in assigned
                ),
                default=(0.0, -1),
            )
            if overlap < TUBE_IOU:
                match = len(active)
                active.append(Tube(box))
            active[match].samples[index] = box
            assigned.add(match)
    yield from active


def _assess_tube(
    tube: Tube,
    frames: list[MatLike],
    times: list[float],
    forward_boxes: dict[int, list[VehicleBox]],
) -> dict | None:
    start, end = next(iter(tube.samples)), next(reversed(tube.samples))
    if times[end] - times[start] < MIN_TUBE_SECONDS:
        return None
    expected_samples = (end - start) // DETECTION_STRIDE + 1
    if len(tube.samples) < MIN_SUPPORT * expected_samples:
        return None
    if not any(
        _overlap(tube.anchor, box) >= TUBE_IOU
        for index in range(start, end + 1)
        for box in forward_boxes.get(index, ())
    ):
        return None

    # Compare fixed crops so changes measure appearance, with no box jitter.
    best_index = max(tube.samples, key=lambda index: tube.samples[index].score)
    box = tube.samples[best_index]
    left, top = int(box.x), int(box.y)
    right, bottom = int(np.ceil(box.x + box.width)), int(np.ceil(box.y + box.height))
    reference = frames[best_index][top:bottom, left:right]
    run = 0
    onset = start
    for index in range(max(0, start - round(TRACEBACK_SECONDS * 5)), end + 1):
        patch = frames[index][top:bottom, left:right]
        if cv2.PSNR(reference, patch) >= APPEARANCE_PSNR:
            run += 1
            if run == 3:
                onset = index - 2
                break
        else:
            run = 0
    confirmation = next(
        index for index in tube.samples if times[index] - times[start] >= MIN_TUBE_SECONDS
    )
    return {
        "type": EVENT_TYPE,
        "lane_id": "all",
        "start_s": times[onset],
        "confirmed_s": max(times[confirmation], times[onset]),
        "tube_start_s": times[start],
        "tube_end_s": times[end],
        "box": asdict(box),
    }


def _deduplicate(events: list[dict]) -> list[dict]:
    # Keep the longer tube for duplicate events supported by the same image region.
    events.sort(key=lambda event: event["tube_end_s"] - event["tube_start_s"], reverse=True)
    kept = []
    for event in events:
        if not any(
            abs(event["start_s"] - other["start_s"]) <= TOLERANCE_S
            and _overlap(VehicleBox(**event["box"]), VehicleBox(**other["box"])) >= TUBE_IOU
            for other in kept
        ):
            kept.append(event)
    return sorted(kept, key=lambda event: event["start_s"])


def run(video: Path, checkpoint: Path, device: str, clip_id: int = 2) -> dict:
    clip = CLIPS[clip_id]
    if sha256(video) != clip.checksum:
        raise ValueError(f"expected the original AI City 2021 Track 4 train-data {clip_id}.mp4")
    cv2.setNumThreads(4)
    started = perf_counter()
    detector = make_detector("rfdetr", checkpoint, CONFIDENCE, device=device)
    times, frames = [], []
    with timed_video(video, timestamp_source="pts", strict_length=True) as (fps, source):
        for index, (timestamp_s, frame) in enumerate(source):
            if index % STRIDE == 0:
                times.append(timestamp_s)
                frames.append(frame)
    duration_s = timestamp_s + 1 / fps
    backward, forward = _background_boxes(frames, detector, clip.scene)
    candidates = [
        event
        for tube in _make_tubes(backward, times)
        if (event := _assess_tube(tube, frames, times, forward)) is not None
    ]
    events = _deduplicate(candidates)
    report = evaluate({"events": events}, clip.labels(), TOLERANCE_S, duration_s=duration_s)
    # Reverse modeling uses future frames. The tube confirmation is not live alert latency.
    del report["overall"]["mean_confirmation_delay_s"]
    return {
        "method": "Team76-inspired offline stop detector",
        "clip_id": clip_id,
        "video_sha256": clip.checksum,
        "checkpoint_sha256": sha256(checkpoint),
        "device": device,
        "duration_s": duration_s,
        "background_fps": fps / STRIDE,
        "detection_fps": fps / STRIDE / DETECTION_STRIDE,
        "background_history_frames": BACKGROUND_HISTORY,
        "background_variance": BACKGROUND_VARIANCE,
        "detector_confidence": CONFIDENCE,
        "tube_iou": TUBE_IOU,
        "tube_max_gap_s": MAX_MISSED_SECONDS,
        "tube_min_duration_s": MIN_TUBE_SECONDS,
        "tube_min_support": MIN_SUPPORT,
        "appearance_traceback_s": TRACEBACK_SECONDS,
        "appearance_psnr_db": APPEARANCE_PSNR,
        "scene": clip.scene,
        "label_onsets_s": clip.onsets_s,
        "onset_tolerance_s": TOLERANCE_S,
        "elapsed_s": perf_counter() - started,
        "candidates_before_deduplication": len(candidates),
        "events": events,
        "overall": report["overall"],
        "matches": report["matches"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--clip", type=int, choices=CLIPS, default=2)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.video, args.checkpoint, args.device, args.clip), indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
