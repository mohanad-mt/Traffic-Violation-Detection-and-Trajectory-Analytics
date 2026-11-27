"""Replay common detections through the trackers and event rules on development clips."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2

from benchmarks.assets import sha256
from traffic_trajectory.detection import DETECTORS, BoxDetector, make_detector
from traffic_trajectory.detector import detect_events
from traffic_trajectory.evaluate import evaluate
from traffic_trajectory.geometry import VehicleBox
from traffic_trajectory.scenario import parse_scenario, track_record
from traffic_trajectory.tracker import FrameDetections, make_tracker
from traffic_trajectory.video import decoded_frames, open_video

DATA = Path(__file__).resolve().parent
TRACK_START = (0.08, 0.20, 0.35)
CONTINUATION = 0.03


@dataclass(frozen=True)
class Case:
    name: str
    video: Path
    scene: Path
    labels: Path
    tolerance: float
    radius_px: float | None = None


def cases(folder: Path, night_video: Path) -> tuple[Case, ...]:
    night = DATA.parent / "nighttime"
    return (
        Case(
            "fog",
            folder / "wrong.avi",
            DATA / "wrong.scene.json",
            DATA / "wrong.labels.json",
            1.0,
            12.0,
        ),
        Case(
            "shadow",
            folder / "second.avi",
            DATA / "shadow.scene.json",
            DATA / "shadow.labels.json",
            1.0,
        ),
        Case(
            "highway-stop",
            folder / "stop.avi",
            DATA / "stop.scene.json",
            DATA / "stop.labels.json",
            1.0,
        ),
        Case(
            "red",
            folder / "red.avi",
            DATA / "red.scene.json",
            DATA / "red.identity-labels.json",
            0.5,
            12.0,
        ),
        Case("night-negative", night_video, night / "scene.json", night / "labels.json", 0.5),
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def observations(detector: BoxDetector, video: Path) -> tuple[float, list, list[float]]:
    frames, times = [], []
    with open_video(video) as (capture, fps):
        for frame in decoded_frames(capture, strict_length=True):
            if not frames:
                for _ in range(5):
                    detector.predict_boxes(frame)
            start = perf_counter()
            frames.append(detector.predict_boxes(frame))
            times.append((perf_counter() - start) * 1000)
    return fps, frames, times


def replay(
    case: Case,
    fps: float,
    detections: list[list[VehicleBox]],
    tracker_name: str,
    cutoff: float,
    weak_cutoff: float,
) -> dict:
    scene = parse_scenario({**read_json(case.scene), "tracks": []})
    labels = read_json(case.labels)
    if case.name == "fog":
        labels["events"][0]["trajectory"] = read_json(DATA / "fog.manual-trace.json")["tracks"][0][
            "observations"
        ]
    start = perf_counter()
    tracker = make_tracker(tracker_name, fps, scene.thresholds.max_gap_s, 24, cutoff)
    for index, boxes in enumerate(detections):
        tracker.update_frame(
            index / fps,
            FrameDetections.from_boxes([box for box in boxes if box.score > weak_cutoff], cutoff),
        )
    tracks = tracker.tracks()
    events = detect_events(replace(scene, tracks=tracks))
    replay_ms = (perf_counter() - start) * 1000
    report = evaluate(
        {
            "events": [asdict(event) for event in events],
            "tracks": [track_record(track) for track in tracks],
        },
        labels,
        case.tolerance,
        identity_required=case.radius_px is not None,
        radius_px=case.radius_px,
        duration_s=len(detections) / fps,
    )
    return {
        "tracker": tracker_name,
        "track_start": cutoff,
        "continuation": weak_cutoff,
        "track_ids": len(tracks),
        "observations": sum(len(track.observations) for track in tracks),
        "replay_ms_per_frame": replay_ms / len(detections),
        "events": [asdict(event) for event in events],
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("road_folder", type=Path)
    parser.add_argument("night_video", type=Path)
    parser.add_argument("--detector", choices=DETECTORS, default="yolox")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input-color", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument(
        "--tracker", choices=("point", "bytetrack", "ocsort"), nargs="+", default=["point"]
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    detector = make_detector(
        args.detector, args.model, CONTINUATION, device=args.device, input_color=args.input_color
    )
    report = {
        "detector": args.detector,
        "device": args.device,
        "split": "development",
        "cases": {},
    }
    for case in cases(args.road_folder, args.night_video):
        fps, boxes, ms = observations(detector, case.video)
        if args.cache_dir is not None:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            path = args.cache_dir / f"{case.name}.json"
            with path.open("w", encoding="utf-8") as stream:
                json.dump(
                    {"fps": fps, "frames": [[asdict(box) for box in frame] for frame in boxes]},
                    stream,
                )
        report["cases"][case.name] = {
            "video_sha256": sha256(case.video),
            "frames": len(boxes),
            "fps": fps,
            "mean_detector_ms": mean(ms),
            "runs": [
                replay(case, fps, boxes, tracker, cutoff, weak)
                for tracker in args.tracker
                for cutoff in TRACK_START
                for weak in (cutoff, CONTINUATION)
            ],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
