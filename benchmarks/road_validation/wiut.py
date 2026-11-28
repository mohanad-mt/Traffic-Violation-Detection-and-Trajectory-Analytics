"""Compare Team404 and the project on the same four fixed-camera videos."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import cv2

from benchmarks.assets import sha256
from benchmarks.road_validation.compare import DATA, read_json
from traffic_trajectory.detection import make_detector
from traffic_trajectory.detector import detect_events
from traffic_trajectory.evaluate import evaluate
from traffic_trajectory.scenario import parse_scenario
from traffic_trajectory.video import open_video, track_video

REVISION = "d12fe4ed33f48906b6dae1a086a56a97529561b3"
WEIGHT_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
CASES = (
    ("wrong", "wrong.avi", "wrong_way", 1.0),
    ("shadow", "second.avi", "wrong_way", 1.0),
    ("stop", "stop.avi", "stopped_vehicle", 1.0),
    ("red", "red.avi", "red_light", 0.5),
)
SIGNAL_CROP = (331, 14, 345, 37)


def _score(events: list[dict], labels: dict, tolerance: float, duration: float) -> dict:
    return {
        "events": events,
        "metrics": evaluate({"events": events}, labels, tolerance, duration_s=duration)["overall"],
    }


def _score_team(
    segments: list[list], category: str, labels: dict, tolerance: float, duration: float
) -> dict:
    (label,) = labels["events"]
    events = [
        {
            "type": label["type"],
            "lane_id": label["lane_id"],
            "start_s": start,
            "confirmed_s": start,
            "end_s": end,
        }
        for start, end, kind in segments
        if kind == category
    ]
    result = _score(events, labels, tolerance, duration)
    # The batch API gives onset and end, with no alert emission timestamp.
    del result["metrics"]["mean_confirmation_delay_s"]
    for event in result["events"]:
        del event["confirmed_s"]
    return result


def _team_red_segments(
    team, video: str, config: dict, intervals: tuple[tuple[float, float], ...]
) -> tuple[list[list], list[list]]:
    shared_phase = []

    class PhaseReplayAnalyzer(team.EventAnalyzer):
        def events(self, duration: float) -> list[list]:
            from_pixels = super().events(duration)
            # Reuse the same tracks and replace only the signal state at each video timestamp.
            for state in self.states:
                state.red = any(start <= state.t_sec < end for start, end in intervals)
            shared_phase.extend(super().events(duration))
            return from_pixels

    with patch.object(team, "EventAnalyzer", PhaseReplayAnalyzer):
        from_pixels = team.analyze_video(video, config)
    return from_pixels, shared_phase


def run(checkout: Path, footage: Path, weights: Path, device: str) -> dict:
    checkout = checkout.resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True)
    if head.strip() != REVISION or sha256(checkout / "weights/yolo11n.pt") != WEIGHT_SHA:
        raise ValueError("expected the pinned Team404 release and YOLO11n weights")
    sys.path.insert(0, str(checkout))
    try:
        team = importlib.import_module("src.baseline")
    finally:
        sys.path.pop(0)
    cv2.setNumThreads(4)
    detector = make_detector("rfdetr", weights, 0.20, 0.03, device=device)
    results = {}
    for name, filename, category, tolerance in CASES:
        print(f"Replaying {filename}", file=sys.stderr, flush=True)
        video = footage / filename
        document = read_json(DATA / f"{name}.scene.json")
        scene = parse_scenario({**document, "tracks": []})
        labels = read_json(DATA / f"{name}.labels.json")
        (label,) = labels["events"]
        with open_video(video) as (capture, fps):
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        def normalize(points):
            return [[x / width, y / height] for x, y in points]

        polygon = normalize(document["lane"]["polygon"])
        dx, dy = document["lane"]["forward"]
        config = {
            **team.DEFAULT_CONFIG,
            "device": "0" if device == "cuda" else "cpu",
            "frame_stride": 1,
            "temporal_enabled": False,
            "temporal_required": False,
            "road_polygon": polygon,
            "lanes": [{"polygon": polygon, "direction": [dx / width, dy / height]}],
        }
        if name == "red":
            x1, y1, x2, y2 = SIGNAL_CROP
            config["traffic_light_roi"] = [x1 / width, y1 / height, x2 / width, y2 / height]
            config["stop_line"] = normalize(document["stop_line"]["segment"])
        intervals = scene.stop_line.red_intervals_s if scene.stop_line is not None else None
        video_path = str(video.resolve())
        with redirect_stdout(sys.stderr):
            if intervals is None:
                segments = team.analyze_video(video_path, config)
            else:
                segments, shared_phase = _team_red_segments(team, video_path, config, intervals)
        tracks = track_video(
            video,
            detector,
            scene.thresholds.max_gap_s,
            24,
            tracker_name="bytetrack",
            confidence=0.20,
            strict_length=True,
        )
        project = [
            asdict(event)
            for event in detect_events(replace(scene, tracks=tracks))
            if event.type == label["type"]
        ]
        results[name] = {
            "video_sha256": sha256(video),
            "fps": fps,
            "duration_s": duration,
            "onset_tolerance_s": tolerance,
            "label": label,
            "team404": _score_team(segments, category, labels, tolerance, duration),
            "project": _score(project, labels, tolerance, duration),
        }
        if intervals is not None:
            results[name]["shared_red_intervals_s"] = intervals
            results[name]["team404_shared_phase"] = _score_team(
                shared_phase, category, labels, tolerance, duration
            )
    return {
        "team404_revision": REVISION,
        "team404_yolo11n_sha256": WEIGHT_SHA,
        "project_rfdetr_sha256": sha256(weights),
        "red_signal_crop_xyxy": SIGNAL_CROP,
        "device": device,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team404_checkout", type=Path)
    parser.add_argument("road_folder", type=Path)
    parser.add_argument("rfdetr_weights", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    offline = {
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
        "HF_HUB_OFFLINE": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    }
    with patch.dict(os.environ, offline):
        result = run(args.team404_checkout, args.road_folder, args.rfdetr_weights, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
