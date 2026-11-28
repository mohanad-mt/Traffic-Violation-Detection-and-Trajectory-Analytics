"""Compare FP32 YOLOX scores and video latency on OpenCV and ONNX Runtime CUDA."""

import argparse
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from benchmarks.assets import sha256
from traffic_trajectory.detector import detect_events
from traffic_trajectory.scenario import parse_scenario
from traffic_trajectory.video import decoded_frames, open_video, track_video
from traffic_trajectory.yolox import YoloXDetector


def compare(model: Path, video: Path, scene: Path) -> dict:
    cv2.setNumThreads(4)
    scenario = parse_scenario({**json.loads(scene.read_text(encoding="utf-8-sig")), "tracks": []})
    raw, runs, event_sets = [], {}, []
    with open_video(video) as (capture, _):
        frames = decoded_frames(capture)
        first = next(frames)
    for device in ("cpu", "cuda"):
        detector = YoloXDetector(model, 0.35, device=device)
        detector.net.setInput(cv2.dnn.blobFromImage(first, size=(640, 640), swapRB=True))
        raw.append(detector.net.forward())
        for _ in range(10):
            detector.predict_boxes(first)
        frame_ms = []

        def detect(frame):
            start = perf_counter()
            result = detector(frame)
            frame_ms.append((perf_counter() - start) * 1000)
            return result

        start = perf_counter()
        tracks = track_video(video, detect, scenario.thresholds.max_gap_s, 24, strict_length=True)
        events = detect_events(replace(scenario, tracks=tracks))
        elapsed = perf_counter() - start
        event_sets.append([asdict(event) for event in events])
        runs[device] = {
            "frames": len(frame_ms),
            "mean_detector_ms": mean(frame_ms),
            "end_to_end_fps": len(frame_ms) / elapsed,
            "track_count": len(tracks),
            "observations": sum(len(track.observations) for track in tracks),
            "events": event_sets[-1],
        }
    # Small GPU rounding differences can affect interpolation below a frame interval.
    paired = all(
        a["type"] == b["type"]
        and a["track_id"] == b["track_id"]
        and abs(a["start_s"] - b["start_s"]) <= 0.001
        and abs(a["confirmed_s"] - b["confirmed_s"]) <= 0.001
        for a, b in zip(*event_sets)
    )
    return {
        "model_sha256": sha256(model),
        "video_sha256": sha256(video),
        "resolution": list(first.shape[1::-1]),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "onnxruntime": version("onnxruntime-gpu"),
        "raw_max_abs_error": float(np.max(np.abs(raw[0] - raw[1]))),
        "same_events_within_1ms": paired and len(event_sets[0]) == len(event_sets[1]),
        "backends": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.model, args.video, args.scene), indent=2))
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
