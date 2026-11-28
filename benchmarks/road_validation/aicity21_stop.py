"""Replay time-only stop comparisons on AI City 2021 Track 4 training clips."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import cv2

from benchmarks.assets import sha256
from traffic_trajectory.detection import make_detector
from traffic_trajectory.detector import detect_events
from traffic_trajectory.evaluate import evaluate
from traffic_trajectory.scenario import parse_scenario
from traffic_trajectory.tracker import make_tracker
from traffic_trajectory.video import timed_video

VIDEO_SHA256 = "c6451f72bc4e67132600f8ebc30f950de207a47a8ec65cde910f3bdff20f59e1"
STRIDE = 6
TOLERANCE_S = 10
EVENT_TYPE = "stopped_vehicle_candidate"
SCENE = {
    "lanes": [
        {
            "id": "far-roadway",
            "polygon": [[0, 176], [765, 28], [768, 45], [0, 303]],
            "forward": [-1, 0.3],
        },
        {
            "id": "near-roadway",
            "polygon": [[0, 310], [775, 38], [795, 37], [371, 409], [0, 409]],
            "forward": [1, -0.5],
        },
    ],
    "tracks": [],
}


@dataclass(frozen=True)
class StopClip:
    checksum: str
    onsets_s: tuple[int, ...]
    scene: dict

    def labels(self) -> dict:
        return {
            "events": [
                {"type": EVENT_TYPE, "lane_id": "all", "start_s": onset} for onset in self.onsets_s
            ]
        }


CLIPS = {
    1: StopClip(
        "92d2f1daa0dc44c0a997acc78f4ad08c4e8c2b03f930f6db9d3ff8ff9f85dcad",
        (),
        {
            "lanes": [
                {
                    "id": "roadway",
                    "polygon": [[0, 275], [669, 28], [756, 100], [620, 409], [0, 409]],
                    "forward": [1, -0.5],
                }
            ],
            "tracks": [],
        },
    ),
    2: StopClip(VIDEO_SHA256, (587,), SCENE),
}


def run(video: Path, checkpoint: Path, device: str, clip_id: int = 2) -> dict:
    clip = CLIPS[clip_id]
    if sha256(video) != clip.checksum:
        raise ValueError(f"expected the original AI City 2021 Track 4 train-data {clip_id}.mp4")
    cv2.setNumThreads(4)
    scene = parse_scenario(clip.scene)
    detector = make_detector("rfdetr", checkpoint, 0.20, 0.03, device=device)
    sampled = 0
    with timed_video(video, timestamp_source="pts", strict_length=True) as (fps, frames):
        tracker = make_tracker("bytetrack", fps / STRIDE, scene.thresholds.max_gap_s, 24, 0.20)
        for index, (timestamp_s, frame) in enumerate(frames):
            if index % STRIDE == 0:
                tracker.update_frame(timestamp_s, detector(frame))
                sampled += 1
    duration_s = timestamp_s + 1 / fps
    scene = replace(scene, tracks=tracker.tracks())
    observations = {
        track.id: {sample.timestamp_s: sample for sample in track.observations}
        for track in scene.tracks
    }
    runs = {}
    for duration, drift in product((3.0, 10.0, 20.0), (0.0, 0.75)):
        thresholds = replace(
            scene.thresholds, stopped_duration_s=duration, max_stopped_drift_px_s=drift
        )
        configured = replace(scene, thresholds=thresholds)
        events = [asdict(event) for event in detect_events(configured) if event.type == EVENT_TYPE]
        # The original annotation supplies onset time and has no roadway or track ID.
        predictions = {"events": [{**event, "lane_id": "all"} for event in events]}
        report = evaluate(predictions, clip.labels(), TOLERANCE_S, duration_s=duration_s)
        anchors = [observations[e["track_id"]][e["start_s"]] for e in events]
        matched = {match["prediction_index"] for match in report["matches"]}
        y_bins = Counter(
            int(anchor.position[1] // 50) * 50
            for index, anchor in enumerate(anchors)
            if index not in matched
        )
        runs[f"{duration:g}:{drift:g}"] = {
            "overall": report["overall"],
            "matched_onsets": [
                {
                    **events[i],
                    "position": anchors[i].position,
                    "box": asdict(anchors[i].box) if anchors[i].box is not None else None,
                }
                for i in sorted(matched)
            ],
            "unmatched_by_anchor_y_bin_px": dict(sorted(y_bins.items())),
        }
    return {
        "clip_id": clip_id,
        "video_sha256": clip.checksum,
        "checkpoint_sha256": sha256(checkpoint),
        "scene": clip.scene,
        "label_onsets_s": clip.onsets_s,
        "onset_tolerance_s": TOLERANCE_S,
        "sampled_fps": fps / STRIDE,
        "sampled_frames": sampled,
        "duration_s": duration_s,
        "settings_key": "stopped_duration_s:max_stopped_drift_px_s",
        "runs": runs,
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
