"""Replay RoadTrack and the project's stop detector on the same released boxes."""

import argparse
import importlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import chdir, redirect_stdout
from dataclasses import dataclass, replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from unittest.mock import patch

from traffic_trajectory.detector import StoppedVehicleEvent, detect_events
from traffic_trajectory.evaluate import _one_to_one
from traffic_trajectory.geometry import VehicleBox
from traffic_trajectory.scenario import parse_scenario
from traffic_trajectory.tracker import FrameDetections, make_tracker

REVISION = "1a72b2613cf4e06725ba25d72ca7aaddb6338772"
FPS = 10
CONFIDENCE = 0.30
IOU = 0.50
CAMERAS = {
    "K204": (9.2, 383),
    "K211": (12.6, 364),
    "K221": (9.9, 399),
    "K249": (6.4, 322),
    "K287": (9.7, 199),
    "K295": (12.3, 443),
}
SCENE = parse_scenario(
    {
        "lanes": [
            {
                "id": "camera",
                "polygon": [[0, 0], [8192, 0], [8192, 8192], [0, 8192]],
                "forward": [1, 0],
            }
        ],
        "tracks": [],
    }
)
type Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Alert:
    frame: int
    box: Box
    track_id: str


def _select(upstream, per_camera: int) -> list[tuple[str, Path, str]]:
    clips = []
    for group in ("P", "N"):
        counts = defaultdict(int)
        for folder in sorted((Path("datasets") / group).iterdir()):
            camera = upstream.extract_sequence_name(folder.name)
            if not folder.is_dir() or camera not in CAMERAS:
                raise ValueError(f"unexpected RoadTrack entry: {folder}")
            if per_camera and counts[camera] >= per_camera:
                continue
            counts[camera] += 1
            clips.append((group, folder, camera))
    if any(
        sum(c == camera and g == group for g, _, c in clips) < max(1, per_camera)
        for camera in CAMERAS
        for group in ("P", "N")
    ):
        raise ValueError("the checkout lacks sequences for the requested groups and cameras")
    return clips


def _roadtrack_alerts(tracks: dict) -> list[Alert]:
    alerts = []
    max_gap = round(FPS * SCENE.thresholds.stopped_realert_gap_s)
    for track_id, observations in tracks.items():
        previous = None
        for frame, box, _ in observations:
            if previous is None or frame - previous > max_gap:
                alerts.append(Alert(frame, box, str(track_id)))
            previous = frame
    return alerts


def _project_alerts(detections: dict, frames: int) -> list[Alert]:
    tracker = make_tracker("bytetrack", FPS, SCENE.thresholds.max_gap_s, 24, CONFIDENCE)
    for frame in range(1, frames + 1):
        boxes = [
            VehicleBox(x1, y1, x2 - x1, y2 - y1, confidence)
            for (x1, y1, x2, y2), confidence, _ in detections.get(frame, ())
            if x2 > x1 and y2 > y1 and 0 <= confidence <= 1
        ]
        tracker.update_frame((frame - 1) / FPS, FrameDetections.from_boxes(boxes, CONFIDENCE))
    tracks = tracker.tracks()
    positions = {
        track.id: {sample.timestamp_s: sample.box for sample in track.observations}
        for track in tracks
    }
    alerts = []
    for event in detect_events(replace(SCENE, tracks=tracks)):
        if isinstance(event, StoppedVehicleEvent):
            box = positions[event.track_id][event.confirmed_s]
            if box is None:
                raise ValueError("expected a source box for the confirmation frame")
            alerts.append(
                Alert(
                    round(event.confirmed_s * FPS) + 1,
                    (box.x, box.y, box.width, box.height),
                    event.track_id,
                )
            )
    return alerts


def _miss_reason(upstream, reference: dict, alerts: list[Alert], tracks: dict) -> str:
    onset = min(reference)
    for alert in alerts:
        for frame, box, _ in tracks[int(alert.track_id)]:
            if (
                frame > alert.frame
                and frame in reference
                and upstream.compute_iou(reference[frame], box) >= IOU
            ):
                if alert.frame < onset:
                    return "announced_before_label"
                return "box_aligned_after_confirmation"
    return "no_later_overlap"


def _matches(upstream, labels: dict, alerts: list[Alert], *, tracks: dict | None = None) -> dict:
    alerts.sort(key=lambda alert: (alert.frame, alert.track_id))
    references = [{frame: box for frame, box, _ in rows} for rows in labels.values()]
    overlaps = [
        [
            index
            for index, alert in enumerate(alerts)
            if alert.frame in ref and upstream.compute_iou(ref[alert.frame], alert.box) >= IOU
        ]
        for ref in references
    ]
    matched = _one_to_one(overlaps)
    report = {
        "tp": len(matched),
        "fp": len(alerts) - len(matched),
        "fn": len(references) - len(matched),
        "delay_s": [
            (alerts[index].frame - min(references[ref])) / FPS for ref, index in matched.items()
        ],
    }
    if tracks is not None:
        report["miss_diagnostics"] = dict(
            Counter(
                "competing_match"
                if overlaps[i]
                else _miss_reason(upstream, reference, alerts, tracks)
                for i, reference in enumerate(references)
                if i not in matched
            )
        )
    return report


def _replay(upstream, group: str, folder: Path, camera: str, filters: dict) -> dict:
    outputs = {}

    def capture(tracker, frames, boxes, pixel_ratio):
        tracks, frame_alerts = upstream_detector(tracker, frames, boxes, pixel_ratio)
        outputs.update(frames=frames, boxes=boxes, tracks=tracks)
        return tracks, frame_alerts

    upstream_detector = upstream.detect_by_speed
    with patch.object(upstream, "detect_by_speed", capture):
        with redirect_stdout(sys.stderr):
            tp, fn, fp, latency_frames = upstream.eval(
                folder.name, filters, CAMERAS, str(folder.parent), group == "P"
            )
    labels = upstream.load_gt(folder / "data" / "gt" / "gt.txt")[0] if group == "P" else {}
    frames = outputs["frames"]
    return {
        "clip": folder.name,
        "camera": camera,
        "group": group,
        "frames": frames,
        "labels": len(labels),
        "roadtrack_author_score": {"tp": tp, "fp": fp, "fn": fn, "latency_frames": latency_frames},
        "roadtrack": _matches(
            upstream, labels, _roadtrack_alerts(outputs["tracks"]), tracks=outputs["tracks"]
        ),
        "project": _matches(upstream, labels, _project_alerts(outputs["boxes"], frames)),
    }


def _summarize(clips: list[dict], method: str) -> dict:
    tp, fp, fn = (sum(clip[method][key] for clip in clips) for key in ("tp", "fp", "fn"))
    negative_seconds = sum(clip["frames"] for clip in clips if clip["group"] == "N") / FPS
    negative_alerts = sum(clip[method]["fp"] for clip in clips if clip["group"] == "N")
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "negative_alerts": negative_alerts,
        "negative_false_alerts_per_hour": negative_alerts * 3600 / negative_seconds,
    }
    if method == "roadtrack_author_score":
        result["mean_author_latency_s"] = (
            sum(clip[method]["latency_frames"] for clip in clips) / (FPS * tp) if tp else None
        )
    else:
        delays = [d for clip in clips for d in clip[method]["delay_s"]]
        result["median_matched_delay_s"] = median(delays) if delays else None
        result["mean_matched_delay_s"] = mean(delays) if delays else None
    if method == "roadtrack":
        reasons = Counter()
        for clip in clips:
            reasons.update(clip[method]["miss_diagnostics"])
        result["miss_diagnostics"] = dict(sorted(reasons.items()))
    return result


def _run_clip(repository: Path, filters: dict, clip: tuple[str, Path, str]) -> dict:
    upstream = importlib.import_module("run_roadtrack")
    with chdir(repository):
        return _replay(upstream, *clip, filters)


def _progress(results, total: int) -> list[dict]:
    clips = []
    for index, clip in enumerate(results, 1):
        clips.append(clip)
        if total <= 30 or index == 1 or index % 50 == 0 or index == total:
            print(f"{index}/{total} {clip['group']}/{clip['clip']}", file=sys.stderr, flush=True)
    return clips


def run(repository: Path, per_camera: int, jobs: int = 1) -> dict:
    repository = repository.resolve()
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip() != REVISION:
        raise ValueError(f"RoadTrack checkout must be at {REVISION}")
    modifications = subprocess.run(
        ["git", "-C", str(repository), "diff", "--name-only", "HEAD", "--"],
        check=True,
        text=True,
        capture_output=True,
    )
    if modifications.stdout:
        raise ValueError("RoadTrack checkout must have no edits to tracked files")
    sys.path.insert(0, str(repository))
    started = perf_counter()
    try:
        import numpy as np

        upstream = importlib.import_module("run_roadtrack")
        with chdir(repository):
            # The author runner allows pickle. Verify calibration arrays without it first.
            for camera in CAMERAS:
                matrix = np.load(Path("homo_files") / f"{camera}.npy", allow_pickle=False)
                if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                    raise ValueError(f"invalid calibration array for {camera}")
            filters = json.loads(Path("filter_files/filter.json").read_text(encoding="utf-8"))
            clips = _select(upstream, per_camera)
        play = partial(_run_clip, repository, filters)
        if jobs == 1:
            results = _progress(map(play, clips), len(clips))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                results = _progress(pool.map(play, clips, chunksize=4), len(clips))
    finally:
        sys.path.pop(0)
    methods = ("roadtrack_author_score", "roadtrack", "project")
    return {
        "roadtrack_revision": REVISION,
        "selection": "first N lexicographically per camera and group, or all for N=0",
        "per_camera_and_group": per_camera,
        "fps": FPS,
        "processes": jobs,
        "elapsed_s": perf_counter() - started,
        "project_track_confidence": CONFIDENCE,
        "common_confirmation_iou": IOU,
        "dependencies": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "opencv-python",
                "filterpy",
                "lap",
                "trackers",
                "supervision",
            )
        },
        "summary": {method: _summarize(results, method) for method in methods},
        "clips": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roadtrack_checkout", type=Path)
    parser.add_argument("--per-camera", type=int, default=2, help="use 0 for all sequences")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.per_camera < 0:
        parser.error("--per-camera must be nonnegative")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        print(json.dumps(run(args.roadtrack_checkout, args.per_camera, args.jobs), indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
