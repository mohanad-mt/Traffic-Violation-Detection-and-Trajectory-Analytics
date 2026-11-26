"""Compare association on the two GRAM-RTM sequences with published detections."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from trackeval.datasets._base_dataset import _BaseDataset
from trackeval.metrics import CLEAR, HOTA, Identity

from traffic_trajectory.geometry import VehicleBox
from traffic_trajectory.tracker import FrameDetections, make_tracker

ARCHIVE_SHA256 = "acf7d560f0eb27e45325c35feea3e64d0ad1ba324db62ca0656c8ecf56bffb00"
SEQUENCES = {"M-30": (7520, 30.0), "Urban1": (23435, 25.0)}
TRACKERS = ("point", "bytetrack", "ocsort")
TUNING = {"max_gap_s": 0.75, "max_distance_px": 24, "confidence": 0.35}


def load_sequence(archive: ZipFile, name: str, frame_count: int) -> tuple[list, list]:
    source = archive.read(f"GRAM-RTMv4/Results/Detection/{name}/detection_results.txt").decode(
        "utf-8"
    )
    detections = [[] for _ in range(frame_count)]
    for row in source.splitlines():
        index, x1, y1, x2, y2, _ = row.split()
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        if not 1 <= int(index) <= frame_count or not (x2 > x1 and y2 > y1):
            raise ValueError(f"invalid GRAM detection: {row}")
        detections[int(index) - 1].append(VehicleBox(x1, y1, x2 - x1, y2 - y1, 1.0))
    labels = []
    for index in range(frame_count):
        path = f"GRAM-RTMv4/Annotations/{name}/xml/{index}.xml"
        root = ElementTree.fromstring(archive.read(path))
        frame = []
        for obj in root.findall("object"):
            x1, y1 = (int(obj.findtext(f"bndbox/{key}min")) for key in ("x", "y"))
            x2, y2 = (int(obj.findtext(f"bndbox/{key}max")) for key in ("x", "y"))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"invalid GRAM reference box in {path}")
            frame.append((obj.findtext("ID"), VehicleBox(x1, y1, x2 - x1, y2 - y1, 1.0)))
        labels.append(frame)
    return labels, detections


def normalize(sequence: list[list[tuple[str, VehicleBox]]]) -> tuple[list, list, int]:
    ids, boxes = [], []
    remap = {}
    for frame in sequence:
        ids.append(np.array([remap.setdefault(key, len(remap)) for key, _ in frame], dtype=int))
        if len(np.unique(ids[-1])) != len(ids[-1]):
            raise ValueError("expected each GRAM object ID to occur at most once per frame")
        boxes.append(np.asarray([(b.x, b.y, b.width, b.height) for _, b in frame]).reshape(-1, 4))
    return ids, boxes, len(remap)


def score(labels: list, predictions: list, sequence: str) -> dict:
    gt_ids, gt_boxes, gt_count = normalize(labels)
    track_ids, track_boxes, track_count = normalize(predictions)
    data = {
        "seq": sequence,
        "num_timesteps": len(gt_ids),
        "gt_ids": gt_ids,
        "tracker_ids": track_ids,
        "num_gt_ids": gt_count,
        "num_tracker_ids": track_count,
        "num_gt_dets": sum(map(len, gt_ids)),
        "num_tracker_dets": sum(map(len, track_ids)),
        "similarity_scores": [
            _BaseDataset._calculate_box_ious(a, b, box_format="xywh")
            for a, b in zip(gt_boxes, track_boxes, strict=True)
        ],
    }
    hota = HOTA().eval_sequence(data)
    clear = CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)
    identity = Identity({"PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "IDF1": float(identity["IDF1"]),
        "MOTA": float(clear["MOTA"]),
        "identity_switches": int(clear["IDSW"]),
        "fragments": int(clear["Frag"]),
        "mostly_tracked": int(clear["MT"]),
        "mostly_lost": int(clear["ML"]),
        "reference_ids": gt_count,
        "predicted_ids": track_count,
    }


def run(archive: Path) -> dict:
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("expected the pinned GRAM-RTMv4.zip")
    results = {}
    with ZipFile(archive) as dataset:
        for name, (frame_count, fps) in SEQUENCES.items():
            labels, detections = load_sequence(dataset, name, frame_count)
            rows = {}
            for tracker_name in TRACKERS:
                tracker = make_tracker(tracker_name, fps, **TUNING)
                start = perf_counter()
                tracks = []
                for index, boxes in enumerate(detections):
                    tracked = tracker.update_frame(
                        index / fps, FrameDetections.from_boxes(boxes, TUNING["confidence"])
                    )
                    tracks.append([(key, sample.box) for key, sample in tracked])
                elapsed = perf_counter() - start
                rows[tracker_name] = {
                    **score(labels, tracks, name),
                    "mean_tracker_ms": 1000 * elapsed / frame_count,
                }
            results[name] = {"frames": frame_count, "fps": fps, "trackers": rows}
    return {
        "archive_sha256": ARCHIVE_SHA256,
        "split": "published test sequences, no tuning",
        "configuration": TUNING,
        "detection_scores": "Retain authors' detections and treat them all as strong",
        "sequences": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.archive), indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
