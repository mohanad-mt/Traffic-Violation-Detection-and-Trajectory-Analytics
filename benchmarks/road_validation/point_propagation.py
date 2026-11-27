"""Check causal point propagation from one labeled seed on each wrong-way clip."""

import argparse
import json
from math import dist
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from benchmarks.assets import sha256
from benchmarks.road_validation.compare import DATA, read_json
from traffic_trajectory.video import decoded_frames, open_video

RADIUS_PX = 12.0
INPUT_RESOLUTION = 512
SEED_OFFSETS_PX = (0, 3)


def follow(model, video: Path, references: list[dict], seed_above_px: int) -> dict:
    with open_video(video) as (capture, fps):
        target = {round(row["timestamp_s"] * fps): row for row in references}
        first_frame, last_frame = min(target), max(target)
        origin = tuple(target[first_frame]["position"])
        seed = np.asarray([[origin[0], max(0, origin[1] - seed_above_px)]], dtype=np.float32)
        y_offset = origin[1] - float(seed[0, 1])
        state = None
        rows, timings = [], []
        for frame_index, image in enumerate(decoded_frames(capture)):
            if frame_index < first_frame:
                continue
            started = perf_counter()
            points, visible, state = model.track_frame(
                image, query_points_xy=seed if frame_index == first_frame else None, state=state
            )
            if frame_index != first_frame:
                timings.append((perf_counter() - started) * 1000)
                if frame_index in target:
                    reference = target[frame_index]
                    actual = reference["position"]
                    estimate = points[0].tolist()
                    estimate[1] += y_offset
                    rows.append(
                        {
                            **reference,
                            "estimate": estimate,
                            "visible": bool(visible[0]),
                            "error_px": dist(actual, estimate),
                            "stationary_error_px": dist(actual, origin),
                        }
                    )
            if frame_index == last_frame:
                break
        if len(rows) != len(references) - 1:
            raise ValueError(f"reference samples are missing from {video}")
    return {
        "video_sha256": sha256(video),
        "fps": fps,
        "seed": target[first_frame],
        "query_point": seed[0].tolist(),
        "samples": rows,
        "visible_hits": sum(row["visible"] and row["error_px"] <= RADIUS_PX for row in rows),
        "stationary_hits": sum(row["stationary_error_px"] <= RADIUS_PX for row in rows),
        "mean_wall_ms_after_seed": mean(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("road_folder", type=Path)
    parser.add_argument("night_transfer_folder", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    model = TAPNextPP.from_checkpoint(
        args.checkpoint, device=args.device, input_resolution=INPUT_RESOLUTION
    )
    fog = read_json(DATA / "fog.manual-trace.json")["tracks"][0]["observations"]
    red = read_json(DATA / "red.identity-labels.json")["events"][0]["trajectory"]
    night = DATA.parent / "night_transfer"
    clips = {
        "fog": (args.road_folder / "wrong.avi", fog),
        "red": (args.road_folder / "red.avi", red),
    }
    for name in ("camera-1", "camera-2"):
        reference = read_json(night / f"{name}.labels.json")["events"][0]["trajectory"]
        clips[name] = (args.night_transfer_folder / f"{name}.avi", reference)
    report = {
        "method": "TAPNext++ 512, single point, first annotated sample, causal",
        "checkpoint_sha256": sha256(args.checkpoint),
        "torch": torch.__version__,
        "device": args.device,
        "hardware": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "radius_px": RADIUS_PX,
        "seed_offsets_above_px": SEED_OFFSETS_PX,
        "cases": {
            name: {offset: follow(model, video, refs, offset) for offset in SEED_OFFSETS_PX}
            for name, (video, refs) in clips.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
