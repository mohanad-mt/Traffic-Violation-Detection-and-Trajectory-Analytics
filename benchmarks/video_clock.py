"""Measure how far a frame-index clock departs from video packet timestamps."""

import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from statistics import median

import av


def measure(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if not rate or not stream.time_base:
            raise ValueError(f"missing nominal rate or time base in {path}")
        pts = sorted({packet.pts for packet in container.demux(stream) if packet.pts is not None})
        if not pts:
            raise ValueError(f"no video packet timestamps in {path}")
        error = max(
            abs(float((time - pts[0]) * stream.time_base - index / rate))
            for index, time in enumerate(pts)
        )
        return {
            "frames": len(pts),
            "average_fps": str(rate),
            "time_base": str(stream.time_base),
            "pts_delta_counts": dict(Counter(b - a for a, b in pairwise(pts))),
            "max_fps_clock_error_s": error,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing MP4 files")
    args = parser.parse_args()
    try:
        clips = {path.name: measure(path) for path in sorted(args.directory.glob("*.mp4"))}
        if not clips:
            raise ValueError("no MP4 files found")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    errors = [row["max_fps_clock_error_s"] for row in clips.values()]
    print(
        json.dumps(
            {
                "clips": len(clips),
                "nonuniform_pts_steps": sum(
                    len(row["pts_delta_counts"]) > 1 for row in clips.values()
                ),
                "clips_over_100ms": sum(error > 0.1 for error in errors),
                "median_max_error_s": median(errors),
                "max_error_s": max(errors),
                "per_clip": clips,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
