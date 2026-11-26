import argparse
import json
import sys
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path

from .detector import detect_events
from .scenario import parse_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect candidate events from tracks or video.")
    parser.add_argument("scenario", help="Path to a scenario JSON file, or - to read from stdin")
    parser.add_argument("--video", type=Path, help="Constant-frame-rate video file")
    detector_args = parser.add_mutually_exclusive_group()
    detector_args.add_argument("--background", type=Path, help="Empty-road reference image")
    detector_args.add_argument("--model", type=Path, help="OpenCV Zoo YOLOX-S ONNX model")
    parser.add_argument(
        "--confidence", type=float, default=0.35, help="YOLOX confidence threshold"
    )
    parser.add_argument(
        "--continuation-confidence",
        type=float,
        help="Lower YOLOX cutoff for extending existing tracks",
    )
    parser.add_argument(
        "--detection-roi",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Run YOLOX on a rectangle in video pixels",
    )
    parser.add_argument(
        "--detection-tile-size",
        type=int,
        help="Also run YOLOX on overlapping square crops of this size in pixels",
    )
    parser.add_argument("--min-blob-area-px", type=int, default=100, help="Foreground area cutoff")
    parser.add_argument(
        "--track-distance-px", type=float, default=24, help="Tracker distance gate"
    )
    parser.add_argument(
        "--include-tracks", action="store_true", help="Include trajectory histories"
    )
    args = parser.parse_args()
    if args.continuation_confidence is not None:
        if args.model is None:
            parser.error("--continuation-confidence requires --model")
        if not 0 < args.continuation_confidence <= args.confidence:
            parser.error("--continuation-confidence must be positive and at most --confidence")
    if args.detection_roi is not None and args.model is None:
        parser.error("--detection-roi requires --model")
    if args.detection_tile_size is not None:
        if args.model is None:
            parser.error("--detection-tile-size requires --model")
        if args.detection_tile_size <= 0:
            parser.error("--detection-tile-size must be positive")

    try:
        if args.scenario == "-":
            payload = json.load(sys.stdin.buffer)
        else:
            payload = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
        if args.video is None:
            if args.background is not None or args.model is not None:
                raise ValueError("--background and --model require --video")
            scenario = parse_scenario(payload)
        else:
            if not isinstance(payload, dict) or "tracks" in payload:
                raise ValueError("video scenario must be an object without a tracks field")
            if args.background is None and args.model is None:
                raise ValueError("--video requires either --background or --model")
            if not 0 < args.confidence < 1:
                raise ValueError("--confidence must be between 0 and 1")
            if args.min_blob_area_px <= 0:
                raise ValueError("--min-blob-area-px must be positive")
            if not isfinite(args.track_distance_px) or args.track_distance_px <= 0:
                raise ValueError("--track-distance-px must be finite and positive")
            scenario = parse_scenario({**payload, "tracks": []})
            try:
                import cv2

                from .video import foreground_detector, track_video
                from .yolox import YoloXDetector
            except ImportError as error:
                raise ValueError("video mode requires requirements-video.txt") from error
            try:
                if args.model is not None:
                    roi = tuple(args.detection_roi) if args.detection_roi is not None else None
                    detect = YoloXDetector(
                        args.model,
                        args.confidence,
                        args.continuation_confidence,
                        roi,
                        tile_size=args.detection_tile_size,
                    )
                else:
                    detect = foreground_detector(args.background, args.min_blob_area_px)
                tracks = track_video(
                    args.video, detect, scenario.thresholds.max_gap_s, args.track_distance_px
                )
            except cv2.error as error:
                raise ValueError(f"OpenCV could not process the input: {error}") from error
            scenario = replace(scenario, tracks=tracks)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output = {"events": [asdict(event) for event in detect_events(scenario)]}
    if args.include_tracks:
        output["tracks"] = [asdict(track) for track in scenario.tracks]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
