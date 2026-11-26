import argparse
from pathlib import Path

import cv2

START_S = 7.5
SAMPLE_FPS = 10
FRAME_COUNT = 80
SIZE = (640, 338)


def prepare_clip(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"MJPG"), SAMPLE_FPS, SIZE)
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise ValueError("cannot open the source video or target AVI")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("source video must report a positive frame rate")
        for index in range(FRAME_COUNT):
            source_frame = round((START_S + index / SAMPLE_FPS) * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot decode source frame {source_frame}")
            writer.write(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the nighttime evaluation clip.")
    parser.add_argument("source", type=Path, help="Original traffic_video.mp4")
    parser.add_argument("target", type=Path, help="Output AVI")
    args = parser.parse_args()
    try:
        prepare_clip(args.source, args.target)
    except ValueError as error:
        parser.error(str(error))
