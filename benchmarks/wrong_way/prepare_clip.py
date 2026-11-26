import argparse
from pathlib import Path

import cv2

SOURCE_FPS = 15
START_FRAME = 45
SOURCE_FRAME_COUNT = 225
FRAME_STRIDE = 2
SIZE = (360, 240)


def prepare_clip(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"MJPG"), SOURCE_FPS / FRAME_STRIDE, SIZE
    )
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise ValueError("cannot open the source video or target AVI")
        if capture.get(cv2.CAP_PROP_FPS) != SOURCE_FPS:
            raise ValueError("expected the 15 FPS source video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        for index in range(SOURCE_FRAME_COUNT):
            ok, frame = capture.read()
            if not ok or frame.shape[1::-1] != SIZE:
                raise ValueError(f"cannot decode source frame {START_FRAME + index} at {SIZE}")
            if index % FRAME_STRIDE == 0:
                writer.write(frame)
    finally:
        capture.release()
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the NJIT wrong-way evaluation clip.")
    parser.add_argument("source", type=Path, help="Original wrong_way_cam1.mp4")
    parser.add_argument("target", type=Path, help="Output AVI")
    args = parser.parse_args()
    try:
        prepare_clip(args.source, args.target)
    except ValueError as error:
        parser.error(str(error))
