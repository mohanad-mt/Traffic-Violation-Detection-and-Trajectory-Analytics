import argparse
from pathlib import Path

import cv2

SOURCE_FPS = 15
START_FRAME = 45
SOURCE_FRAME_COUNT = 225
FRAME_STRIDE = 2
SIZE = (360, 240)


def prepare_clip(
    source: Path,
    target: Path,
    *,
    start_frame: int = START_FRAME,
    source_frames: int = SOURCE_FRAME_COUNT,
    size: tuple[int, int] = SIZE,
    source_fps: int = SOURCE_FPS,
    stride: int = FRAME_STRIDE,
    resize: tuple[int, int] | None = None,
) -> None:
    if source.resolve() == target.resolve():
        raise ValueError("source and target must be different files")
    output_size = resize or size
    if start_frame < 0 or min(source_frames, source_fps, stride, *size, *output_size) <= 0:
        raise ValueError("expected a nonnegative start frame and positive video settings")
    target.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"MJPG"), source_fps / stride, output_size
    )
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise ValueError("cannot open the source video or target AVI")
        if capture.get(cv2.CAP_PROP_FPS) != source_fps:
            raise ValueError(f"expected a {source_fps} FPS source video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for index in range(source_frames):
            ok, frame = capture.read()
            if not ok or frame.shape[1::-1] != size:
                raise ValueError(f"cannot decode source frame {start_frame + index} at {size}")
            if index % stride == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                writer.write(frame)
    finally:
        capture.release()
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim and sample a benchmark video.")
    parser.add_argument("source", type=Path, help="Original source video")
    parser.add_argument("target", type=Path, help="Output AVI")
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--source-frames", type=int, default=SOURCE_FRAME_COUNT)
    parser.add_argument("--size", nargs=2, type=int, default=SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--source-fps", type=int, default=SOURCE_FPS)
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE)
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    try:
        prepare_clip(
            args.source,
            args.target,
            start_frame=args.start_frame,
            source_frames=args.source_frames,
            size=tuple(args.size),
            source_fps=args.source_fps,
            stride=args.stride,
            resize=tuple(args.resize) if args.resize else None,
        )
    except ValueError as error:
        parser.error(str(error))
