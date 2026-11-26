import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 160
FPS = 10


def generate_video(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    background = np.full((HEIGHT, WIDTH, 3), 45, dtype=np.uint8)
    if not cv2.imwrite(str(destination / "background.png"), background):
        raise RuntimeError("cannot write the demo background")
    writer = cv2.VideoWriter(
        str(destination / "demo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    try:
        if not writer.isOpened():
            raise RuntimeError("cannot open the demo video writer")
        for index in range(6 * FPS):
            frame = background.copy()
            centers = [(78, 131), (50 + 4 * index, 94)]
            if index <= 32:
                centers.append((260 - 4 * index, 30))
            if index >= 24:
                centers.append((70 + 4 * (index - 24), 62))
            for x, y in centers:
                cv2.rectangle(frame, (x - 10, y - 7), (x + 10, y + 7), (235, 235, 235), -1)
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the synthetic traffic clip.")
    parser.add_argument("destination", type=Path)
    generate_video(parser.parse_args().destination)
