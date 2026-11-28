import hashlib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile

import cv2
import numpy as np
from cv2.typing import MatLike

from traffic_trajectory.detection import VehicleBox

FOG_SHA256 = "09f4f8987e030b8da9e76eb73ab00b64fb628cd43df42e460bc26b9a0912c7c2"
VEHICLES = {"car", "bus", "truck", "motorcycle"}


def fog_examples(archive: Path) -> Iterator[tuple[MatLike, list[VehicleBox]]]:
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != FOG_SHA256:
            raise ValueError("expected the pinned DAWN V1 Fog.zip")
    with ZipFile(archive) as dataset:
        annotations = sorted(name for name in dataset.namelist() if name.endswith(".xml"))
        if len(annotations) != 300:
            raise ValueError("expected 300 XML annotations in DAWN V1 Fog.zip")
        for name in annotations:
            file = PurePosixPath(name).stem + ".jpg"
            frame = cv2.imdecode(
                np.frombuffer(dataset.read(f"Fog/{file}"), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError(f"cannot decode {file}")
            height, width = frame.shape[:2]
            root = ElementTree.fromstring(dataset.read(name))
            labels = []
            for item in root.findall("object"):
                kind = item.findtext("name")
                if kind not in VEHICLES:
                    continue
                x1 = max(0, int(item.findtext("bndbox/xmin")))
                y1 = max(0, int(item.findtext("bndbox/ymin")))
                x2 = min(width, int(item.findtext("bndbox/xmax")))
                y2 = min(height, int(item.findtext("bndbox/ymax")))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"invalid vehicle coordinates in {name}")
                labels.append(VehicleBox(x1, y1, x2 - x1, y2 - y1, 1.0, kind))
            yield frame, labels
