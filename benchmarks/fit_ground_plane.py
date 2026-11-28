"""Fit image-to-ground homography from measured point pairs."""

import argparse
import json
from dataclasses import asdict
from math import dist
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from traffic_trajectory.calibration import GroundPlane
from traffic_trajectory.scenario import _array, _id, _object, _point


def pairs(value: object, key: str) -> tuple[list, list]:
    image, world = [], []
    for i, raw in enumerate(_array(value, key)):
        path = f"{key}[{i}]"
        row = _object(raw, path)
        image.append(_point(row.get("image"), f"{path}.image"))
        world.append(_point(row.get("world"), f"{path}.world"))
    return image, world


def fit(document: dict) -> dict:
    source, target = pairs(document.get("fit"), "fit")
    if len(source) < 4:
        raise ValueError("at least four ground-plane correspondences are required")
    matrix, _ = cv2.findHomography(np.array(source), np.array(target))
    if matrix is None:
        raise ValueError("the calibration points do not determine a homography")
    ground = GroundPlane(tuple(map(tuple, matrix.tolist())), _id(document.get("unit"), "unit"))

    def residuals(image, world):
        errors = []
        for pixel, reference in zip(image, world, strict=True):
            projected = ground.project(pixel)
            if projected is None:
                raise ValueError("a calibration point projects to infinity")
            errors.append(dist(projected, reference))
        return {"points": len(errors), "mean_error": mean(errors), "max_error": max(errors)}

    report = {"ground_plane": asdict(ground), "fit": residuals(source, target)}
    if "validation" in document:
        check = pairs(document["validation"], "validation")
        if not check[0]:
            raise ValueError("validation must contain at least one point")
        report["validation"] = residuals(*check)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_pairs", type=Path)
    args = parser.parse_args()
    try:
        print(
            json.dumps(fit(json.loads(args.point_pairs.read_text(encoding="utf-8-sig"))), indent=2)
        )
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
