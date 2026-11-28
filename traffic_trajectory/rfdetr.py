import sys
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .detection import BoxDetector, VehicleBox


class RFDetrDetector(BoxDetector):
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        continuation_confidence: float | None = None,
        roi: tuple[int, int, int, int] | None = None,
        *,
        tile_size: int | None = None,
        tile_confidence: float | None = None,
        device: str = "cpu",
        large: bool = False,
    ) -> None:
        super().__init__(confidence, continuation_confidence, roi, tile_size, tile_confidence)
        if not model_path.is_file():
            raise ValueError("RF-DETR requires a local checkpoint file matching the selected size")
        try:
            import torch
            from rfdetr import RFDETRLarge, RFDETRSmall
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ImportError as error:
            raise ValueError("RF-DETR requires requirements-rfdetr.txt") from error

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("RF-DETR CUDA requires a CUDA-enabled PyTorch installation")
        self.categories = {
            index: name
            for index, name in COCO_CLASSES.items()
            if name in ("car", "motorcycle", "bus", "truck")
        }
        # RF-DETR sends checkpoint setup logs to stdout. Keep command output as JSON.
        with redirect_stdout(sys.stderr):
            load_model = RFDETRLarge if large else RFDETRSmall
            model = load_model(pretrain_weights=str(model_path.resolve()), device=device)
            self.predictor = model.inference(compile=False)

    def _crop_boxes(self, frame: MatLike) -> list[VehicleBox]:
        height, width = frame.shape[:2]
        result = self.predictor.predict(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            threshold=self.continuation_confidence,
            include_source_image=False,
        )
        boxes = []
        for label, score, bounds in zip(
            result.class_id, result.confidence, result.xyxy, strict=True
        ):
            category = self.categories.get(int(label))
            if category is None or not np.isfinite(bounds).all():
                continue
            x1, y1, x2, y2 = np.clip(bounds, 0, [width - 1, height - 1, width - 1, height - 1])
            if x2 > x1 and y2 > y1:
                boxes.append(
                    VehicleBox(
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                        float(score),
                        category,
                    )
                )
        return boxes
