from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .detection import BoxDetector, VehicleBox

VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


class RTDetrDetector(BoxDetector):
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
    ) -> None:
        super().__init__(confidence, continuation_confidence, roi, tile_size, tile_confidence)
        if not model_path.is_dir():
            raise ValueError("RT-DETR requires a local checkpoint directory")
        try:
            import torch
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        except ImportError as error:
            raise ValueError("RT-DETR requires PyTorch, torchvision and Transformers") from error

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("RT-DETR CUDA requires a CUDA-enabled PyTorch installation")
        self.torch = torch
        self.device = device
        self.processor = RTDetrImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = (
            RTDetrForObjectDetection.from_pretrained(
                model_path, local_files_only=True, use_safetensors=True
            )
            .eval()
            .to(device)
        )

    def _crop_boxes(self, frame: MatLike) -> list[VehicleBox]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            result = self.processor.post_process_object_detection(
                self.model(**inputs),
                threshold=self.continuation_confidence,
                target_sizes=[(height, width)],
            )[0]
        labels = result["labels"].tolist()
        scores = result["scores"].tolist()
        coordinates = result["boxes"].cpu().numpy()
        boxes = []
        for label, score, bounds in zip(labels, scores, coordinates, strict=True):
            category = self.model.config.id2label[label].lower()
            if category not in VEHICLE_NAMES or not np.isfinite(bounds).all():
                continue
            x1, y1, x2, y2 = np.clip(bounds, 0, [width - 1, height - 1, width - 1, height - 1])
            if x2 > x1 and y2 > y1:
                boxes.append(
                    VehicleBox(
                        float(x1), float(y1), float(x2 - x1), float(y2 - y1), score, category
                    )
                )
        return boxes
