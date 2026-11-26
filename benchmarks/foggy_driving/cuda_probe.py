"""Local CUDA timing probe for this specific YOLOX graph."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import onnx
import torch
from onnx.helper import get_attribute_value
from onnx.numpy_helper import to_array
from torch.nn import functional as F

from benchmarks.foggy_driving.evaluate_boxes import evaluate
from traffic_trajectory.yolox import YoloXDetector

WEIGHT_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"


class TorchGraph:
    def __init__(self, path):
        model = onnx.load(path)
        constants = {weight.name: to_array(weight) for weight in model.graph.initializer}
        self.weights = {
            name: torch.tensor(value.copy(), device="cuda")
            for name, value in constants.items()
            if value.dtype.kind == "f"
        }
        self.ops = []
        for node in model.graph.node:
            attrs = {attr.name: get_attribute_value(attr) for attr in node.attribute}
            args = list(node.input)
            if node.op_type == "Slice":
                start, end, axis, step = (int(constants[key][0]) for key in args[1:])
                slices = [slice(None)] * 4
                slices[axis] = slice(start, end, step)
                attrs["slices"] = tuple(slices)
                args = args[:1]
            elif node.op_type == "Reshape":
                attrs["shape"] = constants[args[1]].tolist()
                args = args[:1]
            elif node.op_type == "Resize":
                assert attrs["mode"] == b"nearest"
                assert attrs["coordinate_transformation_mode"] == b"asymmetric"
                assert attrs["nearest_mode"] == b"floor"
                attrs["scale"] = constants[args[2]].tolist()[2:]
                args = args[:1]
            if "pads" in attrs:
                assert attrs["pads"][:2] == attrs["pads"][2:]
            self.ops.append((node.op_type, args, attrs, node.output[0]))

    def setInput(self, blob):
        self.blob = blob

    @torch.inference_mode()
    def forward(self):
        values = {**self.weights, "images": torch.from_numpy(self.blob).cuda()}
        for op, names, attrs, out in self.ops:
            inputs = [values[name] for name in names]
            x = inputs[0]
            match op:
                case "Conv":
                    result = F.conv2d(
                        *inputs,
                        stride=attrs["strides"],
                        padding=attrs["pads"][:2],
                        dilation=attrs["dilations"],
                        groups=attrs["group"],
                    )
                case "MaxPool":
                    result = F.max_pool2d(
                        x,
                        attrs["kernel_shape"],
                        attrs["strides"],
                        attrs["pads"][:2],
                        ceil_mode=bool(attrs["ceil_mode"]),
                    )
                case "Sigmoid":
                    result = x.sigmoid()
                case "Mul":
                    result = x * inputs[1]
                case "Add":
                    result = x + inputs[1]
                case "Concat":
                    result = torch.cat(inputs, dim=attrs["axis"])
                case "Slice":
                    result = x[attrs["slices"]]
                case "Reshape":
                    result = x.reshape(attrs["shape"])
                case "Resize":
                    result = F.interpolate(x, scale_factor=attrs["scale"], mode="nearest")
                case "Transpose":
                    result = x.permute(attrs["perm"])
                case _:
                    raise ValueError(f"unexpected ONNX operation: {op}")
            values[out] = result
        return values["output"].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Time the pinned YOLOX-S graph on CUDA.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch and an available NVIDIA GPU are required")
    with args.model.open("rb") as model_file:
        if hashlib.file_digest(model_file, "sha256").hexdigest() != WEIGHT_SHA256:
            parser.error("this probe only supports the documented OpenCV Zoo YOLOX-S weights")

    cv2.setNumThreads(4)
    torch.backends.cudnn.allow_tf32 = False
    graph = TorchGraph(args.model)
    cpu = cv2.dnn.readNetFromONNX(str(args.model))
    with ZipFile(args.archive) as dataset:
        assets = sorted(
            n for n in dataset.namelist() if "/leftImg8bit/test/" in n and n.endswith(".png")
        )
        for name in (assets[0], assets[-1]):
            image = cv2.imdecode(
                np.frombuffer(dataset.read(name), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            blob = cv2.dnn.blobFromImage(image, size=(640, 640), swapRB=True)
            cpu.setInput(blob)
            graph.setInput(blob)
            expected, actual = cpu.forward(), graph.forward()
            error = float(np.max(np.abs(expected - actual)))
            assert error < 0.002, error
            print(f"Raw output max abs error: {error:.8f}", flush=True)
    for size in (None, 640):
        detector = YoloXDetector(args.model, 0.35, tile_size=size)
        detector.net = graph
        print(
            json.dumps({"tile_size": size, **evaluate(args.archive, detector)}, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
