from pathlib import Path

from cv2.typing import MatLike


class CudaOnnxNet:
    """OpenCV-style interface for the single-input, single-output YOLOX export."""

    def __init__(self, model: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ValueError("CUDA inference requires requirements-cuda.txt") from error
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ValueError("install onnxruntime-gpu instead of onnxruntime for CUDA inference")

        try:
            ort.preload_dlls()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(model),
                sess_options=options,
                providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
            )
        except Exception as error:
            raise ValueError(f"could not initialize ONNX Runtime CUDA: {error}") from error
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise ValueError(
                "ONNX Runtime did not load CUDA. Check CUDA 13 and cuDNN 9 or select --device cpu"
            )
        self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("expected a YOLOX export with one ONNX input and one output")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def setInput(self, blob: MatLike) -> None:
        self.blob = blob

    def forward(self) -> MatLike:
        return self.session.run([self.output_name], {self.input_name: self.blob})[0]
