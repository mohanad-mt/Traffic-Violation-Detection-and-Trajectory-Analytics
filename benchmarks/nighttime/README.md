# Nighttime road clip check

Eight seconds of fixed-camera footage, starting at 7.5 s. The near-right lane had no wrong-way events or stops lasting three seconds. The signal graphic is an overlay, so this scene runs only the wrong-way and stop rules.

Download both inputs. [Original clip](https://github.com/FarzadNekouee/Traffic-Violation-Detection/blob/master/traffic_video.mp4) · [Full-precision OpenCV Zoo YOLOX-S weights](https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx)

| File | SHA-256 |
| --- | --- |
| `traffic_video.mp4` | `91315f6ab7a854e2e1d6145cdc9cf2663768666edd54e9b992f09bbdacec1a3f` |
| `object_detection_yolox_2022nov.onnx` | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |

With the video requirements installed, run from the repo root.

```sh
python benchmarks/nighttime/prepare_clip.py traffic_video.mp4 build/nighttime/sample.avi
python -m traffic_trajectory benchmarks/nighttime/scene.json --video build/nighttime/sample.avi --model object_detection_yolox_2022nov.onnx --include-tracks > build/nighttime/predictions.json
python -m traffic_trajectory.evaluate build/nighttime/predictions.json benchmarks/nighttime/labels.json
```

Across 80 frames, YOLOX-S produced 419 observations and the tracker assigned 62 IDs over the full frame, including short and fragmented tracks. The annotated lane had zero alerts. Precision, recall, and F1 are undefined here because there were no labels or alerts.
