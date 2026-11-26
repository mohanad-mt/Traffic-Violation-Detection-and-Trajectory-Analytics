# Nighttime road clip check

Run pretrained YOLOX-S on eight seconds of real nighttime traffic. I reviewed the near-right lane from 7.5 through 15.4 seconds of the source and annotated zero wrong-way or stopped-vehicle events under the project's three-second stop rule. The clip contains an overlaid signal graphic, so this scene has no red-light rule configuration.

Download the original [traffic_video.mp4](https://github.com/FarzadNekouee/Traffic-Violation-Detection/blob/master/traffic_video.mp4) and the full-precision [OpenCV Zoo YOLOX-S weights](https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx). The source video and model are external inputs.

| File | SHA-256 |
| --- | --- |
| `traffic_video.mp4` | `91315f6ab7a854e2e1d6145cdc9cf2663768666edd54e9b992f09bbdacec1a3f` |
| `object_detection_yolox_2022nov.onnx` | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |

From the repository root, with the video requirements installed and the two downloads available locally, run:

```sh
python benchmarks/nighttime/prepare_clip.py traffic_video.mp4 build/nighttime/sample.avi
python -m traffic_trajectory benchmarks/nighttime/scene.json --video build/nighttime/sample.avi --model object_detection_yolox_2022nov.onnx --include-tracks > build/nighttime/predictions.json
python -m traffic_trajectory.evaluate build/nighttime/predictions.json benchmarks/nighttime/labels.json
```

The local CPU run on the resampled 80-frame clip produced 419 observations across 62 track IDs for the full frame and zero false alerts in the annotated lane. Precision, recall, and F1 are undefined for this clip because the annotation and prediction sets are both empty. The track-ID count includes short and fragmented tracks. A representative event benchmark requires additional camera views and positive examples of each event type.
