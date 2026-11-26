# Road validation clips

Each clip has one event in the annotated area. The NJIT timestamps come from the project's detector logs, rounded to whole seconds. For the Zenodo clip, the label is a visual estimate from the sedan crossing the line at about 6.3 s in the source. The overhead signals are red for the full extract.

| Case | Source and extract | SHA-256 |
| --- | --- | --- |
| Wrong way | [NJIT foggy highway clip](https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong_way_cam3.mp4), 0–8 s. [Timestamp log](https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong-way.txt) | `91eea63673d648e42d348204d7bbc831cabe36a80f13675ca853e3c6aedb4588` |
| Stopped vehicle | [NJIT highway clip](https://web.njit.edu/~cliu/NJDOT/stopped-veh/CCTV_1-2014-9-4-11-0.mp4), 30–51 s. [Timestamp log](https://web.njit.edu/~cliu/NJDOT/stopped-veh/stopped-veh.txt) | `5126407d4e3de00a22a36abf8a4e4bb5d5252952505c01d763e1a956be43b23f` |
| Red light | [Tiago Silva, Dataset for Traffic Evaluation](https://doi.org/10.5281/zenodo.14041920), `rlr_occasions.mp4`, 4.5–7.8 s | `58b953698121e06c43a4f6398b11bab7485782f8b82c91a556b6d32923d7bd78` |

## Results

| Run | Onset tolerance | Matched | Extra alerts | Notes |
| --- | ---: | ---: | ---: | --- |
| Foggy wrong way, full frame at 0.35 | ±1.0 s | 0/1 | 0 | Only 15 vehicle observations across the clip. |
| Foggy wrong way, crop at 0.08 | ±1.0 s | 0/1 | 0 | 11 observations. The reference vehicle did not yield a usable track. |
| Highway stop, full frame at 0.35 | ±1.0 s | 0/1 | 0 | No stop alert. |
| Highway stop, full frame at 0.08 | ±1.0 s | 1/1 | 1 | The matched alert started at 4.40 s. A second track near the same spot emitted another alert after a gap. |
| Highway stop, crop at 0.08 | ±1.0 s | 0/1 | 1 | The single alert started at 2.80 s, 2.20 s before the reference. It matches at ±3.0 s. |
| Highway stop, crop with continuation cutoff 0.08 | ±1.0 s | 0/1 | 0 | No alert at either track-start cutoff, 0.20 or 0.35. |
| Red light, full frame at 0.35 | ±0.5 s | 1/1 | 0 | Estimated crossing at 2.24 s, confirmed at 2.50 s. The label is 1.80 s. |

The foggy clip had intermittent detections. A contrast trial applied OpenCV CLAHE to the Lab luminance channel with clip limit 2 and 8 × 8 tiles. It produced 22 observations with the original crop and 65 with crop `100 133 202 240`. Both runs missed the reference event. Each clip has one positive, so these scores describe the clips.

## Reproduce

Use the YOLOX-S weights from the [nighttime check](../nighttime/README.md). From the repo root, put the source files next to the commands or replace the input paths.

```sh
python benchmarks/wrong_way/prepare_clip.py wrong_way_cam3.mp4 build/road/wrong.avi --start-frame 0 --source-frames 120
python benchmarks/wrong_way/prepare_clip.py CCTV_1-2014-9-4-11-0.mp4 build/road/stop.avi --start-frame 450 --source-frames 315 --size 320 240
python benchmarks/wrong_way/prepare_clip.py rlr_occasions.mp4 build/road/red.avi --start-frame 135 --source-frames 99 --size 1280 720 --source-fps 30 --stride 3 --resize 480 270

python -m traffic_trajectory benchmarks/road_validation/wrong.scene.json --video build/road/wrong.avi --model object_detection_yolox_2022nov.onnx --confidence 0.08 --detection-roi 78 85 211 240 > build/road/wrong.json
python -m traffic_trajectory benchmarks/road_validation/stop.scene.json --video build/road/stop.avi --model object_detection_yolox_2022nov.onnx --confidence 0.08 > build/road/stop.json
python -m traffic_trajectory benchmarks/road_validation/red.scene.json --video build/road/red.avi --model object_detection_yolox_2022nov.onnx > build/road/red.json

python -m traffic_trajectory.evaluate build/road/wrong.json benchmarks/road_validation/wrong.labels.json --tolerance-s 1
python -m traffic_trajectory.evaluate build/road/stop.json benchmarks/road_validation/stop.labels.json --tolerance-s 1
python -m traffic_trajectory.evaluate build/road/red.json benchmarks/road_validation/red.labels.json --tolerance-s 0.5
```

Omit `--confidence` and `--detection-roi` for the full-frame 0.35 runs. For the highway stop crop, add `--detection-roi 130 25 260 146` to the stop command. Use `--include-tracks` to inspect how long each vehicle track survives.
