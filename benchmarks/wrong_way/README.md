# NJIT wrong-way research clip

The NJIT detector log marks wrong-way events at 5 s and 14 s in this 360 × 240 highway clip. The extract starts at 3 s and samples 15 seconds at 7.5 FPS, putting the labels at 2 s and 11 s. The original clip includes detection overlays and the reference times are rounded to seconds. [NJIT demo](https://web.njit.edu/~cliu/NJDOT/DEMOS.html#wrongway_vehicle) · [Timestamp log](https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong-way.txt)

Download [wrong_way_cam1.mp4](https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong_way_cam1.mp4) and use the same full-precision YOLOX-S weights as the [nighttime check](../nighttime/README.md). The source video's SHA-256 is `784058550dfd9278789ee24a0222c533b50322525032b6a7e577f945a26a651f`.

```sh
python benchmarks/wrong_way/prepare_clip.py wrong_way_cam1.mp4 build/wrong-way/sample.avi
python -m traffic_trajectory benchmarks/wrong_way/scene.json --video build/wrong-way/sample.avi --model object_detection_yolox_2022nov.onnx --include-tracks > build/wrong-way/baseline.json
python -m traffic_trajectory.evaluate build/wrong-way/baseline.json benchmarks/wrong_way/labels.json --tolerance-s 1
python -m traffic_trajectory benchmarks/wrong_way/scene.json --video build/wrong-way/sample.avi --model object_detection_yolox_2022nov.onnx --confidence 0.08 --include-tracks > build/wrong-way/low-confidence.json
python -m traffic_trajectory.evaluate build/wrong-way/low-confidence.json benchmarks/wrong_way/labels.json --tolerance-s 1
```

The scene uses a 0.6 px minimum step because the video is low resolution. The polygon covers the left carriageway, where legal traffic moves toward the bottom of the frame. Both video runs use the default tracking gate, duration, reverse distance, and heading thresholds.

| YOLOX confidence | Track IDs | Observations | True positives | False positives | False negatives | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.35 | 25 | 159 | 0 | 0 | 2 | 0.00 |
| 0.08 | 52 | 332 | 1 | 0 | 1 | 0.50 |

At 0.08, the first event started at 2.53 s and confirmed at 3.60 s. Some detections of the second vehicle fell outside the lane or landed on different tracks. Spot checks on full frames and cropped views both showed intermittent detections. The scored runs used full frames.

To rerun the rule trials, copy `tracks` from `low-confidence.json` into a copy of `scene.json` and run the CLI on that JSON. Try `min_duration_s: 0.5` and polygon `[[126, 90], [174, 96], [206, 236], [5, 214]]` individually and together. All three trials matched one event at the same one-second tolerance.
