# NJIT wrong-way research clip

This exploratory run uses 15 seconds of 360 × 240 highway video sampled at 7.5 FPS. The published detector log reports two events starting at 5 s and 14 s in the source. The extracted clip starts at 3 s, so the corresponding label times are 2 s and 11 s. The source has detection overlays and the reference times are rounded to seconds. [NJIT demo and reference timestamps](https://web.njit.edu/~cliu/NJDOT/DEMOS.html#wrongway_vehicle) · [Timestamp log](https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong-way.txt)

Download [wrong_way_cam1.mp4](https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong_way_cam1.mp4) and use the same full-precision YOLOX-S weights as the [nighttime check](../nighttime/README.md). The source video's SHA-256 is `784058550dfd9278789ee24a0222c533b50322525032b6a7e577f945a26a651f`.

```sh
python benchmarks/wrong_way/prepare_clip.py wrong_way_cam1.mp4 build/wrong-way/sample.avi
python -m traffic_trajectory benchmarks/wrong_way/scene.json --video build/wrong-way/sample.avi --model object_detection_yolox_2022nov.onnx --include-tracks > build/wrong-way/baseline.json
python -m traffic_trajectory.evaluate build/wrong-way/baseline.json benchmarks/wrong_way/labels.json --tolerance-s 1
python -m traffic_trajectory benchmarks/wrong_way/scene.json --video build/wrong-way/sample.avi --model object_detection_yolox_2022nov.onnx --confidence 0.08 --include-tracks > build/wrong-way/low-confidence.json
python -m traffic_trajectory.evaluate build/wrong-way/low-confidence.json benchmarks/wrong_way/labels.json --tolerance-s 1
```

The source resolution motivated a 0.6-pixel minimum step. The lane bounds approximate the left carriageway, with forward motion toward the bottom of the frame. The distance gate, minimum duration, reverse displacement, and heading thresholds use the project defaults in the two video runs.

| YOLOX confidence | Track IDs | Observations | True positives | False positives | False negatives | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.35 | 25 | 159 | 0 | 0 | 2 | 0.00 |
| 0.08 | 52 | 332 | 1 | 0 | 1 | 0.50 |

The 0.08 run matched the first label, with a predicted onset at 2.53 s and confirmation at 3.60 s. Some measurements for the second vehicle fell outside the lane or were split across tracks. Inspecting individual frames at both full size and with a left-carriageway crop showed intermittent detections. A crop was used only for this frame-level diagnosis.

To repeat the rule diagnostics, copy the `tracks` array from `low-confidence.json` into a copy of `scene.json`, then run the CLI in track mode. Set `min_duration_s` to 0.5 for the duration trial. The polygon trial used `[[126, 90], [174, 96], [206, 236], [5, 214]]`. Evaluate each change alone and together at the same one-second tolerance. All three variant runs matched one event.
