# Traffic Trajectory Analytics

A single-lane traffic event detector for fixed-camera footage. Give the CLI a video or a JSON file of vehicle tracks. It reports wrong-way movement, stopped vehicles, and red-light crossing candidates with onset and confirmation times.

## Quick start

From the repo root, use Python 3.13 to generate and score the demo clip.

```sh
python -m pip install -r requirements-video.txt
python examples/generate_video.py build/demo
python -m traffic_trajectory examples/video.scene.json --video build/demo/demo.avi --background build/demo/background.png > build/demo/events.json
python -m traffic_trajectory.evaluate build/demo/events.json examples/video.labels.json
```

| Candidate | Onset (s) | Confirmation (s) |
| --- | ---: | ---: |
| Wrong-way movement | 0.0 | 1.0 |
| Stopped vehicle | 0.0 | 3.0 |
| Red-light crossing | 3.0 | 3.1 |

The demo matches all three [labels](examples/video.labels.json) within 0.5 seconds of onset, with no extra alerts. A fourth vehicle crosses after the red phase.

## Camera footage

Use a constant-frame-rate clip and a [scene JSON](examples/video.scene.json) with lane coordinates in video pixels. Include signal timing to enable the red-light rule.

- `--model` takes a local YOLOX-S ONNX file and keeps cars, buses, trucks, and motorcycles above `--confidence`. Use the full-precision [OpenCV Zoo export](https://huggingface.co/opencv/opencv_zoo/tree/main/models/object_detection_yolox).
- `--background empty-road.png` compares each frame against an empty-road image from the same camera and resolution. Shadows, lighting changes, and other objects can show up as foreground detections.

```sh
python -m traffic_trajectory scene.json --video clip.mp4 --model object_detection_yolox_2022nov.onnx
```

The tracker predicts each vehicle's next bottom center from its last two positions. It matches the closest detection within `--track-distance-px` and drops tracks after `max_gap_s` without an observation. Occlusion and overlapping boxes can cause ID switches. Use `--include-tracks` to inspect the positions and IDs behind an alert.

For YOLOX, `--continuation-confidence` lets a lower-score detection extend an existing track within half the normal distance gate. Only detections above `--confidence` can start a track. `--detection-roi LEFT TOP RIGHT BOTTOM` runs the detector on part of the frame. Keep the scene coordinates in full-frame pixels and leave room around vehicles at the crop edges.

## Docker

Generate the demo clip from Quick start, then run this from the repo root in Bash or WSL.

```sh
docker build -t traffic-trajectory .
docker run --rm -i --network=none --read-only \
  --mount "type=bind,src=$PWD/build/demo,dst=/demo,readonly" \
  traffic-trajectory - --video /demo/demo.avi --background /demo/background.png \
  < examples/video.scene.json
```

## Precomputed tracks

The track-only path uses the Python standard library.

```sh
python -m traffic_trajectory examples/demo.json
```

With the Docker image built, pipe the same fixture into the container.

```sh
docker run --rm -i --network=none --read-only traffic-trajectory - < examples/demo.json
```

The Docker equivalent in PowerShell is `Get-Content -Raw examples/demo.json | docker run --rm -i --network=none --read-only traffic-trajectory -`.

The [track fixture](examples/demo.json) emits one wrong-way candidate after 1.0 second and 40 pixels of reverse travel. The forward-moving and out-of-lane tracks produce no alerts. [Expected JSON](examples/demo.expected.json)

## Scene configuration

Every scene has a `lane` with an `id`, a convex `polygon` in boundary order, and a nonzero `forward` vector pointing in the legal direction. Use image coordinates, with x increasing rightward and y increasing downward. For track input, supply `tracks` with unique IDs and observations in strictly increasing `timestamp_s` order. Use the bottom center of each vehicle box for its `position`.

- **Wrong way.** Follow reverse steps in the lane that meet the heading and observation-gap thresholds. Allow brief steps below `min_step_px` for detection jitter. Emit one event when both the duration and net reverse displacement pass their thresholds.
- **Stopped.** Start a timer when the vehicle enters the lane or moves outside its current stop radius. A tracking gap or lane exit resets the timer. After an alert, a nearby stationary fragment can keep the reported stop active across a short gap. An observed departure allows a new alert.
- **Red light.** Set `stop_line.segment` with two endpoints and `stop_line.red_intervals_s` with ordered, non-overlapping `[start, end)` intervals relative to the first frame. The vehicle must approach and clear a margin on both sides of the line. The red interval is checked at the interpolated crossing time. Use synchronized signal data or annotate the red intervals from the clip. [Example scene](examples/video.scene.json)

Set overrides in `thresholds`. Pixel-based thresholds depend on camera perspective and frame rate. The stopped rule can flag cars waiting in a queue. Output is sorted by confirmation time and track ID.

| Threshold | Default | Controls |
| --- | ---: | --- |
| `min_reverse_distance_px` | 35 | Cumulative projected reverse displacement |
| `min_duration_s` | 1.0 | Minimum length of a reverse episode |
| `max_gap_s` | 0.75 | Maximum interval between consecutive observations |
| `min_step_px` | 1.0 | Minimum displacement per qualifying step |
| `min_heading_cosine` | 0.7 | Minimum cosine with the reverse direction |
| `max_pause_s` | 0.25 | Time a reverse episode can tolerate subthreshold steps |
| `stopped_duration_s` | 3.0 | Minimum time within the stopped radius |
| `stopped_radius_px` | 4 | Maximum distance from the stop anchor |
| `stopped_realert_gap_s` | 2.0 | Longest gap for treating nearby stationary fragments as a reported stop. Set to 0 to disable. |
| `red_line_margin_px` | 2 | Minimum perpendicular distance to arm and confirm a crossing |

## Evaluation

Annotate event onsets in seconds using the [label fixture](examples/video.labels.json) as a template. Pick thresholds and a matching tolerance before scoring held-out clips.

```sh
python -m traffic_trajectory.evaluate predictions.json labels.json --tolerance-s 0.5
```

The evaluator matches events one-to-one by lane, event type, and onset time. It reports precision, recall, F1, and mean confirmation delay. Review simultaneous events of the same type in the same lane by vehicle since generated track IDs can differ from the annotations.

The real-road checks cover a nighttime lane with no events, highway wrong-way and stopped-vehicle examples, and one red-light crossing. The recorded runs show where low resolution or intermittent detections break a track. See the [experiment log](benchmarks/README.md) for settings, results, and failed trials.

## Development

```sh
python -m pip install -r requirements-dev.txt -r requirements-video.txt
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
```

Open PRs against `main`. CI runs the checks above and verifies both the track and video demos in Docker.
