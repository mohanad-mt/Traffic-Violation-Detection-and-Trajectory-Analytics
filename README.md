# Traffic Trajectory Analytics

Detect wrong-way movement, stopped vehicles, and red-light crossings in a fixed-camera view of a single lane. The CLI accepts video or precomputed vehicle tracks and returns candidate events with estimated onset and confirmation times.

## Run the video demo

Use Python 3.13. Generate the synthetic clip, transform it into events, and score the results against the [known event labels](examples/video.labels.json).

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

The clip also contains a crossing after the red phase. The demo produces three true positives, zero false positives, and zero false negatives at a 0.5-second onset tolerance. This is a synthetic integration check.

### Run on camera footage

Use a constant-frame-rate clip and create a scene JSON with its lane and, optionally, signal timing. Use the same pixel coordinate system as the video. Add `--include-tracks` to inspect the assigned track IDs and positions alongside the events.

- `--background empty-road.png` extracts foreground blobs by comparing each frame to a clean reference of the same camera and dimensions. It works with fixed framing and stable lighting. It can also pick up shadows and other moving objects.
- `--model object_detection_yolox_2022nov.onnx` uses the full-precision [OpenCV Zoo YOLOX-S COCO export](https://huggingface.co/opencv/opencv_zoo/tree/main/models/object_detection_yolox). Supply the weights as a local file. This path keeps car, bus, truck, and motorcycle detections. `--confidence` sets the score cutoff.

```sh
python -m traffic_trajectory scene.json --video clip.mp4 --model object_detection_yolox_2022nov.onnx
```

The tracker predicts each track's next bottom-center position from its last two observations. It associates the closest detections within `--track-distance-px` and expires tracks after `max_gap_s`. Heavy occlusion and overlapping detections can cause ID switches.

### Run the same clip in Docker

Generate the demo files using the earlier commands, then mount them read-only. This command uses Bash syntax.

```sh
docker build -t traffic-trajectory .
docker run --rm -i --network=none --read-only \
  --mount "type=bind,src=$PWD/build/demo,dst=/demo,readonly" \
  traffic-trajectory - --video /demo/demo.avi --background /demo/background.png \
  < examples/video.scene.json
```

## Run on precomputed tracks

The track-only mode uses the Python standard library. Run the bundled [track scenario](examples/demo.json) from the repository root.

```sh
python -m traffic_trajectory examples/demo.json
```

To run the same scenario in Docker from Bash or WSL:

```sh
docker build -t traffic-trajectory .
docker run --rm -i --network=none --read-only traffic-trajectory - < examples/demo.json
```

The Docker equivalent in PowerShell is `Get-Content -Raw examples/demo.json | docker run --rm -i --network=none --read-only traffic-trajectory -`.

The fixture contains three tracks. The vehicle moving against the lane direction triggers one candidate after 1.0 second and 40 pixels of reverse displacement. The forward-moving and out-of-lane tracks produce no events. The full output is in [demo.expected.json](examples/demo.expected.json).

## Scene inputs and event rules

The input JSON contains `lane` and may include `thresholds` and `stop_line`. Track mode also requires `tracks`. Video mode creates them internally. See the [video scene example](examples/video.scene.json) for the stop line and signal configuration.

- `lane` holds a string `id`, a convex `polygon` in boundary order, and a nonzero `forward` vector in the legal direction of travel. Coordinates use image pixels, with x increasing rightward and y increasing downward.
- Each track has a unique string `id` and an array of `observations`. Each observation has a `position` and a strictly increasing `timestamp_s`. For tracked bounding boxes, use a consistent ground-contact proxy such as bottom center.
- A qualifying step has both endpoints in the lane, falls within the maximum observation gap, and travels far enough and close enough to the reverse heading. Any step outside those criteria ends the episode.
- An episode emits one event when both the elapsed time and the accumulated reverse displacement projected onto the lane axis reach their thresholds. Events are sorted by confirmation time and track ID.
- A track counts as stopped when it stays inside the lane and within `stopped_radius_px` of an anchor point for `stopped_duration_s`. Leaving the radius, leaving the lane, or exceeding the observation gap resets the timer.
- `stop_line.segment` holds two endpoints. `stop_line.red_intervals_s` holds ordered, non-overlapping `[start, end)` intervals measured from the clip's first frame. The red-light rule uses the direction of travel and the time interpolated between observations on either side. A crossing must pass through the finite segment inside the lane. Supply the red intervals from synchronized signal data or annotate them from the clip.

| Threshold | Default | Controls |
| --- | ---: | --- |
| `min_reverse_distance_px` | 35 | Cumulative projected reverse displacement |
| `min_duration_s` | 1.0 | Minimum length of a reverse episode |
| `max_gap_s` | 0.75 | Maximum interval between consecutive observations |
| `min_step_px` | 1.0 | Minimum displacement per qualifying step |
| `min_heading_cosine` | 0.7 | Minimum cosine with the reverse direction |
| `stopped_duration_s` | 3.0 | Minimum time within the stopped radius |
| `stopped_radius_px` | 4 | Maximum distance from the stop anchor |

The rules use image coordinates, so camera perspective and sampling rate affect the thresholds. The stopped rule can also flag ordinary queues. Event outputs are candidates for review.

## Evaluate labeled footage

For each clip, annotate event onset in seconds using the schema in [video.labels.json](examples/video.labels.json). Keep a separate set of clips for choosing thresholds. Run the evaluator per held-out clip with a fixed tolerance.

```sh
python -m traffic_trajectory.evaluate predictions.json labels.json --tolerance-s 0.5
```

The evaluator pairs each prediction with at most one label by lane, event type, and onset tolerance. It reports precision, recall, F1, and mean confirmation delay relative to the annotated onset. Matching uses time and lane because video-generated track IDs can differ from the annotated IDs. Simultaneous events of the same type in the same lane need identity-aware review.

The [nighttime road clip check](benchmarks/nighttime/README.md) uses pretrained YOLOX-S on an eight-second sample of real traffic with a manually annotated lane. It covers one clip with no labeled events.

The [AI City Challenge 2021 Traffic Anomaly Detection track](https://www.aicitychallenge.org/2021-data-and-evaluation/#:~:text=Track%204%3A%20Traffic%20Anomaly%20Detection) provides real traffic footage with timestamps for anomalies such as stalls and crashes. For this project's three event classes, review scene-specific labels and record red phases where visible or available.

## Development

```sh
python -m pip install -r requirements-dev.txt -r requirements-video.txt
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
```

Use a feature branch and open a pull request into `main`. GitHub Actions runs linting, unit tests, and a Docker smoke test against the expected demo output on pull requests and pushes to `main`.
