# Traffic Trajectory Analytics

A traffic event detector for fixed-camera footage. Give the CLI a video or a JSON file of vehicle tracks. It reports wrong-way movement, stopped vehicles, and red-light crossing candidates with onset and confirmation times. The results are for review and research.

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

## Detectors and trackers

Use a local video and a [scene JSON](examples/video.scene.json) with lane coordinates in video pixels. Include signal timing relative to the first decoded frame to enable the red-light rule.

- `--model` takes a local YOLOX ONNX file. The default RGB preprocessing works with the full-precision OpenCV Zoo YOLOX-S export. Use `--yolox-input-color bgr` for the upstream YOLOX-M export. [Zoo checkpoint](https://huggingface.co/opencv/opencv_zoo/tree/main/models/object_detection_yolox) · [Upstream weights](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md)
- `--detector rtdetr --model build/assets/rtdetr-r18` uses a local RT-DETR checkpoint. Fetch the pinned files with `python -m benchmarks.assets build/assets rtdetr-config rtdetr-processor rtdetr-weights`.
- `--detector rfdetr --model build/assets/rf-detr-small.pth` uses RF-DETR-S. Fetch it with `python -m benchmarks.assets build/assets rfdetr-small`. [Comparison on fog images and road clips](benchmarks/UPGRADE.md#rf-detr-small)
- `--detector rfdetr-large --model build/assets/rf-detr-large-2026.pth` uses the 704 px RF-DETR-L checkpoint. Fetch it with `python -m benchmarks.assets build/assets rfdetr-large`. [Small vs. Large benchmark](benchmarks/UPGRADE.md#external-references-and-pipeline-revisions)
- `--background empty-road.png` compares each frame against an empty-road image from the same camera and resolution. Shadows, lighting changes, and other objects can show up as foreground detections.

```sh
python -m traffic_trajectory scene.json --video clip.mp4 --model object_detection_yolox_2022nov.onnx
```

The tracker predicts each vehicle's next bottom center from its last two positions. It matches the closest detection within `--track-distance-px` and drops tracks after `max_gap_s` without an observation. Occlusion and overlapping boxes can cause ID switches. Use `--include-tracks` to inspect the positions and IDs behind an alert.

Select `--tracker bytetrack` or `--tracker ocsort` to pass detector boxes to the Roboflow tracking implementations. Install `requirements-trackers.txt` in a Python 3.13 environment for these modes. That file installs the regular OpenCV wheel, so use it in place of `requirements-video.txt` for an environment with the box trackers.

For YOLOX, `--continuation-confidence` lets a lower-score detection extend an existing track within half the normal distance gate. Only detections above `--confidence` can start a track. `--detection-roi LEFT TOP RIGHT BOTTOM` runs the detector on part of the frame. Keep the scene coordinates in full-frame pixels and leave room around vehicles at the crop edges.

Use `--detection-tile-size 640` to add overlapping crops when distant vehicles are too small in the full view. The detector keeps the full-view pass and merges duplicate boxes. Use `--tile-confidence` to require a higher score for crop boxes before merging. Crops can improve recall at the cost of runtime and extra detections. [Foggy Driving results](benchmarks/foggy_driving/README.md)

### CUDA setup

On Windows, use an NVIDIA driver that supports CUDA 13 and install the CUDA 13 PyTorch wheels if CUDA and cuDNN 9 are not already available. ONNX Runtime can load the libraries from those wheels. Use `--device cuda` for acceleration or `--device cpu` for CPU execution. A CUDA request exits with an error if ONNX Runtime cannot load its CUDA provider.

```sh
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-cuda.txt
python -m traffic_trajectory scene.json --video clip.mp4 --model object_detection_yolox_2022nov.onnx --device cuda --include-tracks
```

For a joint detector/tracker environment install `requirements-trackers.txt` first, then add `onnxruntime-gpu==1.32.0 transformers==5.17.0` and the PyTorch wheels above. RT-DETR uses PyTorch directly. The CPU image and baseline tests use the headless OpenCV wheel. [CUDA provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

RF-DETR also uses PyTorch directly. Install the CUDA PyTorch wheels above, then install `requirements-rfdetr.txt` alongside `requirements-trackers.txt` for box tracking. Use `requirements-video.txt` in place of the tracker requirements for point tracking.

```sh
python -m pip install -r requirements-trackers.txt -r requirements-rfdetr.txt
python -m benchmarks.assets build/assets rfdetr-small
python -m traffic_trajectory scene.json --video clip.mp4 --model build/assets/rf-detr-small.pth --detector rfdetr --device cuda --tracker bytetrack --confidence .20 --continuation-confidence .03 --include-tracks
```

The detector and tracker settings above recovered the development fog alert. Check the score thresholds on footage from the target camera before deploying them.

### Incremental output and video overlays

Add `--stream-events` to flush each confirmed event as a JSON line while decoding a local video. Track and rule state expire as the video advances. The online stop rule can suppress a nearby confirmed stop after a track expires. Offline scoring can also examine short intervening fragments. Use `--strict-video` to check a clip against its declared frame count when that metadata is trustworthy.

Use `--timestamp-source pts` for variable-frame-rate clips and when small timing errors affect signal transitions. PyAV decodes presentation timestamps, and the pipeline measures time from the first decoded frame. For `--include-tracks` output, the overlay renderer reads the timestamp mode from the JSON and preserves frame spacing in the exported MP4. Use the same timestamp mode on both sides of a comparison overlay.

Render a saved `--include-tracks` result with the lane, legal direction, stop line, track IDs, boxes, scores and alerts. The optional comparison uses the same underlying clip.

```sh
python -m traffic_trajectory.visualize scene.json clip.mp4 predictions.json annotated.mp4
python -m traffic_trajectory.visualize scene.json clip.mp4 baseline.json comparison.mp4 --compare candidate.json --title Baseline --compare-title Candidate
```

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

Each lane has an `id`, a convex `polygon` in boundary order, and a nonzero `forward` vector pointing in the legal direction. A scene accepts one `lane` with top-level rules or a `lanes` array with `stop_line` and `stop_exclusions` on each lane. Use image coordinates, with x increasing rightward and y increasing downward. For track input, supply `tracks` with unique IDs and observations in strictly increasing `timestamp_s` order. Use the bottom center of each vehicle box for its `position`.

- **Wrong way.** Follow reverse steps in the lane that meet the heading and observation-gap thresholds. Allow brief steps below `min_step_px` for detection jitter. Emit one event when both the duration and net reverse displacement pass their thresholds.
- **Stopped.** Start a timer when the vehicle enters the lane or moves outside its current stop radius. A tracking gap or lane exit resets the timer. After an alert, a nearby stationary fragment can keep the reported stop active across a short gap. An observed departure allows a new alert.
- **Red light.** Set `stop_line.segment` with two endpoints and `stop_line.red_intervals_s` with ordered, non-overlapping `[start, end)` intervals relative to the first frame. The vehicle must approach and clear a margin on both sides of the line. The red interval is checked at the interpolated crossing time. Use synchronized signal data or annotate the red intervals from the clip. [Example scene](examples/video.scene.json)

Set overrides in `thresholds`. Mark expected queue or parking regions in `stop_exclusions` as convex polygons. Set `min_step_px` to zero and use a positive `min_reverse_speed_px_s` to express the wrong-way step threshold per second. Camera perspective still affects pixel displacements.

| Threshold | Default | Controls |
| --- | ---: | --- |
| `min_reverse_distance_px` | 35 | Cumulative projected reverse displacement |
| `min_duration_s` | 1.0 | Minimum length of a reverse episode |
| `max_gap_s` | 0.75 | Maximum interval between consecutive observations |
| `min_step_px` | 1.0 | Minimum displacement per qualifying step |
| `min_reverse_speed_px_s` | 0.0 | Optional minimum projected reverse speed in image pixels per second |
| `min_heading_cosine` | 0.7 | Minimum cosine with the reverse direction |
| `max_pause_s` | 0.25 | Time a reverse episode can tolerate subthreshold steps |
| `stopped_duration_s` | 3.0 | Minimum time within the stopped radius |
| `stopped_radius_px` | 4 | Maximum distance from the stop anchor |
| `max_stopped_drift_px_s` | 0 | Optional bound on net displacement divided by elapsed time at stop confirmation. Set to 0 to disable. Tune with the target camera's scale and stop examples. |
| `stopped_realert_gap_s` | 2.0 | Longest gap for treating nearby stationary fragments as a reported stop. Set to 0 to disable. |
| `red_line_margin_px` | 2 | Minimum perpendicular distance to arm and confirm a crossing |

For a calibrated camera, run `python -m benchmarks.fit_ground_plane point-pairs.json` using a `unit` and at least four `fit` pairs with `image` and `world` coordinates. Use additional `validation` pairs to check projection error. Copy the resulting `ground_plane` object into the scene and add `--include-ground-tracks` to export projected observations and velocities. The violation rules still use image-space geometry. [Geometry walkthrough](docs/implementation-guide.md#camera-geometry-and-event-rules)

## Evaluation

Annotate event onsets in seconds using the [label fixture](examples/video.labels.json) as a template. Pick thresholds and a matching tolerance before scoring held-out clips.

```sh
python -m traffic_trajectory.evaluate predictions.json labels.json --tolerance-s 0.5
```

Matching uses lane, event type and onset, followed by a shared `track_id` or reference `trajectory` when supplied. For independently annotated reference paths, add `--identity-required --radius-px 12` and generate predictions with `--include-tracks`. The matcher requires spatial agreement on a majority of reference points and at least two samples by default. It interpolates only within bounded observed gaps. Add annotation coverage as `duration_s` to report false alarms per hour. Metrics also include duplicate alerts, onset error and confirmation delay. [Reference annotation](benchmarks/road_validation/red.identity-labels.json)

YOLOX-M improved class-agnostic AP50–95 from 47.1 to 54.2 on DAWN fog images. ByteTrack cut identity switches in both GRAM-RTM tracking sequences. The best development alert setting reached two of four events with no extra alerts and missed both wrong-way events on the new nighttime views. [Results and reproduction](benchmarks/UPGRADE.md) · [Research decisions](benchmarks/RESEARCH.md) · [Earlier experiments](benchmarks/README.md)

## Development

```sh
python -m pip install -r requirements-dev.txt -r requirements-video.txt
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
```

Open PRs against `main`. CI runs the checks above and verifies both the track and video demos in Docker.

Use the [implementation guide](docs/implementation-guide.md) to trace the code and recreate the experiments.
