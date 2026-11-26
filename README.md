# Traffic Trajectory Analytics

Detect candidate traffic events from timestamped vehicle trajectories and lane geometry. The first milestone detects wrong-way movement in a single lane and records when each event starts and meets the detection thresholds.

## Run the demo

From the repository root, use Python 3.13 to run the bundled synthetic scenario.

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

## Input and event rule

The input is a JSON object containing `lane`, `tracks`, and optional `thresholds`. The [demo scenario](examples/demo.json) is a complete example.

- `lane` holds a string `id`, a convex `polygon` in boundary order, and a nonzero `forward` vector in the legal direction of travel. Coordinates use image pixels, with x increasing rightward and y increasing downward.
- Each track has a unique string `id` and an array of `observations`. Each observation has a `position` and a strictly increasing `timestamp_s`. For tracked bounding boxes, use a consistent ground-contact proxy such as bottom center.
- A qualifying step has both endpoints in the lane, falls within the maximum observation gap, and travels far enough and close enough to the reverse heading. Any step outside those criteria ends the episode.
- An episode emits one event when both the elapsed time and the accumulated reverse displacement projected onto the lane axis reach their thresholds. Events are sorted by confirmation time and track ID.

| Threshold | Default | Controls |
| --- | ---: | --- |
| `min_reverse_distance_px` | 35 | Cumulative projected reverse displacement |
| `min_duration_s` | 1.0 | Minimum length of a reverse episode |
| `max_gap_s` | 0.75 | Maximum interval between consecutive observations |
| `min_step_px` | 1.0 | Minimum displacement per qualifying step |
| `min_heading_cosine` | 0.7 | Minimum cosine with the reverse direction |

This baseline consumes tracks supplied by the caller and uses pixel-space thresholds that depend on camera perspective and sampling rate. The demo and tests use synthetic trajectories. Real-world evaluation requires labeled footage, lane annotations, and metrics for event precision, recall and confirmation delay.

## Development

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
```

Use a feature branch and open a pull request into `main`. GitHub Actions runs linting, unit tests, and a Docker smoke test against the expected demo output on pull requests and pushes to `main`.
