import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from traffic_trajectory.evaluate import evaluate
from traffic_trajectory.tracker import PointTracker

ROOT = Path(__file__).resolve().parents[1]
VIDEO_INSTALLED = all(importlib.util.find_spec(name) for name in ("cv2", "numpy"))


class TrackingTests(unittest.TestCase):
    def test_velocity_can_continue_a_track_after_a_missed_detection(self):
        tracker = PointTracker(max_gap_s=0.75, max_distance_px=5)
        tracker.update(0.0, [(10, 20)])
        tracker.update(0.25, [(14, 20)])
        tracker.update(0.5, [])
        tracker.update(0.75, [(22, 20)])
        tracker.update(1.75, [(22, 20)])
        first, second = tracker.tracks()

        self.assertEqual([sample.timestamp_s for sample in first.observations], [0, 0.25, 0.75])
        self.assertEqual(second.observations[0].timestamp_s, 1.75)


class EvaluationTests(unittest.TestCase):
    def test_does_not_match_duplicates_or_other_lanes(self):
        event = {"type": "stopped_vehicle_candidate", "lane_id": "A", "start_s": 1.0}
        labels = {"events": [event, {**event, "lane_id": "B"}]}
        predictions = {"events": [{**event, "confirmed_s": 4.0}] * 2}
        report = evaluate(predictions, labels, 0.5)["overall"]

        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["recall"], 0.5)
        self.assertEqual(report["mean_confirmation_delay_s"], 3.0)

    def test_matches_earlier_candidates_when_tolerances_overlap(self):
        event = {"type": "wrong_way_candidate", "lane_id": "A"}
        labels = {"events": [{**event, "start_s": t} for t in (1.0, 1.5)]}
        predictions = {
            "events": [{**event, "start_s": t, "confirmed_s": t + 1} for t in (0.5, 1.125)]
        }
        self.assertEqual(evaluate(predictions, labels, 0.5)["overall"]["true_positives"], 2)


@unittest.skipUnless(VIDEO_INSTALLED, "install requirements-video.txt for video tests")
class VideoTests(unittest.TestCase):
    def test_video_cli_reproduces_all_three_labels(self):
        from examples.generate_video import generate_video

        with TemporaryDirectory() as directory:
            destination = Path(directory)
            generate_video(destination)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "traffic_trajectory",
                    "examples/video.scene.json",
                    "--video",
                    str(destination / "demo.avi"),
                    "--background",
                    str(destination / "background.png"),
                    "--include-tracks",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
        predictions = json.loads(result.stdout)
        labels = json.loads((ROOT / "examples" / "video.labels.json").read_text(encoding="utf-8"))
        report = evaluate(predictions, labels, 0.25)["overall"]

        self.assertEqual(report["true_positives"], 3)
        self.assertEqual(report["f1"], 1)
        self.assertEqual(len(predictions["tracks"]), 4)

    def test_yolox_preprocess_and_vehicle_decoding(self):
        import numpy as np

        from traffic_trajectory.yolox import YoloXDetector

        prediction = np.zeros((1, 8400, 85), dtype=np.float32)
        car_index = 20 * 80 + 40
        pedestrian_index = 20 * 80 + 30
        prediction[0, car_index, 4:8] = [0.9, 0, 0, 0.9]
        prediction[0, pedestrian_index, 4:6] = [0.95, 0.95]
        net = Mock()
        net.forward.return_value = prediction
        frame = np.full((160, 320, 3), [30, 100, 210], dtype=np.uint8)

        with (
            patch("traffic_trajectory.yolox.Path.is_file", return_value=True),
            patch("traffic_trajectory.yolox.cv2.dnn.readNetFromONNX", return_value=net),
        ):
            points = YoloXDetector(Path("model.onnx"), 0.35)(frame)

        blob = net.setInput.call_args.args[0]
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob[0, :, 0, 0].tolist(), [210, 100, 30])
        self.assertEqual(blob[0, :, -1, -1].tolist(), [114, 114, 114])
        self.assertEqual(points, [(160, 82)])
