import importlib.util
import json
import os
import subprocess
import sys
import unittest
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from traffic_trajectory.evaluate import evaluate
from traffic_trajectory.tracker import PointTracker

ROOT = Path(__file__).resolve().parents[1]
VIDEO_INSTALLED = all(importlib.util.find_spec(name) for name in ("cv2", "numpy"))


class TrackingTests(unittest.TestCase):
    def test_online_point_tracker_discards_expired_histories(self):
        tracker = PointTracker(max_gap_s=0.75, max_distance_px=5, keep_history=False)
        for i in range(100):
            tracker.update(i * 0.2, [(i, 0)])
        (track,) = tracker.tracks()
        self.assertEqual(len(track.observations), 2)
        tracker.update(30, [(300, 0)])
        self.assertEqual([track.id for track in tracker.tracks()], ["vehicle-2"])

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

    def test_weak_detection_continues_a_track_without_starting_one(self):
        tracker = PointTracker(max_gap_s=0.75, max_distance_px=10)
        tracker.update(0.0, [(10, 20)], weak_detections=[(40, 20)])
        tracker.update(0.25, [], weak_detections=[(14, 20), (40, 20)])
        tracker.update(0.5, [(18, 20)], weak_detections=[(17, 20)])
        tracker.update(1.5, [], weak_detections=[(22, 20)])
        (track,) = tracker.tracks()

        self.assertEqual(
            [sample.position for sample in track.observations], [(10, 20), (14, 20), (18, 20)]
        )


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

    def test_reference_path_rejects_a_nearby_events_wrong_vehicle(self):
        event = {"type": "wrong_way_candidate", "lane_id": "A", "start_s": 1.1}

        def observation(time, x):
            return {"timestamp_s": time, "position": [x, 15]}

        labels = {
            "duration_s": 36,
            "events": [
                {
                    **event,
                    "trajectory": [observation(1.1, 12), observation(1.3, 16)],
                }
            ],
        }
        predictions = {
            "events": [
                {**event, "track_id": track, "confirmed_s": 1.5}
                for track in ("other", "match", "match")
            ],
            "tracks": [
                {"id": "other", "observations": [observation(1.0, 40), observation(1.5, 50)]},
                {"id": "match", "observations": [observation(1.0, 10), observation(1.5, 20)]},
            ],
        }
        result = evaluate(predictions, labels, 0.5, radius_px=1, identity_required=True)
        self.assertEqual(result["matches"], [{"label_index": 0, "prediction_index": 1}])
        self.assertEqual(result["duplicate_prediction_indices"], [2])
        self.assertEqual(result["overall"]["false_alarms_per_hour"], 200)
        self.assertEqual(result["overall"]["mean_abs_onset_error_s"], 0)
        self.assertEqual(result["overall"]["duplicate_alert_rate"], 1 / 3)

        too_sparse = evaluate(predictions, labels, 0.5, radius_px=1, max_interpolation_gap_s=0.25)
        self.assertEqual(too_sparse["missed_label_indices"], [0])

    def test_shared_track_id_and_negative_coverage(self):
        event = {"type": "stopped_vehicle_candidate", "lane_id": "A", "start_s": 1.0}
        predictions = {"events": [{**event, "track_id": "2", "confirmed_s": 2.0}]}
        labels = {"events": [{**event, "track_id": "1"}]}
        self.assertEqual(evaluate(predictions, labels, 0.5)["missed_label_indices"], [0])
        result = evaluate({"events": []}, {"events": [], "duration_s": 30}, 0.5)["overall"]
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["false_alarms_per_hour"], 0)


@unittest.skipUnless(VIDEO_INSTALLED, "install requirements-video.txt for video tests")
class VideoTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("av"), "requires PyAV")
    def test_variable_frame_timing_reaches_tracks_alerts_and_overlay(self):
        import av
        import numpy as np

        from traffic_trajectory.detector import detect_events
        from traffic_trajectory.scenario import parse_scenario, track_record
        from traffic_trajectory.tracker import FrameDetections
        from traffic_trajectory.video import stream_video_events, timed_video, track_video
        from traffic_trajectory.visualize import render

        scene = parse_scenario(
            {
                "lane": {
                    "id": "road",
                    "polygon": [[0, 0], [150, 0], [150, 40], [0, 40]],
                    "forward": [1, 0],
                },
                "tracks": [],
            }
        )
        positions = [120, 116, 104, 88, 72]

        def detector():
            points = iter(positions)
            return lambda frame: FrameDetections([(next(points), 20)])

        with TemporaryDirectory() as folder:
            path = Path(folder) / "input.mp4"
            overlay = Path(folder) / "overlay.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=10)
                stream.width, stream.height = 160, 64
                stream.pix_fmt = "yuv420p"
                stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
                for pts in (40, 140, 440, 840, 1240):
                    frame = av.VideoFrame.from_ndarray(np.zeros((64, 160, 3), np.uint8), "bgr24")
                    frame.pts, frame.time_base = pts, stream.time_base
                    container.mux(stream.encode(frame))
                container.mux(stream.encode())

            tracks = track_video(path, detector(), 0.75, 24, timestamp_source="pts")
            self.assertEqual(
                [o.timestamp_s for o in tracks[0].observations], [0, 0.1, 0.4, 0.8, 1.2]
            )
            events = detect_events(replace(scene, tracks=tracks))
            (event,) = events
            self.assertEqual(event.confirmed_s, 1.2)
            self.assertEqual(
                list(stream_video_events(path, detector(), scene, 24, timestamp_source="pts")),
                events,
            )
            data = {
                "timestamp_source": "pts",
                "events": [asdict(event)],
                "tracks": [track_record(track) for track in tracks],
            }
            render(path, scene, [("PTS", data)], overlay)
            with timed_video(overlay, timestamp_source="pts", strict_length=True) as (_, frames):
                self.assertEqual([time for time, _ in frames], [0, 0.1, 0.4, 0.8, 1.2])

    def test_strict_video_detects_a_truncated_capture(self):
        from traffic_trajectory.video import decoded_frames

        capture = Mock()
        capture.get.return_value = 3
        capture.read.side_effect = [(True, object()), (False, None)]
        with self.assertRaisesRegex(ValueError, "ended after 1 frames, expected 3"):
            list(decoded_frames(capture, strict_length=True))

    @unittest.skipUnless(
        os.getenv("TRAFFIC_YOLOX_MODEL"), "set TRAFFIC_YOLOX_MODEL to smoke-test CUDA"
    )
    def test_real_yolox_cpu_cuda_agree(self):
        import cv2
        import numpy as np

        from traffic_trajectory.yolox import YoloXDetector

        model = Path(os.environ["TRAFFIC_YOLOX_MODEL"])
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[120:160, 120:230] = (20, 125, 200)
        cpu = YoloXDetector(model, 0.35)
        gpu = YoloXDetector(model, 0.35, device="cuda")
        blob = cv2.dnn.blobFromImage(frame, size=(640, 640), swapRB=True)
        for network in (cpu.net, gpu.net):
            network.setInput(blob)
        np.testing.assert_allclose(cpu.net.forward(), gpu.net.forward(), atol=0.003, rtol=0.001)

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

    def test_stream_emits_the_three_events_from_the_generated_fixture(self):
        from examples.generate_video import generate_video
        from traffic_trajectory.scenario import parse_scenario
        from traffic_trajectory.video import foreground_detector, stream_video_events

        with TemporaryDirectory() as directory:
            destination = Path(directory)
            generate_video(destination)
            scene = parse_scenario(
                {
                    **json.loads(
                        (ROOT / "examples" / "video.scene.json").read_text(encoding="utf-8")
                    ),
                    "tracks": [],
                }
            )
            detect = foreground_detector(destination / "background.png", 100)
            events = list(stream_video_events(destination / "demo.avi", detect, scene, 24))
        self.assertEqual(len(events), 3)
        self.assertEqual(
            {event.type for event in events},
            {"red_light_crossing_candidate", "stopped_vehicle_candidate", "wrong_way_candidate"},
        )

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
            detections = YoloXDetector(Path("model.onnx"), 0.35)(frame)

        blob = net.setInput.call_args.args[0]
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob[0, :, 0, 0].tolist(), [210, 100, 30])
        self.assertEqual(blob[0, :, -1, -1].tolist(), [114, 114, 114])
        self.assertEqual(detections.strong, [(160, 82)])
        self.assertEqual(detections.weak, [])

    def test_yolox_separates_weak_vehicle_candidates(self):
        import numpy as np

        from traffic_trajectory.yolox import YoloXDetector

        prediction = np.zeros((1, 8400, 85), dtype=np.float32)
        prediction[0, 20 * 80 + 40, 4:8] = [0.9, 0, 0, 0.9]
        prediction[0, 20 * 80 + 30, 4:8] = [0.2, 0, 0, 0.9]
        net = Mock()
        net.forward.return_value = prediction
        with (
            patch("traffic_trajectory.yolox.Path.is_file", return_value=True),
            patch("traffic_trajectory.yolox.cv2.dnn.readNetFromONNX", return_value=net),
        ):
            detector = YoloXDetector(Path("model.onnx"), 0.35, 0.08)
            detections = detector(np.zeros((160, 320, 3), dtype=np.uint8))

        self.assertEqual(detections.strong, [(160, 82)])
        self.assertEqual(detections.weak, [(120, 82)])

    def test_yolox_crop_returns_full_frame_coordinates(self):
        import numpy as np

        from traffic_trajectory.yolox import YoloXDetector

        prediction = np.zeros((1, 8400, 85), dtype=np.float32)
        prediction[0, 20 * 80 + 40, 4:8] = [0.9, 0, 0, 0.9]
        net = Mock()
        net.forward.return_value = prediction
        frame = np.zeros((320, 640, 3), dtype=np.uint8)
        with (
            patch("traffic_trajectory.yolox.Path.is_file", return_value=True),
            patch("traffic_trajectory.yolox.cv2.dnn.readNetFromONNX", return_value=net),
        ):
            detector = YoloXDetector(Path("model.onnx"), 0.35, roi=(70, 40, 390, 200))
            self.assertEqual(detector(frame).strong, [(230, 122)])
            with self.assertRaisesRegex(ValueError, "detection ROI"):
                detector(frame[:, :300])

    def test_yolox_tiles_merge_duplicate_boxes_and_drop_internal_edge_boxes(self):
        import numpy as np

        from traffic_trajectory.yolox import VehicleBox, YoloXDetector

        duplicate = VehicleBox(360, 100, 40, 30, 0.8)
        partial = VehicleBox(455, 100, 24, 30, 0.9)
        extra = VehicleBox(520, 150, 20, 20, 0.7)
        with (
            patch("traffic_trajectory.yolox.Path.is_file", return_value=True),
            patch("traffic_trajectory.yolox.cv2.dnn.readNetFromONNX"),
            patch.object(
                YoloXDetector,
                "_window_boxes",
                side_effect=[[duplicate], [partial], [extra, duplicate]],
            ) as predict_window,
        ):
            detector = YoloXDetector(Path("model.onnx"), 0.35, tile_size=480)
            boxes = detector.predict_boxes(np.zeros((480, 800, 3), dtype=np.uint8))

        self.assertEqual(
            [call.args[1] for call in predict_window.call_args_list],
            [
                (0, 0, 800, 480),
                (0, 0, 480, 480),
                (320, 0, 800, 480),
            ],
        )
        self.assertEqual(boxes, [duplicate, extra])

    def test_yolox_tile_cutoff_filters_before_merging_with_full_view(self):
        import numpy as np

        from traffic_trajectory.yolox import VehicleBox, YoloXDetector

        full = VehicleBox(80, 80, 30, 30, 0.5)
        duplicate = VehicleBox(80, 80, 30, 30, 0.7)
        extra = VehicleBox(520, 150, 20, 20, 0.8)
        with (
            patch("traffic_trajectory.yolox.Path.is_file", return_value=True),
            patch("traffic_trajectory.yolox.cv2.dnn.readNetFromONNX"),
            patch.object(
                YoloXDetector, "_window_boxes", side_effect=[[full], [duplicate], [extra]]
            ),
        ):
            detector = YoloXDetector(Path("model.onnx"), 0.35, tile_size=480, tile_confidence=0.75)
            boxes = detector.predict_boxes(np.zeros((480, 800, 3), dtype=np.uint8))

        self.assertEqual(boxes, [extra, full])

    def test_box_evaluator_matches_each_annotation_once(self):
        from benchmarks.foggy_driving.evaluate_boxes import _labels, _matched_labels
        from traffic_trajectory.yolox import VehicleBox

        (label,) = _labels("0 11 21 40 40\n1 1 1 30 20\n")
        correct = VehicleBox(10, 20, 30, 20, 0.9)
        duplicate = VehicleBox(10, 20, 30, 20, 0.8)
        distant = VehicleBox(70, 60, 20, 20, 0.95)

        self.assertEqual(label, VehicleBox(10, 20, 30, 20, 1.0, "car"))
        self.assertEqual(_matched_labels([correct, duplicate, distant], [label]), [label])
