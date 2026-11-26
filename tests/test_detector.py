import json
import subprocess
import sys
import unittest
from pathlib import Path

from traffic_trajectory.detector import detect_wrong_way
from traffic_trajectory.scenario import parse_scenario

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo.json"


def scenario_with_track(samples):
    return {
        "lane": {
            "id": "eastbound",
            "polygon": [[0, 0], [200, 0], [200, 40], [0, 40]],
            "forward": [1, 0],
        },
        "tracks": [
            {
                "id": "vehicle",
                "observations": [
                    {"timestamp_s": timestamp, "position": [x, y]} for timestamp, x, y in samples
                ],
            }
        ],
    }


class WrongWayDetectionTests(unittest.TestCase):
    def test_demo_reports_one_candidate(self):
        scenario = parse_scenario(json.loads(DEMO.read_text(encoding="utf-8")))
        events = detect_wrong_way(scenario)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].track_id, "vehicle-reversing")
        self.assertEqual(events[0].confirmed_s, 1.0)
        self.assertEqual(events[0].reverse_distance_px, 40.0)

    def test_ignores_short_small_lateral_or_interrupted_motion(self):
        cases = {
            "short duration": [(0, 120, 20), (0.5, 70, 20)],
            "small motion": [(0, 120, 20), (0.5, 116, 20), (1, 112, 20)],
            "stationary": [(0, 120, 20), (0.5, 120, 20), (1, 120, 20)],
            "lateral": [(0, 120, 5), (0.5, 115, 20), (1, 110, 35)],
            "observation gap": [(0, 120, 20), (0.5, 100, 20), (2, 80, 20), (2.5, 60, 20)],
            "leaves lane": [(0, 120, 20), (0.5, 100, 41), (1, 80, 20), (1.5, 60, 20)],
        }
        for name, samples in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    detect_wrong_way(parse_scenario(scenario_with_track(samples))), []
                )

    def test_rearms_after_direction_changes(self):
        payload = scenario_with_track(
            [
                (0, 160, 20),
                (0.5, 140, 20),
                (1, 120, 20),
                (1.5, 100, 20),
                (2, 110, 20),
                (2.5, 90, 20),
                (3, 70, 20),
            ]
        )
        events = detect_wrong_way(parse_scenario(payload))

        self.assertEqual(
            [(event.start_s, event.confirmed_s) for event in events], [(0, 1), (2, 3)]
        )

    def test_respects_configured_direction_and_boundary(self):
        payload = scenario_with_track([(0, 10, 0), (0.5, 30, 0), (1, 50, 0)])
        payload["lane"]["forward"] = [-4, 0]
        payload["lane"]["polygon"].reverse()

        events = detect_wrong_way(parse_scenario(payload))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reverse_distance_px, 40)

    def test_rejects_duplicate_timestamps(self):
        payload = scenario_with_track([(0.5, 120, 20), (0.5, 100, 20)])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_scenario(payload)

    def test_rejects_self_crossing_boundary(self):
        payload = scenario_with_track([])
        payload["lane"]["polygon"] = [[0, 0], [200, 40], [0, 40], [200, 0]]
        with self.assertRaisesRegex(ValueError, "strictly convex"):
            parse_scenario(payload)

    def test_rejects_wrong_threshold_name(self):
        payload = scenario_with_track([])
        payload["thresholds"] = {"min_distance_px": 20}
        with self.assertRaisesRegex(ValueError, "unknown thresholds: min_distance_px"):
            parse_scenario(payload)


class CommandLineTests(unittest.TestCase):
    def test_path_and_stdin_match_expected_output(self):
        expected = (ROOT / "examples" / "demo.expected.json").read_text(encoding="utf-8")
        for argument in (str(DEMO), "-"):
            with self.subTest(argument=argument):
                result = subprocess.run(
                    [sys.executable, "-m", "traffic_trajectory", argument],
                    input=DEMO.read_text(encoding="utf-8") if argument == "-" else None,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertEqual(result.stdout, expected)

    def test_invalid_input_returns_argument_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "traffic_trajectory", "-"],
            input="{}",
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("lane must be an object", result.stderr)
