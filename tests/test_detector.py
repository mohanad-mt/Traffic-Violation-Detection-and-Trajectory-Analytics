import json
import subprocess
import sys
import unittest
from pathlib import Path

from traffic_trajectory.detector import detect_events, detect_wrong_way
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


class AdditionalEventTests(unittest.TestCase):
    def test_wrong_way_tolerates_brief_jitter_and_uses_net_displacement(self):
        payload = scenario_with_track(
            [(0, 120, 20), (0.4, 104, 20), (0.5, 104.5, 20), (0.9, 90, 20), (1.1, 80, 20)]
        )
        (event,) = detect_wrong_way(parse_scenario(payload))

        self.assertEqual(event.start_s, 0)
        self.assertEqual(event.confirmed_s, 1.1)
        self.assertEqual(event.reverse_distance_px, 40)
        payload["thresholds"] = {"max_pause_s": 0}
        self.assertEqual(detect_wrong_way(parse_scenario(payload)), [])

    def test_stop_rearms_after_motion_and_uses_anchor_radius(self):
        positions = [20, 21, 23, 20, 25, 26, 25, 24]
        payload = scenario_with_track([(0.5 * index, x, 20) for index, x in enumerate(positions)])
        payload["thresholds"] = {"stopped_duration_s": 1.0}
        events = detect_events(parse_scenario(payload))

        self.assertEqual(
            [(event.start_s, event.confirmed_s) for event in events], [(0, 1), (2, 3)]
        )
        self.assertTrue(all(event.type == "stopped_vehicle_candidate" for event in events))

    def test_gap_resets_the_stop_timer(self):
        payload = scenario_with_track([(0, 20, 20), (0.5, 20, 20), (3, 20, 20), (3.5, 20, 20)])
        payload["thresholds"] = {"stopped_duration_s": 1.0}

        self.assertEqual(detect_events(parse_scenario(payload)), [])

    def test_crossing_uses_interpolated_time_and_half_open_red_intervals(self):
        payload = scenario_with_track([(0, 45, 20), (0.5, 55, 20)])
        payload["stop_line"] = {"segment": [[50, 10], [50, 30]], "red_intervals_s": [[0.25, 1]]}
        (event,) = detect_events(parse_scenario(payload))

        self.assertEqual(event.type, "red_light_crossing_candidate")
        self.assertEqual((event.start_s, event.confirmed_s), (0.25, 0.5))
        payload["stop_line"]["red_intervals_s"] = [[0, 0.25]]
        self.assertEqual(detect_events(parse_scenario(payload)), [])

    def test_requires_an_approach_to_the_finite_line_without_a_tracking_gap(self):
        cases = {
            "starts on the line": [(0, 50, 20), (0.5, 55, 20)],
            "outside the stop line": [(0, 45, 35), (0.5, 55, 35)],
            "touches then retreats": [(0, 45, 20), (0.5, 50, 20), (1, 45, 20)],
            "travels backward": [(0, 55, 20), (0.5, 45, 20)],
            "tracking gap": [(0, 45, 20), (2, 55, 20)],
        }
        for name, samples in cases.items():
            with self.subTest(name=name):
                payload = scenario_with_track(samples)
                payload["stop_line"] = {
                    "segment": [[50, 30], [50, 10]],
                    "red_intervals_s": [[0, 3]],
                }
                self.assertEqual(detect_events(parse_scenario(payload)), [])

    def test_ignores_lateral_movement_across_an_angled_line(self):
        payload = scenario_with_track([(0, 47, 25), (0.5, 47, 13)])
        payload["stop_line"] = {"segment": [[45, 10], [55, 30]], "red_intervals_s": [[0, 1]]}

        self.assertEqual(detect_events(parse_scenario(payload)), [])

    def test_red_light_ignores_jitter_that_never_clears_the_line_margin(self):
        payload = scenario_with_track(
            [(0, 46, 20), (0.2, 49, 20), (0.4, 50.5, 20), (0.6, 49.5, 20), (0.8, 50.5, 20)]
        )
        payload["stop_line"] = {"segment": [[50, 10], [50, 30]], "red_intervals_s": [[0, 1]]}

        self.assertEqual(detect_events(parse_scenario(payload)), [])

    def test_red_light_confirmation_preserves_crossing_time(self):
        payload = scenario_with_track(
            [(0, 46, 20), (0.5, 51, 20), (0.7, 54, 20), (0.8, 49, 20), (0.9, 53, 20)]
        )
        payload["stop_line"] = {"segment": [[50, 10], [50, 30]], "red_intervals_s": [[0, 0.6]]}
        (event,) = detect_events(parse_scenario(payload))

        self.assertAlmostEqual(event.start_s, 0.4)
        self.assertEqual(event.confirmed_s, 0.7)
