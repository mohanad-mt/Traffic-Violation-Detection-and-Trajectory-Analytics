import argparse
import json
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass
from math import dist, isfinite
from pathlib import Path
from statistics import mean

from .geometry import Point
from .scenario import Observation, _array, _id, _number, _object, _parse_track

type EventKey = tuple[str, str]


@dataclass(frozen=True)
class EventTime:
    index: int
    start_s: float
    confirmed_s: float
    track_id: str | None
    trajectory: tuple[Observation, ...]


@dataclass(frozen=True)
class Trajectory:
    timestamps: tuple[float, ...]
    positions: tuple[Point, ...]

    def position_at(self, timestamp_s: float, max_gap_s: float) -> Point | None:
        right = bisect_left(self.timestamps, timestamp_s)
        if right < len(self.timestamps) and self.timestamps[right] == timestamp_s:
            return self.positions[right]
        if right == 0 or right == len(self.timestamps):
            return None
        start, end = self.timestamps[right - 1 : right + 1]
        if end - start > max_gap_s:
            return None
        fraction = (timestamp_s - start) / (end - start)
        (ax, ay), (bx, by) = self.positions[right - 1 : right + 1]
        return ax + fraction * (bx - ax), ay + fraction * (by - ay)


def _events(document: dict, *, predictions: bool) -> dict[EventKey, list[EventTime]]:
    groups = defaultdict(list)
    for index, item in enumerate(_array(document.get("events"), "events")):
        path = f"events[{index}]"
        event = _object(item, path)
        lane = _id(event.get("lane_id"), f"{path}.lane_id")
        kind = _id(event.get("type"), f"{path}.type")
        start = _number(event.get("start_s"), f"{path}.start_s")
        confirmed = (
            _number(event.get("confirmed_s"), f"{path}.confirmed_s") if predictions else start
        )
        if not 0 <= start <= confirmed:
            raise ValueError(f"{path} must have 0 <= start_s <= confirmed_s")
        track_id = _id(event["track_id"], f"{path}.track_id") if "track_id" in event else None
        trajectory = ()
        if not predictions and "trajectory" in event:
            trajectory = _parse_track(
                {"id": "reference", "observations": event["trajectory"]}, f"{path}.trajectory"
            ).observations
            if not trajectory:
                raise ValueError(f"{path}.trajectory must contain observations")
        groups[lane, kind].append(EventTime(index, start, confirmed, track_id, trajectory))
    return groups


def _trajectories(document: dict) -> dict[str, Trajectory]:
    tracks = {}
    for index, raw in enumerate(_array(document.get("tracks", []), "tracks")):
        track = _parse_track(raw, f"tracks[{index}]")
        if track.id in tracks:
            raise ValueError("prediction track IDs must be unique")
        tracks[track.id] = Trajectory(
            tuple(sample.timestamp_s for sample in track.observations),
            tuple(sample.position for sample in track.observations),
        )
    return tracks


def _spatial_error(
    label: EventTime,
    trajectory: Trajectory,
    radius_px: float,
    max_gap_s: float,
    min_samples: int,
) -> float | None:
    errors = []
    for sample in label.trajectory:
        position = trajectory.position_at(sample.timestamp_s, max_gap_s)
        if position is not None:
            error = dist(sample.position, position)
            if error <= radius_px:
                errors.append(error)
    if len(errors) < max(min_samples, len(label.trajectory) // 2 + 1):
        return None
    return mean(errors)


def _one_to_one(candidates: list[list[int]]) -> dict[int, int]:
    """Find a maximum-cardinality matching, preferring free candidates in list order."""
    assigned = {}
    label_matches = {}
    for root in range(len(candidates)):
        search = deque([root])
        visited = {root}
        via = {}
        while search:
            label = search.popleft()
            for prediction in candidates[label]:
                if prediction in via:
                    continue
                via[prediction] = label
                if prediction not in assigned:
                    break
                other = assigned[prediction]
                if other not in visited:
                    visited.add(other)
                    search.append(other)
            else:
                continue
            # Move the prior assignments along the path to this free prediction.
            while prediction is not None:
                label = via[prediction]
                assigned[prediction] = label
                prediction, label_matches[label] = label_matches.get(label), prediction
            break
    return label_matches


@dataclass(frozen=True)
class Match:
    label: EventTime
    prediction: EventTime


def _metrics(
    actual_count: int,
    predicted_count: int,
    matches: list[Match],
    duplicates: int,
    duration_s: float | None,
) -> dict:
    tp = len(matches)
    false_positives = predicted_count - tp
    denominator = actual_count + predicted_count
    return {
        "true_positives": tp,
        "false_positives": false_positives,
        "false_negatives": actual_count - tp,
        "precision": tp / predicted_count if predicted_count else None,
        "recall": tp / actual_count if actual_count else None,
        "f1": 2 * tp / denominator if denominator else None,
        "mean_confirmation_delay_s": mean(
            pair.prediction.confirmed_s - pair.label.start_s for pair in matches
        )
        if matches
        else None,
        "mean_abs_onset_error_s": mean(
            abs(pair.prediction.start_s - pair.label.start_s) for pair in matches
        )
        if matches
        else None,
        "duplicate_alerts": duplicates,
        "duplicate_alert_rate": duplicates / predicted_count if predicted_count else None,
        "false_alarms_per_hour": 3600 * false_positives / duration_s if duration_s else None,
    }


def evaluate(
    predictions: object,
    labels: object,
    tolerance_s: float,
    *,
    identity_required: bool = False,
    radius_px: float | None = None,
    max_interpolation_gap_s: float = 0.75,
    min_spatial_samples: int = 2,
    duration_s: float | None = None,
) -> dict:
    if not isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if radius_px is not None and (not isfinite(radius_px) or radius_px <= 0):
        raise ValueError("spatial radius must be finite and positive")
    if not isfinite(max_interpolation_gap_s) or max_interpolation_gap_s <= 0:
        raise ValueError("maximum interpolation gap must be finite and positive")
    if type(min_spatial_samples) is not int or min_spatial_samples < 1:
        raise ValueError("minimum spatial samples must be a positive integer")
    label_doc = _object(labels, "labels")
    prediction_doc = _object(predictions, "predictions")
    if duration_s is None and "duration_s" in label_doc:
        duration_s = _number(label_doc["duration_s"], "duration_s")
    if duration_s is not None and (not isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration must be finite and positive")
    actual = _events(label_doc, predictions=False)
    predicted = _events(prediction_doc, predictions=True)
    has_trajectory = False
    time_only = 0
    for events in actual.values():
        for label in events:
            if label.trajectory:
                has_trajectory = True
                if radius_px is None or len(label.trajectory) < min_spatial_samples:
                    raise ValueError(
                        f"label {label.index} needs a spatial radius and enough trajectory samples"
                    )
            elif label.track_id is None:
                time_only += 1
                if identity_required:
                    raise ValueError(f"label {label.index} needs a track_id or trajectory")
    trajectories = _trajectories(prediction_doc) if has_trajectory else {}

    actual_count = defaultdict(int)
    predicted_count = defaultdict(int)
    matches = defaultdict(list)
    duplicate_indices = defaultdict(list)
    missed_indices = []
    for lane, kind in sorted(actual.keys() | predicted.keys()):
        labeled = actual.get((lane, kind), [])
        alerts = predicted.get((lane, kind), [])
        actual_count[kind] += len(labeled)
        predicted_count[kind] += len(alerts)
        candidates = []
        for label in labeled:
            eligible = []
            for index, alert in enumerate(alerts):
                offset = abs(alert.start_s - label.start_s)
                if offset > tolerance_s or (
                    label.track_id is not None and label.track_id != alert.track_id
                ):
                    continue
                error = 0.0
                if label.trajectory:
                    trajectory = trajectories.get(alert.track_id)
                    if trajectory is None:
                        raise ValueError(f"alert {alert.index} is missing its predicted track")
                    error = _spatial_error(
                        label, trajectory, radius_px, max_interpolation_gap_s, min_spatial_samples
                    )
                    if error is None:
                        continue
                eligible.append((offset, error, index))
            candidates.append([index for _, _, index in sorted(eligible)])
        pairs = _one_to_one(candidates)
        matches[kind].extend(Match(labeled[i], alerts[j]) for i, j in pairs.items())
        missed_indices.extend(label.index for i, label in enumerate(labeled) if i not in pairs)
        extras = {j for options in candidates for j in options} - set(pairs.values())
        duplicate_indices[kind].extend(alerts[j].index for j in sorted(extras))

    all_matches = [pair for values in matches.values() for pair in values]
    all_duplicates = sorted(index for values in duplicate_indices.values() for index in values)
    return {
        "tolerance_s": tolerance_s,
        "spatial_radius_px": radius_px,
        "time_only_labels": time_only,
        "duration_s": duration_s,
        "overall": _metrics(
            sum(actual_count.values()),
            sum(predicted_count.values()),
            all_matches,
            len(all_duplicates),
            duration_s,
        ),
        "by_type": {
            kind: _metrics(
                actual_count[kind],
                predicted_count[kind],
                matches[kind],
                len(duplicate_indices[kind]),
                duration_s,
            )
            for kind in sorted(actual_count)
        },
        "matches": [
            {"label_index": pair.label.index, "prediction_index": pair.prediction.index}
            for pair in sorted(all_matches, key=lambda pair: pair.label.index)
        ],
        "missed_label_indices": sorted(missed_indices),
        "duplicate_prediction_indices": all_duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate events against scene labels.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    parser.add_argument("--identity-required", action="store_true")
    parser.add_argument("--radius-px", type=float, help="Maximum distance from a reference point")
    parser.add_argument("--max-interpolation-gap-s", type=float, default=0.75)
    parser.add_argument("--min-spatial-samples", type=int, default=2)
    parser.add_argument(
        "--duration-s", type=float, help="Duration of the annotated video coverage"
    )
    args = parser.parse_args()
    try:
        report = evaluate(
            json.loads(args.predictions.read_text(encoding="utf-8-sig")),
            json.loads(args.labels.read_text(encoding="utf-8-sig")),
            args.tolerance_s,
            identity_required=args.identity_required,
            radius_px=args.radius_px,
            max_interpolation_gap_s=args.max_interpolation_gap_s,
            min_spatial_samples=args.min_spatial_samples,
            duration_s=args.duration_s,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
