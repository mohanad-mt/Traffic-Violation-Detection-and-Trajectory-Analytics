import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean

from .scenario import _array, _id, _number, _object

type EventKey = tuple[str, str]


@dataclass(frozen=True)
class EventTime:
    start_s: float
    confirmed_s: float


def _by_lane_and_type(value: object, *, predictions: bool) -> dict[EventKey, list[EventTime]]:
    groups = defaultdict(list)
    document = _object(value, "document")
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
        groups[lane, kind].append(EventTime(start, confirmed))
    return groups


def _match(actual: list[EventTime], predicted: list[EventTime], tolerance: float) -> list[float]:
    remaining = deque(sorted(predicted, key=lambda event: (event.start_s, event.confirmed_s)))
    delays = []
    for label in sorted(actual, key=lambda event: event.start_s):
        while remaining and remaining[0].start_s < label.start_s - tolerance:
            remaining.popleft()
        if remaining and remaining[0].start_s <= label.start_s + tolerance:
            delays.append(remaining.popleft().confirmed_s - label.start_s)
    return delays


def _metrics(actual_count: int, predicted_count: int, delays: list[float]) -> dict:
    tp = len(delays)
    denominator = actual_count + predicted_count
    return {
        "true_positives": tp,
        "false_positives": predicted_count - tp,
        "false_negatives": actual_count - tp,
        "precision": tp / predicted_count if predicted_count else None,
        "recall": tp / actual_count if actual_count else None,
        "f1": 2 * tp / denominator if denominator else None,
        "mean_confirmation_delay_s": mean(delays) if delays else None,
    }


def evaluate(predictions: object, labels: object, tolerance_s: float) -> dict:
    if not isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    actual = _by_lane_and_type(labels, predictions=False)
    predicted = _by_lane_and_type(predictions, predictions=True)
    actual_count = defaultdict(int)
    predicted_count = defaultdict(int)
    delays = defaultdict(list)
    for lane, kind in sorted(actual.keys() | predicted.keys()):
        labeled_events = actual.get((lane, kind), [])
        predicted_events = predicted.get((lane, kind), [])
        actual_count[kind] += len(labeled_events)
        predicted_count[kind] += len(predicted_events)
        delays[kind].extend(_match(labeled_events, predicted_events, tolerance_s))

    return {
        "tolerance_s": tolerance_s,
        "overall": _metrics(
            sum(actual_count.values()),
            sum(predicted_count.values()),
            [delay for values in delays.values() for delay in values],
        ),
        "by_type": {
            kind: _metrics(actual_count[kind], predicted_count[kind], values)
            for kind, values in delays.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate events against scene labels.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    args = parser.parse_args()
    try:
        report = evaluate(
            json.loads(args.predictions.read_text(encoding="utf-8-sig")),
            json.loads(args.labels.read_text(encoding="utf-8-sig")),
            args.tolerance_s,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
