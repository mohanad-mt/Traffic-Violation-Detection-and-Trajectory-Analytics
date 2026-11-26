import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .detector import detect_wrong_way
from .scenario import parse_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect wrong-way candidates in vehicle tracks.")
    parser.add_argument("scenario", help="Path to a scenario JSON file, or - to read from stdin")
    args = parser.parse_args()

    try:
        if args.scenario == "-":
            payload = json.load(sys.stdin.buffer)
        else:
            payload = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
        scenario = parse_scenario(payload)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    events = detect_wrong_way(scenario)
    print(json.dumps({"events": [asdict(event) for event in events]}, indent=2))


if __name__ == "__main__":
    main()
