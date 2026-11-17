"""Summarise paired elapsed times from the historical serving CSV."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def summary(values):
    ordered = sorted(values)
    return {
        "median_seconds": round(statistics.median(values), 6),
        "p95_nearest_rank_seconds": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 6),
        "min_seconds": round(ordered[0], 6),
        "max_seconds": round(ordered[-1], 6),
    }


def benchmark(path):
    pairs = {}
    names = Counter()
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = row["image_id"]
            if not key or key in pairs:
                raise ValueError(f"Missing or repeated image ID: {key!r}")
            a, b = (float(row[f"model_{m}_inference_time"]) for m in ("a", "b"))
            if not all(math.isfinite(value) and value > 0 for value in (a, b)):
                raise ValueError(f"Expected positive, finite elapsed times for image {key}")
            pairs[key] = (a, b)
            names[row["original_filename"]] += 1
    if not pairs:
        raise ValueError("The CSV contains no timing pairs.")

    a_times, b_times = zip(*pairs.values())
    return {
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "logged_requests": len(pairs),
        "distinct_original_filenames": len(names),
        "ssd_model_a": summary(a_times),
        "faster_rcnn_model_b": summary(b_times),
        "median_paired_gap_b_minus_a_seconds": round(
            statistics.median(b - a for a, b in pairs.values()), 6,
        ),
        "model_a_faster_requests": sum(a < b for a, b in pairs.values()),
        "model_b_faster_requests": sum(b < a for a, b in pairs.values()),
        "ties": sum(b == a for a, b in pairs.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path(__file__).with_name("detection_results.csv"),
    )
    args = parser.parse_args()
    print(json.dumps(benchmark(args.csv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
