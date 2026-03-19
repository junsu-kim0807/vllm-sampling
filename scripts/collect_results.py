#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("/scratch/jhwoo36/vllm-sampling/results/spec_decode")
DEFAULT_OUTPUT_CSV = Path("/scratch/jhwoo36/vllm-sampling/results/spec_decode/spec_decode_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect speculative decoding JSONL results into one CSV."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory that contains spec decode results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pattern",
        default="aggregate.jsonl",
        help="Filename pattern to search for. Default: aggregate.jsonl",
    )
    return parser.parse_args()


def format_acceptance_rate_per_pos(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, list):
        return str(value)
    return ",".join(f"{float(x):.6f}" for x in value)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON in {path} line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                continue
            records.append(obj)
    return records


def main() -> None:
    args = parse_args()

    if not args.results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")

    jsonl_paths = sorted(args.results_root.rglob(args.pattern))
    if not jsonl_paths:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' under {args.results_root}"
        )

    rows: list[dict[str, Any]] = []

    for jsonl_path in jsonl_paths:
        records = load_jsonl_records(jsonl_path)
        for rec in records:
            row = {
                "dataset": rec.get("dataset", ""),
                "draft_model": rec.get("draft_model", ""),
                "target_model": rec.get("target_model", ""),
                "avg_acceptance_length": rec.get("avg_acceptance_length", ""),
                "avg_acceptance_rate": rec.get("avg_acceptance_rate", ""),
                "acceptance_rate_per_pos": format_acceptance_rate_per_pos(
                    rec.get("acceptance_rate_per_pos")
                ),
            }
            rows.append(row)

    rows.sort(
        key=lambda x: (
            str(x["dataset"]),
            str(x["draft_model"]),
            str(x["target_model"]),
        )
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "draft_model",
        "target_model",
        "avg_acceptance_length",
        "avg_acceptance_rate",
        "acceptance_rate_per_pos",
    ]

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(jsonl_paths)} JSONL files")
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()