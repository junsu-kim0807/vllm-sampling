#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


CANDIDATE_FILENAMES = (
    "metrics.jsonl",
    "metrics.json",
    "metris.jsonl",
    "metris.json",
)

REQUIRED_TOPLEVEL_KEYS = (
    "method",
    "draft_model",
    "intermediate_model",
    "target_model",
    "batch_size",
    "dataset",
    "num_speculative_tokens",
    "source_file",
    "record_key",
)

OPTIONAL_INTERMEDIATE_KEYS = (
    "intermediate_model",
    "intermediate_verify_model",
    "verification_model_intermediate",
    "intermediate",
)

OPTIONAL_METRIC_KEYS = (
    "wall_time_s",
    "total_output_tokens",
    "num_drafts",
    "num_draft_tokens",
    "num_accepted_tokens",
    "avg_acceptance_rate",
    "avg_acceptance_length",
    "acceptance_rate_per_pos",
    "draft_time_s",
    "verification_time_s",
    "reject_sample_time_s",
    "avg_draft_time_s",
    "avg_verification_time_s",
    "avg_reject_sample_time_s",
    "draft_forward_ms",
    "avg_draft_forward_ms",
    "target_verify_ms",
    "avg_target_verify_ms",
    "intermediate_verify_ms",
    "avg_intermediate_verify_ms",
    "expand_collapse_ms",
    "avg_expand_collapse_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate speculative decoding metrics files into a single "
            "cost_breakdown.pt artifact."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./results/spec_decode"),
        help="Root directory that contains method/bX/dataset/kY/... metrics files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results/spec_decode/cost_breakdown.pt"),
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every discovered record.",
    )
    parser.add_argument(
        "--max-print-records",
        type=int,
        default=10,
        help="Maximum number of sample records to print during verification.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on payload/path mismatches or missing required fields. "
            "Duplicate record keys always fail."
        ),
    )
    return parser.parse_args()


def candidate_metric_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name in CANDIDATE_FILENAMES:
            files.append(path)
    return sorted(files)


def read_metrics_payload(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Empty metrics file: {path}")

    if suffix == ".jsonl":
        objs = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL parse error in {path} at line {line_no}: {exc}"
                ) from exc

        if not objs:
            raise ValueError(f"No JSON objects found in JSONL file: {path}")

        if len(objs) > 1:
            return objs[-1]

        obj = objs[0]
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error in {path}: {exc}") from exc

        if isinstance(obj, list):
            if not obj:
                raise ValueError(f"Empty JSON list in metrics file: {path}")
            obj = obj[-1]

    if not isinstance(obj, dict):
        raise ValueError(f"Metrics payload must be a JSON object: {path}")

    return obj


def parse_int_component(comp: str, prefix: str) -> int:
    if not comp.startswith(prefix):
        raise ValueError(f"Expected component starting with {prefix!r}, got: {comp!r}")
    try:
        return int(comp[len(prefix):])
    except ValueError as exc:
        raise ValueError(f"Could not parse integer from component: {comp!r}") from exc


def desanitize_model_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return name.replace("__", "/")


def parse_models_from_dir(model_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if "__TO__" in model_dir:
        left, right = model_dir.split("__TO__", 1)
        return desanitize_model_name(left), None, desanitize_model_name(right)

    if model_dir.startswith("draft__") and "__target__" in model_dir:
        remainder = model_dir[len("draft__"):]
        if "__intermediate__" in remainder:
            draft_part, remainder = remainder.split("__intermediate__", 1)
            inter_part, target_part = remainder.split("__target__", 1)
            return (
                desanitize_model_name(draft_part),
                desanitize_model_name(inter_part),
                desanitize_model_name(target_part),
            )
        draft_part, target_part = remainder.split("__target__", 1)
        return desanitize_model_name(draft_part), None, desanitize_model_name(target_part)

    return None, None, None


def first_present(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def normalize_record_key(
    method: str,
    draft_model: Optional[str],
    intermediate_model: Optional[str],
    target_model: Optional[str],
    batch_size: int,
    dataset: str,
    num_spec_tokens: int,
) -> Tuple[str, str, str, str, int, str, int]:
    return (
        str(method),
        "" if draft_model is None else str(draft_model),
        "" if intermediate_model is None else str(intermediate_model),
        "" if target_model is None else str(target_model),
        int(batch_size),
        str(dataset),
        int(num_spec_tokens),
    )


def build_record(path: Path, root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) < 6:
        raise ValueError(
            f"Expected path like method/bX/dataset/kY/model_dir/file, got: {path}"
        )

    method = rel_parts[0]
    batch_size = parse_int_component(rel_parts[1], "b")
    dataset = rel_parts[2]
    num_spec_tokens = parse_int_component(rel_parts[3], "k")
    model_dir = rel_parts[4]

    path_draft, path_intermediate, path_target = parse_models_from_dir(model_dir)

    draft_model = payload.get("draft_model", path_draft)
    target_model = payload.get("target_model", path_target)
    intermediate_model = first_present(payload, OPTIONAL_INTERMEDIATE_KEYS)
    if intermediate_model is None:
        intermediate_model = path_intermediate

    record_key = normalize_record_key(
        method=method,
        draft_model=draft_model,
        intermediate_model=intermediate_model,
        target_model=target_model,
        batch_size=batch_size,
        dataset=dataset,
        num_spec_tokens=num_spec_tokens,
    )

    record: Dict[str, Any] = {
        "method": method,
        "draft_model": draft_model,
        "intermediate_model": intermediate_model,
        "target_model": target_model,
        "batch_size": batch_size,
        "dataset": dataset,
        "num_speculative_tokens": num_spec_tokens,
        "source_file": str(path),
        "model_dir": model_dir,
        "record_key": record_key,
        "raw_payload": payload,
    }

    for key, value in payload.items():
        if key not in record:
            record[key] = value

    record["path_payload_consistency"] = {
        "batch_size_match": payload.get("batch_size", batch_size) == batch_size,
        "dataset_match": payload.get("dataset", dataset) == dataset,
        "num_speculative_tokens_match": payload.get("num_speculative_tokens", num_spec_tokens)
        == num_spec_tokens,
        "draft_model_match": (
            True if path_draft is None else payload.get("draft_model", path_draft) == path_draft
        ),
        "target_model_match": (
            True if path_target is None else payload.get("target_model", path_target) == path_target
        ),
        "intermediate_model_match": (
            True
            if path_intermediate is None
            else first_present(payload, OPTIONAL_INTERMEDIATE_KEYS) == path_intermediate
        ),
    }

    return record


def validate_records(records: List[Dict[str, Any]], strict: bool) -> Dict[str, Any]:
    issues: Dict[str, Any] = {
        "num_records": len(records),
        "missing_required_fields": [],
        "duplicates": {},
        "path_payload_mismatches": [],
        "missing_optional_metric_keys": [],
    }

    key_to_paths: Dict[Tuple[str, str, str, str, int, str, int], List[str]] = {}
    for record in records:
        for req_key in REQUIRED_TOPLEVEL_KEYS:
            if req_key not in record:
                issues["missing_required_fields"].append(
                    {
                        "record_key": record.get("record_key"),
                        "source_file": record.get("source_file"),
                        "missing_key": req_key,
                    }
                )

        rec_key = record["record_key"]
        key_to_paths.setdefault(rec_key, []).append(record["source_file"])

        consistency = record.get("path_payload_consistency", {})
        mismatched_fields = [k for k, ok in consistency.items() if not ok]
        if mismatched_fields:
            issues["path_payload_mismatches"].append(
                {
                    "record_key": rec_key,
                    "source_file": record["source_file"],
                    "mismatched_fields": mismatched_fields,
                }
            )

        missing_metrics = [k for k in OPTIONAL_METRIC_KEYS if k not in record]
        if missing_metrics:
            issues["missing_optional_metric_keys"].append(
                {
                    "record_key": rec_key,
                    "source_file": record["source_file"],
                    "missing_metric_keys": missing_metrics,
                }
            )

    duplicates = {k: v for k, v in key_to_paths.items() if len(v) > 1}
    issues["duplicates"] = duplicates

    issues["summary"] = {
        "num_unique_keys": len(key_to_paths),
        "num_duplicate_keys": len(duplicates),
        "num_missing_required_fields": len(issues["missing_required_fields"]),
        "num_path_payload_mismatches": len(issues["path_payload_mismatches"]),
        "num_records_missing_some_optional_metrics": len(issues["missing_optional_metric_keys"]),
    }

    if duplicates:
        duplicate_lines = []
        for key, paths in duplicates.items():
            duplicate_lines.append(f"  key={key} appears in {len(paths)} files")
            for p in paths:
                duplicate_lines.append(f"    {p}")
        raise ValueError("Duplicate aggregation keys found:\n" + "\n".join(duplicate_lines))

    if strict:
        if issues["missing_required_fields"]:
            raise ValueError(
                f"Missing required fields detected in {len(issues['missing_required_fields'])} records."
            )
        if issues["path_payload_mismatches"]:
            raise ValueError(
                f"Path/payload mismatches detected in {len(issues['path_payload_mismatches'])} records."
            )

    return issues


def build_artifact(records: List[Dict[str, Any]], validation: Dict[str, Any]) -> Dict[str, Any]:
    by_key = {record["record_key"]: record for record in records}

    stats = {
        "num_records": len(records),
        "num_unique_keys": len(by_key),
        "methods": sorted({record["method"] for record in records}),
        "datasets": sorted({record["dataset"] for record in records}),
        "batch_sizes": sorted({record["batch_size"] for record in records}),
        "num_speculative_tokens": sorted({record["num_speculative_tokens"] for record in records}),
        "draft_models": sorted({record["draft_model"] for record in records if record["draft_model"]}),
        "intermediate_models": sorted(
            {record["intermediate_model"] for record in records if record["intermediate_model"]}
        ),
        "target_models": sorted({record["target_model"] for record in records if record["target_model"]}),
        "records_per_method": dict(Counter(record["method"] for record in records)),
        "records_per_dataset": dict(Counter(record["dataset"] for record in records)),
    }

    return {
        "type": "cost_breakdown",
        "records": records,
        "by_key": by_key,
        "validation": validation,
        "stats": stats,
    }


def verify_saved_pt(output_path: Path, max_print_records: int) -> None:
    obj = torch.load(output_path, map_location="cpu", weights_only=False)

    required_root_keys = ("type", "records", "by_key", "validation", "stats")
    for key in required_root_keys:
        if key not in obj:
            raise ValueError(f"Saved PT file is missing root key: {key}")

    records = obj["records"]
    by_key = obj["by_key"]
    validation = obj["validation"]
    stats = obj["stats"]

    print("=" * 100)
    print(f"[VERIFY] loaded: {output_path}")
    print(f"[VERIFY] artifact type: {obj['type']}")
    print(f"[VERIFY] num records: {len(records)}")
    print(f"[VERIFY] num unique keys: {len(by_key)}")
    print(f"[VERIFY] validation summary: {validation.get('summary', {})}")
    print(f"[VERIFY] stats keys: {sorted(stats.keys())}")
    print(f"[VERIFY] methods: {stats.get('methods', [])}")
    print(f"[VERIFY] datasets: {stats.get('datasets', [])}")
    print(f"[VERIFY] batch_sizes: {stats.get('batch_sizes', [])}")
    print(f"[VERIFY] num_speculative_tokens: {stats.get('num_speculative_tokens', [])}")

    print("=" * 100)
    print("[VERIFY] sample records")
    for idx, record in enumerate(records[:max_print_records], start=1):
        sample = {
            "record_key": record["record_key"],
            "source_file": record["source_file"],
            "avg_acceptance_rate": record.get("avg_acceptance_rate"),
            "avg_acceptance_length": record.get("avg_acceptance_length"),
            "draft_forward_ms": record.get("draft_forward_ms"),
            "target_verify_ms": record.get("target_verify_ms"),
            "intermediate_verify_ms": record.get("intermediate_verify_ms"),
            "expand_collapse_ms": record.get("expand_collapse_ms"),
        }
        print(f"  [{idx}] {json.dumps(sample, ensure_ascii=False)}")

    print("=" * 100)
    print("[VERIFY] key lookup demo")
    if records:
        first = records[0]
        looked_up = by_key[first["record_key"]]
        print(f"  lookup key: {first['record_key']}")
        print(f"  lookup source_file: {looked_up['source_file']}")
        print(f"  lookup avg_acceptance_rate: {looked_up.get('avg_acceptance_rate')}")
        print(f"  lookup draft_forward_ms: {looked_up.get('draft_forward_ms')}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    files = candidate_metric_files(root)
    if not files:
        raise FileNotFoundError(
            f"No candidate metrics files found under {root}. "
            f"Supported names: {CANDIDATE_FILENAMES}"
        )

    print(f"[INFO] root: {root}")
    print(f"[INFO] found {len(files)} candidate metrics files")

    records: List[Dict[str, Any]] = []
    for path in files:
        payload = read_metrics_payload(path)
        record = build_record(path=path, root=root, payload=payload)
        records.append(record)

        if args.verbose:
            print(json.dumps(
                {
                    "record_key": record["record_key"],
                    "source_file": record["source_file"],
                    "avg_acceptance_rate": record.get("avg_acceptance_rate"),
                    "avg_acceptance_length": record.get("avg_acceptance_length"),
                },
                ensure_ascii=False,
            ))

    records = sorted(records, key=lambda r: r["record_key"])
    validation = validate_records(records, strict=args.strict)
    artifact = build_artifact(records, validation)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    print(f"[INFO] saved cost_breakdown artifact to: {output}")
    verify_saved_pt(output, max_print_records=args.max_print_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
