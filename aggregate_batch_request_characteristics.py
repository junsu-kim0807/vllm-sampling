#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import torch


PROFILE_FILE_RE = re.compile(
    r"^spec_decode_request_metadata\.worker_(\d+)(?:\.jsonl|jsonl|\.json)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate speculative decoding request metadata JSONL files into "
            "batch_request_characteristics.pt."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./results/spec_decode"),
        help="Root directory that contains method/bX/dataset/kY/... profile files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results/spec_decode/batch_request_characteristics.pt"),
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every discovered selected worker file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if duplicate record keys are found or malformed paths are detected.",
    )
    parser.add_argument(
        "--max-print-records",
        type=int,
        default=10,
        help="How many sample aggregated records to print during verification.",
    )
    return parser.parse_args()


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


def find_profile_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and PROFILE_FILE_RE.match(path.name):
            files.append(path)
    return sorted(files)


def get_record_dir(profile_path: Path) -> Path:
    if profile_path.parent.name == "spec_decode_profile":
        return profile_path.parent.parent
    return profile_path.parent


def parse_worker_id(path: Path) -> int:
    m = PROFILE_FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Not a recognized worker metadata filename: {path}")
    return int(m.group(1))


def choose_lowest_worker_files(root: Path) -> Dict[Path, Dict[str, Any]]:
    grouped: DefaultDict[Path, List[Path]] = defaultdict(list)
    for path in find_profile_files(root):
        grouped[get_record_dir(path)].append(path)

    chosen: Dict[Path, Dict[str, Any]] = {}
    for record_dir, paths in grouped.items():
        ranked = sorted(paths, key=lambda p: (parse_worker_id(p), str(p)))
        chosen[record_dir] = {
            "selected_file": ranked[0],
            "selected_worker_id": parse_worker_id(ranked[0]),
            "all_candidate_files": [str(p) for p in ranked],
            "all_candidate_worker_ids": [parse_worker_id(p) for p in ranked],
        }
    return chosen


def parse_record_dir(
    record_dir: Path,
    root: Path,
) -> Dict[str, Any]:
    rel_parts = record_dir.relative_to(root).parts
    if len(rel_parts) < 5:
        raise ValueError(
            f"Expected record directory like method/bX/dataset/kY/model_dir, got: {record_dir}"
        )

    method = rel_parts[0]
    batch_size = parse_int_component(rel_parts[1], "b")
    dataset = rel_parts[2]
    num_spec_tokens = parse_int_component(rel_parts[3], "k")
    model_dir = rel_parts[4]

    draft_model, intermediate_model, target_model = parse_models_from_dir(model_dir)
    record_key = normalize_record_key(
        method=method,
        draft_model=draft_model,
        intermediate_model=intermediate_model,
        target_model=target_model,
        batch_size=batch_size,
        dataset=dataset,
        num_spec_tokens=num_spec_tokens,
    )

    return {
        "method": method,
        "batch_size": batch_size,
        "dataset": dataset,
        "num_speculative_tokens": num_spec_tokens,
        "model_dir": model_dir,
        "draft_model": draft_model,
        "intermediate_model": intermediate_model,
        "target_model": target_model,
        "record_key": record_key,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL parse error in {path} at line {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path} line {line_no}, got {type(obj)}"
                )
            rows.append(obj)
    if not rows:
        raise ValueError(f"No JSON objects found in JSONL file: {path}")
    return rows


def acceptance_length(entry: Dict[str, Any]) -> int:
    return int(entry.get("num_accepted_tokens", 0))


def has_valid_topk_info(entry: Dict[str, Any], k: int) -> bool:
    topk_ids = entry.get("first_draft_topk_token_ids")
    target_top1 = entry.get("first_draft_target_top1_token_id")
    return isinstance(topk_ids, list) and len(topk_ids) >= k and target_top1 is not None


def aggregate_entries(entries: List[Dict[str, Any]], num_spec_tokens: int) -> Dict[str, Any]:
    entries = sorted(
        entries,
        key=lambda x: (
            int(x.get("step_id", -1)),
            int(x.get("req_index", 10**18)) if x.get("req_index") is not None else 10**18,
            str(x.get("req_id", "")),
        ),
    )

    step_to_entries: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    req_id_to_entries: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    req_id_to_req_index: Dict[str, int] = {}
    acceptance_counter: Counter = Counter()

    for entry in entries:
        step = int(entry.get("step_id", -1))
        req_id = str(entry.get("req_id"))
        req_index_raw = entry.get("req_index")
        req_index = int(req_index_raw) if req_index_raw is not None else 10**18

        step_to_entries[step].append(entry)
        req_id_to_entries[req_id].append(entry)
        req_id_to_req_index[req_id] = min(req_id_to_req_index.get(req_id, req_index), req_index)
        acceptance_counter[acceptance_length(entry)] += 1

    unique_steps = sorted(step_to_entries.keys())
    num_unique_steps = len(unique_steps)
    largest_step_id = max(unique_steps) if unique_steps else None

    max_observed_acceptance = max(acceptance_counter.keys()) if acceptance_counter else 0
    max_acceptance_bucket = max(num_spec_tokens, max_observed_acceptance)

    avg_count_per_step: Dict[int, float] = {}
    avg_fraction_per_step: Dict[int, float] = {}
    per_step_distribution_counts: Dict[int, Dict[int, int]] = {}

    for step in unique_steps:
        step_entries = step_to_entries[step]
        step_counts = Counter(acceptance_length(e) for e in step_entries)
        per_step_distribution_counts[step] = {
            acc_len: int(step_counts.get(acc_len, 0))
            for acc_len in range(max_acceptance_bucket + 1)
        }

    for acc_len in range(max_acceptance_bucket + 1):
        counts = []
        fractions = []
        for step in unique_steps:
            step_total = len(step_to_entries[step])
            c = per_step_distribution_counts[step].get(acc_len, 0)
            counts.append(c)
            fractions.append(c / step_total if step_total > 0 else 0.0)
        avg_count_per_step[acc_len] = float(sum(counts) / num_unique_steps) if num_unique_steps else 0.0
        avg_fraction_per_step[acc_len] = float(sum(fractions) / num_unique_steps) if num_unique_steps else 0.0

    ordered_req_ids = sorted(
        req_id_to_entries.keys(),
        key=lambda rid: (req_id_to_req_index.get(rid, 10**18), rid),
    )
    first_five_req_ids = ordered_req_ids[:5]

    first_five_request_traces: Dict[str, List[Dict[str, Any]]] = {}
    for rid in first_five_req_ids:
        trace_entries = sorted(
            req_id_to_entries[rid],
            key=lambda x: int(x.get("step_id", -1)),
        )
        first_five_request_traces[rid] = [
            {
                "step_id": int(e.get("step_id", -1)),
                "acceptance_length": acceptance_length(e),
                "num_draft_tokens": int(e.get("num_draft_tokens", 0)),
                "num_rejected_tokens": int(e.get("num_rejected_tokens", 0)),
            }
            for e in trace_entries
        ]

    num_rows = len(req_id_to_entries)
    num_total_records = len(entries)
    num_total_misspeculation = int(sum(1 for e in entries if acceptance_length(e) == 0))
    misspeculation_probability = (
        float(num_total_misspeculation / num_total_records) if num_total_records > 0 else None
    )

    topk_match_probability: Dict[str, Dict[str, Optional[float]]] = {}
    misspec_rows = [e for e in entries if acceptance_length(e) == 0]
    for k in (2, 3, 4, 5):
        valid_rows = [e for e in misspec_rows if has_valid_topk_info(e, k)]
        matched = [
            e for e in valid_rows
            if e["first_draft_target_top1_token_id"] in e["first_draft_topk_token_ids"][:k]
        ]
        topk_match_probability[f"top_{k}"] = {
            "numerator": len(matched),
            "denominator": len(valid_rows),
            "probability": float(len(matched) / len(valid_rows)) if valid_rows else None,
        }

    positive_rows = [e for e in entries if acceptance_length(e) > 0]
    oracle_acceptance_length = (
        float(sum(acceptance_length(e) for e in positive_rows) / len(positive_rows))
        if positive_rows else None
    )

    per_request_summary: Dict[str, Dict[str, Any]] = {}
    for rid in ordered_req_ids:
        req_entries = req_id_to_entries[rid]
        req_acc = [acceptance_length(e) for e in req_entries]
        per_request_summary[rid] = {
            "req_index": req_id_to_req_index.get(rid),
            "num_rounds": len(req_entries),
            "avg_acceptance_length": float(sum(req_acc) / len(req_acc)) if req_acc else None,
            "num_misspeculation": int(sum(1 for x in req_acc if x == 0)),
            "misspeculation_probability": float(sum(1 for x in req_acc if x == 0) / len(req_acc)) if req_acc else None,
        }

    return {
        "num_rows": num_rows,
        "num_total_records": num_total_records,
        "largest_step_id": largest_step_id,
        "num_unique_steps": num_unique_steps,
        "num_total_verification_round": largest_step_id,
        "intra_batch_distribution": {
            "average_count_per_step_by_acceptance_length": avg_count_per_step,
            "average_fraction_per_step_by_acceptance_length": avg_fraction_per_step,
            "per_step_distribution_counts": per_step_distribution_counts,
        },
        "first_five_request_ids": first_five_req_ids,
        "first_five_request_acceptance_traces": first_five_request_traces,
        "num_total_misspeculation": num_total_misspeculation,
        "misspeculation_probability": misspeculation_probability,
        "topk_match_probability_under_misspeculation": topk_match_probability,
        "oracle_acceptance_length_excluding_misspeculation": oracle_acceptance_length,
        "acceptance_length_histogram_global": dict(sorted(acceptance_counter.items())),
        "per_request_summary": per_request_summary,
    }


def build_record(record_dir: Path, root: Path, chosen_info: Dict[str, Any]) -> Dict[str, Any]:
    path_info = parse_record_dir(record_dir, root)
    selected_file = Path(chosen_info["selected_file"])
    selected_worker_id = int(chosen_info["selected_worker_id"])
    entries = read_jsonl(selected_file)
    aggregated = aggregate_entries(entries, num_spec_tokens=path_info["num_speculative_tokens"])

    return {
        **path_info,
        **aggregated,
        "selected_worker_file": str(selected_file),
        "selected_worker_id": selected_worker_id,
        "all_candidate_files": list(chosen_info["all_candidate_files"]),
        "all_candidate_worker_ids": list(chosen_info["all_candidate_worker_ids"]),
        "profile_writer_ids_observed": sorted(
            {str(e.get("profile_writer_id")) for e in entries if e.get("profile_writer_id") is not None}
        ),
        "raw_entry_example": entries[0],
    }


def validate_records(records: List[Dict[str, Any]], strict: bool) -> Dict[str, Any]:
    issues: Dict[str, Any] = {
        "num_records": len(records),
        "duplicates": {},
        "records_with_empty_first_five_traces": [],
        "records_without_misspeculation": [],
        "records_missing_topk_probabilities": [],
    }

    key_to_paths: Dict[Tuple[str, str, str, str, int, str, int], List[str]] = {}
    for record in records:
        rec_key = record["record_key"]
        key_to_paths.setdefault(rec_key, []).append(record["selected_worker_file"])

        if not record.get("first_five_request_acceptance_traces"):
            issues["records_with_empty_first_five_traces"].append(
                {"record_key": rec_key, "selected_worker_file": record["selected_worker_file"]}
            )

        if record.get("num_total_misspeculation", 0) == 0:
            issues["records_without_misspeculation"].append(
                {"record_key": rec_key, "selected_worker_file": record["selected_worker_file"]}
            )

        topk = record.get("topk_match_probability_under_misspeculation", {})
        missing_prob = []
        for name in ("top_2", "top_3", "top_4", "top_5"):
            if topk.get(name, {}).get("probability", None) is None:
                missing_prob.append(name)
        if missing_prob:
            issues["records_missing_topk_probabilities"].append(
                {
                    "record_key": rec_key,
                    "selected_worker_file": record["selected_worker_file"],
                    "missing_probabilities": missing_prob,
                }
            )

    duplicates = {k: v for k, v in key_to_paths.items() if len(v) > 1}
    issues["duplicates"] = duplicates
    issues["summary"] = {
        "num_unique_keys": len(key_to_paths),
        "num_duplicate_keys": len(duplicates),
        "num_records_with_empty_first_five_traces": len(issues["records_with_empty_first_five_traces"]),
        "num_records_without_misspeculation": len(issues["records_without_misspeculation"]),
        "num_records_missing_topk_probabilities": len(issues["records_missing_topk_probabilities"]),
    }

    if duplicates:
        duplicate_lines = []
        for key, paths in duplicates.items():
            duplicate_lines.append(f"  key={key} appears in {len(paths)} files")
            for p in paths:
                duplicate_lines.append(f"    {p}")
        raise ValueError("Duplicate aggregation keys found:\n" + "\n".join(duplicate_lines))

    if strict and issues["records_with_empty_first_five_traces"]:
        raise ValueError("At least one record had empty first_five_request_acceptance_traces.")

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
        "intermediate_models": sorted({record["intermediate_model"] for record in records if record["intermediate_model"]}),
        "target_models": sorted({record["target_model"] for record in records if record["target_model"]}),
        "records_per_method": dict(Counter(record["method"] for record in records)),
        "records_per_dataset": dict(Counter(record["dataset"] for record in records)),
    }
    return {
        "type": "batch_request_characteristics",
        "records": records,
        "by_key": by_key,
        "validation": validation,
        "stats": stats,
    }


def verify_saved_pt(output_path: Path, max_print_records: int) -> None:
    obj = torch.load(output_path, map_location="cpu", weights_only=False)

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
    print(f"[VERIFY] methods: {stats.get('methods', [])}")
    print(f"[VERIFY] datasets: {stats.get('datasets', [])}")
    print(f"[VERIFY] batch_sizes: {stats.get('batch_sizes', [])}")
    print(f"[VERIFY] num_speculative_tokens: {stats.get('num_speculative_tokens', [])}")

    print("=" * 100)
    print("[VERIFY] sample records")
    for idx, record in enumerate(records[:max_print_records], start=1):
        sample = {
            "record_key": record["record_key"],
            "selected_worker_id": record["selected_worker_id"],
            "selected_worker_file": record["selected_worker_file"],
            "num_rows": record["num_rows"],
            "num_total_records": record["num_total_records"],
            "largest_step_id": record["largest_step_id"],
            "num_total_misspeculation": record["num_total_misspeculation"],
            "misspeculation_probability": record["misspeculation_probability"],
            "oracle_acceptance_length_excluding_misspeculation": record["oracle_acceptance_length_excluding_misspeculation"],
            "top_2_prob": record["topk_match_probability_under_misspeculation"]["top_2"]["probability"],
            "top_5_prob": record["topk_match_probability_under_misspeculation"]["top_5"]["probability"],
            "first_five_request_ids": record["first_five_request_ids"],
        }
        print(f"  [{idx}] {json.dumps(sample, ensure_ascii=False)}")

    print("=" * 100)
    print("[VERIFY] key lookup demo")
    if records:
        first = records[0]
        looked_up = by_key[first["record_key"]]
        print(f"  lookup key: {first['record_key']}")
        print(f"  lookup selected_worker_file: {looked_up['selected_worker_file']}")
        print(f"  lookup num_rows: {looked_up['num_rows']}")
        print(f"  lookup misspeculation_probability: {looked_up['misspeculation_probability']}")
        print(f"  lookup avg count per step distribution: {looked_up['intra_batch_distribution']['average_count_per_step_by_acceptance_length']}")
        print(f"  lookup first five traces keys: {list(looked_up['first_five_request_acceptance_traces'].keys())}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    chosen_files = choose_lowest_worker_files(root)
    if not chosen_files:
        raise FileNotFoundError(f"No request metadata worker files found under {root}.")

    print(f"[INFO] root: {root}")
    print(f"[INFO] found {len(chosen_files)} record directories with worker metadata")

    records: List[Dict[str, Any]] = []
    for record_dir, chosen_info in sorted(chosen_files.items(), key=lambda x: str(x[0])):
        record = build_record(record_dir=record_dir, root=root, chosen_info=chosen_info)
        records.append(record)
        if args.verbose:
            print(json.dumps(
                {
                    "record_key": record["record_key"],
                    "selected_worker_id": record["selected_worker_id"],
                    "selected_worker_file": record["selected_worker_file"],
                    "num_total_records": record["num_total_records"],
                    "misspeculation_probability": record["misspeculation_probability"],
                },
                ensure_ascii=False,
            ))

    records = sorted(records, key=lambda r: r["record_key"])
    validation = validate_records(records, strict=args.strict)
    artifact = build_artifact(records, validation)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    print(f"[INFO] saved batch_request_characteristics artifact to: {output}")
    verify_saved_pt(output, max_print_records=args.max_print_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
