#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Download LongBench (THUDM/LongBench) data for use without HuggingFace dataset scripts.

  python scripts/download_longbench.py [--out-dir DIR] [--subsets SUBSETS]

Downloads data.zip from the Hub, extracts it to --out-dir (default: ./data/longbench).
Then run_spec_decode_metrics.py can load via env: LONGBENCH_DATA_DIR=<out-dir>.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


LONGBENCH_REPO = "THUDM/LongBench"
DEFAULT_OUT_DIR = Path("data/longbench")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Extract data.zip here (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--subsets",
        type=str,
        default="gov_report,qmsum",
        help="Comma-separated subset names to ensure exist (default: gov_report,qmsum).",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Only list paths inside data.zip; do not download or extract.",
    )
    args = p.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required. Install: pip install huggingface_hub"
        )

    zip_path = hf_hub_download(
        repo_id=LONGBENCH_REPO,
        filename="data.zip",
        repo_type="dataset",
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()

    if args.list_only:
        for n in sorted(namelist):
            print(n)
        print(f"\nTotal entries: {len(namelist)}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.out_dir)

    # Zip may be data/... or LongBench/data/...; find where "data" lives
    data_dir = args.out_dir / "data"
    if not data_dir.is_dir():
        for d in args.out_dir.iterdir():
            if d.is_dir() and (d / "data").is_dir():
                data_dir = d / "data"
                break

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    for cfg in subsets:
        found = list(data_dir.rglob(f"*{cfg}*.jsonl")) or list(data_dir.rglob(f"*{cfg}*.json"))
        if found:
            print(f"  {cfg}: {found[0]}")
        else:
            print(f"  {cfg}: (no file under {data_dir})")

    # LONGBENCH_DATA_DIR = dir that contains "data" so .../data/<cfg>.jsonl works
    root_for_env = data_dir.parent
    print(f"Extracted to {args.out_dir.absolute()}")
    print(f"Usage: LONGBENCH_DATA_DIR={root_for_env.absolute()} python scripts/run_spec_decode_metrics.py ...")


if __name__ == "__main__":
    main()
