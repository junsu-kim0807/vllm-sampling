# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runs ``scripts/verify_tetris_pivot.py`` smoke mode (no CUDA, no --e2e)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "verify_tetris_pivot.py"


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="verify_tetris_pivot.py missing")
def test_verify_tetris_pivot_smoke_script() -> None:
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
