"""Bounded Kaggle output discovery; this module needs only the standard library."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def restore_previous_output(results_root, expected_stage, expected_artifact,
                            input_root=Path("/kaggle/input")):
    results_root, input_root = Path(results_root), Path(input_root)
    if not input_root.is_dir():
        print("Local: dùng thư mục kết quả hiện có.")
        return
    explicit = os.environ.get("CS231_CNN_PREVIOUS_RESULTS")
    if explicit:
        candidates = [Path(explicit)]
    else:
        modern = input_root / "notebooks"
        mounts = list(modern.glob("*/*")) if modern.is_dir() else list(input_root.glob("*"))
        candidates = [mount / suffix for mount in mounts
                      for suffix in ("CS231_CNN/results_224", "results_224")]
    sources = sorted({p.resolve() for p in candidates
                      if (p / "inputs/experiment_config.json").is_file()
                      and (p / "stage_completed.json").is_file()})
    if len(sources) != 1:
        raise FileNotFoundError(
            "Gắn đúng một Notebook Output CNN của bước liền trước, hoặc đặt "
            f"CS231_CNN_PREVIOUS_RESULTS. Tìm thấy: {sources}")
    source = sources[0]
    stage = json.loads((source / "stage_completed.json").read_text())
    if stage["stage"] != expected_stage or not (source / expected_artifact).is_file():
        raise ValueError(f"Cần output bước {expected_stage}; output đang gắn là {stage}.")
    local_protocol = results_root / "inputs/experiment_config.json"
    if local_protocol.is_file() and local_protocol.read_bytes() != (
            source / "inputs/experiment_config.json").read_bytes():
        raise ValueError("Thư mục kết quả chứa lần thực nghiệm khác. Dùng phiên mới hoặc root mới.")
    if source != results_root.resolve():
        shutil.copytree(source, results_root, dirs_exist_ok=True)
    print("Đã chép:", source, "→", results_root)


def mark_stage(results_root, stage):
    Path(results_root, "stage_completed.json").write_text(
        json.dumps({"model": "CNN", "stage": stage}, indent=2) + "\n", encoding="utf-8")
