"""Đường dẫn dùng chung cho notebook trên máy cá nhân và Kaggle."""

from pathlib import Path
import json
import os
import shutil


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def kaggle_inputs(input_root=Path("/kaggle/input")):
    return [
        p
        for pattern in ("*", "datasets/*/*", "notebooks/*/*")
        for p in Path(input_root).glob(pattern)
        if p.is_dir()
    ]


def find_project():
    """Tìm thư mục chứa src; có thể chỉ định CS231_PROJECT_ROOT nếu cần."""
    hint = os.environ.get("CS231_PROJECT_ROOT")
    candidates = [Path(hint)] if hint else [Path.cwd(), *Path.cwd().parents, *kaggle_inputs()]
    matches = sorted({p.resolve() for p in candidates if (p / "src/preprocessing.py").is_file()})
    if len(matches) != 1:
        raise FileNotFoundError("Đặt CS231_PROJECT_ROOT trỏ tới Dataset code hoặc thư mục repo.")
    return matches[0]


def setup(model, project_root):
    """Kết quả chạy lại nằm ngoài bộ kết quả đã công bố trong src/.../results_224."""
    project_root = Path(project_root)
    hint = os.environ.get("CS231_DATA_ROOT")
    candidates = (
        [Path(hint)]
        if hint
        else [project_root / "DATA"]
        + [p for mount in kaggle_inputs() for p in (mount / "DATA", mount)]
    )
    matches = sorted(
        {
            p.resolve()
            for p in candidates
            if (p / "train/10000").is_dir() and (p / "dev/10000").is_dir()
        }
    )
    if len(matches) != 1:
        raise FileNotFoundError("Đặt CS231_DATA_ROOT trỏ tới thư mục có train/dev/test.")
    default = (
        Path("/kaggle/working") / f"CS231_{model}" / "results_224"
        if Path("/kaggle/working").is_dir()
        else project_root / "runs" / model
    )
    results = Path(os.environ.get(f"CS231_{model}_RESULTS_ROOT", default)).resolve()
    published = project_root / "src" / model / "results_224"
    if results == published.resolve():
        raise ValueError("Dùng thư mục runs để giữ nguyên bộ kết quả đã công bố.")
    results.mkdir(parents=True, exist_ok=True)
    print("Code:", project_root, "\nDATA:", matches[0], "\nKết quả chạy lại:", results)
    return matches[0], results


def restore_previous_output(results_root, model, required_file, input_root=Path("/kaggle/input")):
    """Sao chép Output bước trước khi dùng nhiều notebook Kaggle riêng biệt."""
    results_root = Path(results_root)
    if (results_root / required_file).is_file():
        return
    hint = os.environ.get(f"CS231_{model}_PREVIOUS_RESULTS")
    candidates = (
        [Path(hint)]
        if hint
        else [m / f"CS231_{model}" / "results_224" for m in kaggle_inputs(input_root)]
    )
    matches = [p for p in candidates if (p / required_file).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Chạy bước trước hoặc gắn Output chứa {required_file}. "
            f"Có thể đặt CS231_{model}_PREVIOUS_RESULTS."
        )
    shutil.copytree(matches[0], results_root, dirs_exist_ok=True)
    print("Đã lấy Output từ:", matches[0])
