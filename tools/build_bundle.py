"""Đóng gói source GitHub hoặc code Kaggle, bỏ ảnh DATA và file model lớn."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import argparse

ROOT = Path(__file__).resolve().parents[1]
BINARY = {".npy", ".npz", ".pkl", ".joblib", ".pt", ".pth", ".keras", ".h5", ".zip", ".pyc"}


def build_bundle(kind="github", destination=None):
    folders = (
        ["src", "docs", "tests", "tools", "DATA/metadata"]
        if kind == "github"
        else ["src/SVM", "src/CNN", "src/evaluation", "src/experiments"]
    )
    files = {ROOT / "README.md", ROOT / "requirements.txt"}
    if kind == "github":
        files.update([ROOT / ".gitignore", ROOT / ".gitattributes"])
    else:
        files.update([ROOT / "src/preprocessing.py", ROOT / "docs/REBUILD_KAGGLE.md"])
    for folder in folders:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix in BINARY:
                continue
            if any(part in {"__pycache__", ".ipynb_checkpoints"} for part in path.parts):
                continue
            if kind == "kaggle" and (
                "results_224" in path.parts or path.name.startswith("00_READ_RESULTS")
            ):
                continue
            files.add(path)
    destination = Path(destination or ROOT / "output" / f"CS231_{kind.upper()}_CODE.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.stat().st_size > 50 * 1024 * 1024:
                raise ValueError(f"File quá lớn, kiểm tra trước khi upload: {path}")
            archive.write(path, path.relative_to(ROOT))
    return destination, len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["github", "kaggle"], default="github")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path, count = build_bundle(args.kind, args.output)
    print(path, "|", count, "files |", path.stat().st_size, "bytes")
