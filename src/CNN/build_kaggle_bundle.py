"""Package executable CNN code, reference evidence and the report builder for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]


def build_bundle(destination, root=ROOT):
    root, destination = Path(root), Path(destination)
    files = set()
    for path in (root / "src/CNN").rglob("*"):
        relative = path.relative_to(root / "src/CNN")
        if (not path.is_file() or "__pycache__" in relative.parts
                or relative.parts[0] == "results_224"):
            continue
        if path.suffix in {".py", ".ipynb", ".md", ".txt", ".json", ".csv"}:
            files.add(path)
    files.update((root / "src/evaluation").glob("*.py"))
    files.add(root / "src/preprocessing.py")
    files.add(root / "src/CNN/results_224/README.md")
    for path in (root / "docs/CNN_REPORT").rglob("*"):
        if (path.is_file() and "__pycache__" not in path.parts
                and path.suffix in {".py", ".md", ".json", ".svg", ".png", ".csv"}):
            files.add(path)
    manifest = {"purpose": "CNN Kaggle code and baseline evidence; no DATA images or CNN runtime cache/weights", "sha256": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            content = path.read_bytes()
            name = path.relative_to(root).as_posix()
            archive.writestr(name, content)
            manifest["sha256"][name] = hashlib.sha256(content).hexdigest()
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    return destination, len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src/CNN/CS231_CNN_CODE.zip")
    args = parser.parse_args()
    destination, count = build_bundle(args.output)
    print(f"{destination}: {count} files; {destination.stat().st_size:,} bytes")
