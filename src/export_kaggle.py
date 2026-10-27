"""Create an isolated, verified Kaggle upload without modifying local datasets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import yaml


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(*posix.parts)


def dataset_files(root: Path) -> tuple[list[Path], dict]:
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    files, seen_images, seen_labels, days = [], set(), set(), {}
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for row in rows:
        image, label = safe_relative(row["image"]), safe_relative(row["label"])
        if image.parts[0] != "images" or label != Path("labels") / Path(*image.parts[1:]).with_suffix(".txt"):
            raise ValueError("Image-label layout does not match YOLO")
        if image in seen_images or label in seen_labels:
            raise ValueError("Duplicate image or label in manifest")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid split")
        if days.setdefault(row["date"], row["split"]) != row["split"]:
            raise ValueError("Date leakage between splits")
        for path in (image, label):
            resolved = (root / path).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                raise ValueError(f"Missing or external file: {path}")
            files.append(path)
        seen_images.add(image)
        seen_labels.add(label)
    if len(rows) != summary["images"]:
        raise ValueError("Manifest count differs from summary")
    for split in ("train", "val", "test"):
        actual = (root / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines()
        expected = ["./" + row["image"] for row in rows if row["split"] == split]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(f"Manifest and {split}.txt disagree")
        if len(actual) != summary["splits"][split]["images"]:
            raise ValueError("Split count differs from summary")
    files += [Path(name) for name in ("train.txt", "val.txt", "test.txt", "manifest.csv",
                                     "excluded.csv", "date_corrections.csv")]
    return files, summary


def export_bundle(project: Path, output: Path) -> dict:
    project, output = project.resolve(), output.resolve()
    dataset = project / "data/pklot_yolo_v1"
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Output must be separate from the source dataset")
    if output.exists():
        raise FileExistsError("Use a new output directory; existing exports are preserved")
    files, summary = dataset_files(dataset)
    weights = project / "models/yolo26n.pt"
    runtime = project / "cloud/kaggle_runtime.py"
    notebook = project / "notebooks/smart_parking_kaggle.ipynb"
    for path in (weights, runtime, notebook, project / "docs/kaggle.md"):
        if not path.is_file():
            raise FileNotFoundError(path)
    protected = [dataset / name for name in ("dataset.yaml", "summary.json", "train.txt", "val.txt", "test.txt", "manifest.csv")]
    protected += [project / "pyproject.toml", project / "uv.lock", weights,
                  project / "outputs/smoke_yolo26n/train/weights/best.pt"]
    before = {str(p.relative_to(project)): digest(p) for p in protected if p.is_file()}
    output.mkdir(parents=True)
    archive_path = output / "smart_parking_dataset.zip"
    partial = output / "smart_parking_dataset.zip.partial"
    registry = []
    def add_file(archive, source, relative):
        hasher = hashlib.sha256()
        size = 0
        with source.open("rb") as src, archive.open("smart_parking/" + relative, "w", force_zip64=True) as dest:
            while chunk := src.read(1024 * 1024):
                hasher.update(chunk)
                dest.write(chunk)
                size += len(chunk)
        registry.append(dict(path=relative, bytes=size, sha256=hasher.hexdigest()))
    def add_text(archive, relative, text):
        data = text.encode("utf-8")
        archive.writestr("smart_parking/" + relative, data)
        registry.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for index, relative in enumerate(files, 1):
            add_file(archive, dataset / relative, "data/" + relative.as_posix())
            if index % 4000 == 0:
                print(f"Packed {index}/{len(files)} dataset files", flush=True)
        portable_summary = dict(summary, source="original PKLot full frames; see ATTRIBUTION.md")
        add_text(archive, "data/summary.json", json.dumps(portable_summary, indent=2, ensure_ascii=False))
        add_text(archive, "data/dataset.yaml", yaml.safe_dump(dict(path=".", train="train.txt", val="val.txt",
                                                                test="test.txt", names={0: "FREE", 1: "OCCUPIED"})))
        add_file(archive, weights, "weights/yolo26n.pt")
        add_file(archive, runtime, "code/kaggle_runtime.py")
        add_file(archive, project / "data/PKLot/licence", "PKLot-LICENSE.txt")
        add_text(archive, "ATTRIBUTION.md", """# Sources and modifications

PKLot by P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr.,
and A. L. Koerich: *PKLot – A robust dataset for parking lot classification*,
Expert Systems with Applications 42(11), 4937–4949, 2015.
DOI: https://doi.org/10.1016/j.eswa.2015.02.009
Dataset: https://web.inf.ufpr.br/vri/databases/parking-lot-database/
Images and original annotations: Creative Commons Attribution 4.0 International.
License: https://creativecommons.org/licenses/by/4.0/

Derived data: XML polygons converted to normalized axis-aligned YOLO boxes;
incomplete/ambiguous frames excluded; grouped date split fixed at seed 42.
See manifest.csv, excluded.csv, date_corrections.csv and summary.json.
Original image bytes have not been resized or recompressed.

Pretrained yolo26n.pt is an Ultralytics asset, separate from the PKLot license.
Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
Ultralytics licensing: https://www.ultralytics.com/license
This upload also contains project-specific Python workflow code.
The data license must not be represented as licensing the model/software.
""")
        bundle_id = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()
        bundle = dict(kind="smart-parking-kaggle", schema=1, bundle_id=bundle_id,
                      ultralytics_version="8.4.187", dataset_images=summary["images"],
                      splits={k: v["images"] for k, v in summary["splits"].items()}, files=registry)
        archive.writestr("smart_parking/bundle.json", json.dumps(bundle, indent=2))
    print("Checking ZIP CRC for every member...", flush=True)
    with zipfile.ZipFile(partial) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError(f"ZIP corruption: {broken}")
    after = {str(p.relative_to(project)): digest(p) for p in protected if p.is_file()}
    if before != after:
        raise RuntimeError("Protected source files changed during export")
    partial.rename(archive_path)
    notebook_bytes = notebook.read_bytes()
    (output / notebook.name).write_bytes(notebook_bytes)
    (output / "START_HERE.md").write_bytes((project / "docs/kaggle.md").read_bytes())
    report = dict(bundle_id=bundle_id, archive=archive_path.name, bytes=archive_path.stat().st_size,
                  archive_sha256=digest(archive_path), notebook_sha256=hashlib.sha256(notebook_bytes).hexdigest(),
                  dataset_images=summary["images"], splits=bundle["splits"], zip_crc="passed",
                  original_files_unchanged=True, protected_files=before,
                  excludes=["my_photo", "PKLotSegmented", ".venv", ".git", "smoke outputs", "credentials"])
    (output / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "protected_files"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/kaggle_export_v1"))
    args = parser.parse_args()
    export_bundle(Path.cwd(), args.output)
