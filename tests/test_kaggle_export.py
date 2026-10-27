import ast
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import sys
from types import SimpleNamespace

import pytest
import yaml

from cloud.build_notebook import build
from cloud.kaggle_runtime import archive_run, count_metrics, find_bundle, prepare_data, verify_bundle
from src.export_kaggle import dataset_files, export_bundle, safe_relative
from cloud.kaggle_runtime import train


def make_project(root):
    data = root / "data/pklot_yolo_v1"
    data.mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "val", "test"), 1):
        image, label = f"images/camera/{split}.jpg", f"labels/camera/{split}.txt"
        for name, content in ((image, b"image fixture"), (label, b"0 0.5 0.5 0.2 0.2\n")):
            path = data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        rows.append(dict(image=image, label=label, split=split, date=f"2026-10-{index:02}", free=1, occupied=0))
        (data / f"{split}.txt").write_text(f"./{image}\n")
    with (data / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(images=3, source="private-local-path", splits={s: {"images": 1} for s in ("train", "val", "test")})
    (data / "summary.json").write_text(json.dumps(summary))
    (data / "dataset.yaml").write_text("path: E:/original\n")
    (data / "excluded.csv").write_text("image,reasons\n")
    (data / "date_corrections.csv").write_text("image,date\n")
    (data / "unwanted.cache").write_text("not for upload")
    for name in ("models/yolo26n.pt", "cloud/kaggle_runtime.py", "docs/kaggle.md", "data/PKLot/licence",
                 "outputs/smoke_yolo26n/train/weights/best.pt", "pyproject.toml", "uv.lock", "data/my_photo/private.jpg"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    build(root / "notebooks/smart_parking_kaggle.ipynb")
    return data


@pytest.mark.parametrize("path", ["../a", "/absolute", "E:/data", "a\\b", "images/../../secret"])
def test_paths_cannot_escape(path):
    with pytest.raises(ValueError):
        safe_relative(path)


def test_package_roundtrip_preserves_input_and_excludes_private_files(tmp_path):
    project = tmp_path / "projekt ze spacją"
    data = make_project(project)
    before = {p.relative_to(project).as_posix(): p.read_bytes() for p in project.rglob("*") if p.is_file()}
    export = tmp_path / "export"
    report = export_bundle(project, export)
    assert report["zip_crc"] == "passed" and report["original_files_unchanged"]
    assert report["dataset_images"] == 3
    for name, content in before.items():
        assert (project / name).read_bytes() == content
    with zipfile.ZipFile(export / "smart_parking_dataset.zip") as archive:
        assert not any("my_photo" in p or ".cache" in p or "smoke_yolo" in p for p in archive.namelist())
        bundle = json.loads(archive.read("smart_parking/bundle.json"))
        for item in bundle["files"]:
            assert hashlib.sha256(archive.read("smart_parking/" + item["path"])).hexdigest() == item["sha256"]
        archive.extractall(tmp_path / "input")
    root = find_bundle(tmp_path / "input")
    work = tmp_path / "work"
    prepare_data(root, work)
    train_cfg = yaml.safe_load((work / "trainval.yaml").read_text())
    assert "test" not in train_cfg
    assert Path(train_cfg["path"]) == work / "data"
    assert (work / "data/train.txt").read_bytes() == (data / "train.txt").read_bytes()
    prepare_data(root, work)  # Safe, idempotent reuse of verified working copy.
    (work / "data/labels/camera/train.txt").write_text("modified")
    with pytest.raises(ValueError, match="Working data changed"):
        prepare_data(root, work)
    with pytest.raises(FileExistsError):
        export_bundle(project, export)
    (root / "data/images/camera/train.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupted bundle"):
        verify_bundle(root)


def test_manifest_date_leakage_stops_export(tmp_path):
    data = make_project(tmp_path)
    path = data / "manifest.csv"
    path.write_text(path.read_text(encoding="utf-8-sig").replace("2026-10-02", "2026-10-01"), encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Date leakage"):
        dataset_files(data)


def test_split_list_mismatch_stops_export(tmp_path):
    data = make_project(tmp_path)
    (data / "train.txt").write_text("./images/camera/test.jpg\n")
    with pytest.raises(ValueError, match="disagree"):
        dataset_files(data)


def test_count_metrics_do_not_hide_class_errors_in_total():
    result = count_metrics([dict(free=10, occupied=5, predicted_free=5, predicted_occupied=10)])
    assert result["total_mae"] == 0
    assert result["free_mae"] == result["occupied_mae"] == result["class_count_mae"] == 5


def test_results_archive_contains_only_run(tmp_path):
    run = tmp_path / "runs/experiment"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best.pt").write_text("model")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/private.txt").write_text("not results")
    path = archive_run(run, tmp_path)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["experiment/weights/best.pt"]


def test_notebook_cells_are_valid_python_and_test_is_disabled(tmp_path):
    target = tmp_path / "notebook.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            assert cell["execution_count"] is None and cell["outputs"] == []
            sources.append(cell["source"])
    assert any("RUN_FINAL_TEST = False" in s for s in sources)
    assert any("RESUME_FROM = None" in s for s in sources)
    assert any("'--no-deps'" in s for s in sources)


@pytest.mark.parametrize("scenario", ["existing_run", "wrong_dataset_resume", "completed_resume"])
def test_training_guards_preserve_existing_runs(tmp_path, monkeypatch, scenario):
    fake_torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="fixture"),
                                 cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fixture"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(__version__="fixture"))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(__version__="8.4.187",
                        YOLO=lambda _: SimpleNamespace(ckpt={"epoch": -1, "optimizer": None})))
    bundle, work = tmp_path / "bundle", tmp_path / "work"
    bundle.mkdir()
    work.mkdir()
    (bundle / "bundle.json").write_text(json.dumps(dict(bundle_id="expected", ultralytics_version="8.4.187")))
    (work / "prepared.json").write_text(json.dumps(dict(bundle_id="expected")))
    if scenario == "existing_run":
        run = work / "runs/existing"
        run.mkdir(parents=True)
        (run / "keep.txt").write_text("keep existing result")
        with pytest.raises(FileExistsError):
            train(bundle, work, run_name="existing")
        assert (run / "keep.txt").read_text() == "keep existing result"
    else:
        previous = tmp_path / "previous"
        (previous / "weights").mkdir(parents=True)
        (previous / "weights/last.pt").write_text("checkpoint")
        (previous / "provenance.json").write_text(json.dumps(dict(
            bundle_id="wrong" if scenario == "wrong_dataset_resume" else "expected")))
        with pytest.raises(ValueError, match="another dataset" if scenario == "wrong_dataset_resume" else "finalized"):
            train(bundle, work, resume_from=previous / "weights/last.pt")
        assert (previous / "weights/last.pt").read_text() == "checkpoint"
        assert not (work / "runs").exists()
