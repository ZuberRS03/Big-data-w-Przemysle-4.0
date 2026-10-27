"""Portable Kaggle workflow. Imports GPU libraries only when training/evaluating."""

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import zipfile

import yaml


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_bundle(input_root):
    candidates = []
    for path in Path(input_root).rglob("bundle.json"):
        try:
            if json.loads(path.read_text())["kind"] == "smart-parking-kaggle":
                candidates.append(path.parent)
        except (ValueError, KeyError):
            continue
    if len(candidates) != 1:
        raise ValueError(f"Expected one attached Smart Parking dataset, found {len(candidates)}: {candidates}")
    return candidates[0]


def verify_bundle(root):
    root = Path(root).resolve()
    bundle = json.loads((root / "bundle.json").read_text())
    if bundle.get("kind") != "smart-parking-kaggle" or bundle.get("schema") != 1:
        raise ValueError("Unsupported bundle")
    expected_id = hashlib.sha256(json.dumps(bundle["files"], sort_keys=True).encode()).hexdigest()
    if bundle["bundle_id"] != expected_id:
        raise ValueError("Bundle manifest digest mismatch")
    seen = set()
    for index, record in enumerate(bundle["files"], 1):
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"] or ":" in record["path"]:
            raise ValueError("Unsafe bundle path")
        path = (root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(root) or record["path"] in seen:
            raise ValueError("Duplicate or external bundle path")
        seen.add(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Corrupted bundle file: {record['path']}")
        if index % 4000 == 0:
            print(f"Verified {index}/{len(bundle['files'])} files", flush=True)
    return bundle


def prepare_data(bundle_root, work_root):
    bundle_root, work_root = Path(bundle_root).resolve(), Path(work_root).resolve()
    bundle = verify_bundle(bundle_root)
    target = work_root / "data"
    ready = work_root / "prepared.json"
    work_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not ready.is_file() or json.loads(ready.read_text())["bundle_id"] != bundle["bundle_id"]:
            raise FileExistsError("Working data incomplete or from another export; use a new WORK_ROOT")
        for item in bundle["files"]:
            if item["path"].startswith("data/"):
                if sha256(work_root / item["path"]) != item["sha256"]:
                    raise ValueError("Working data changed; use a new WORK_ROOT")
    else:
        needed = sum(item["bytes"] for item in bundle["files"] if item["path"].startswith("data/"))
        if shutil.disk_usage(work_root).free < needed + 1024**3:
            raise OSError("Need dataset size plus at least 1 GiB free working space")
        shutil.copytree(bundle_root / "data", target)
        # Check the copy too; the source dataset stays read-only.
        for item in bundle["files"]:
            if item["path"].startswith("data/") and sha256(work_root / item["path"]) != item["sha256"]:
                raise ValueError(f"Copy verification failed: {item['path']}")
    rows = read_rows(target / "manifest.csv")
    seen, dates, counts = set(), {}, Counter()
    for row in rows:
        if row["image"] in seen or dates.setdefault(row["date"], row["split"]) != row["split"]:
            raise ValueError("Duplicate image or overlapping split dates")
        seen.add(row["image"])
        counts[row["split"]] += 1
    if dict(counts) != bundle["splits"] or len(rows) != bundle["dataset_images"]:
        raise ValueError("Dataset size mismatch")
    for split in ("train", "val", "test"):
        listed = (target / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines()
        if len(listed) != len(set(listed)) or set(listed) != {"./" + r["image"] for r in rows if r["split"] == split}:
            raise ValueError("Split manifest mismatch")
    # Separate configs make accidental test use during model selection harder.
    for name, with_test in (("trainval.yaml", False), ("evaluation.yaml", True)):
        cfg = dict(path=target.as_posix(), train="train.txt", val="val.txt", names={0: "FREE", 1: "OCCUPIED"})
        if with_test:
            cfg["test"] = "test.txt"
        (work_root / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(ready, dict(bundle_id=bundle["bundle_id"], images=len(rows), splits=dict(counts)))
    return bundle


def archive_run(run, work_root):
    run, work_root = Path(run), Path(work_root)
    if not run.is_dir():
        return None
    path = work_root / f"{run.name}_results.zip"
    partial = path.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for item in sorted(run.rglob("*")):
            if item.is_file():
                archive.write(item, f"{run.name}/{item.relative_to(run).as_posix()}")
    partial.replace(path)
    return path


def train(bundle_root, work_root, run_name="parking_yolo26n_v1", epochs=30, batch=8, workers=2, resume_from=None):
    import torch
    import ultralytics
    from ultralytics import YOLO
    bundle_root, work_root = Path(bundle_root).resolve(), Path(work_root).resolve()
    if not run_name or Path(run_name).name != run_name or any(c in run_name for c in "/\\:"):
        raise ValueError("run_name must be a simple directory name")
    if epochs < 1 or batch < 1 or workers < 0:
        raise ValueError("Invalid training settings")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU accelerator in Kaggle settings")
    bundle = json.loads((bundle_root / "bundle.json").read_text())
    if ultralytics.__version__ != bundle["ultralytics_version"]:
        raise RuntimeError("Install the pinned Ultralytics version before training")
    if json.loads((work_root / "prepared.json").read_text())["bundle_id"] != bundle["bundle_id"]:
        raise ValueError("Prepare this dataset before training")
    run = work_root / "runs" / run_name
    provenance = dict(bundle_id=bundle["bundle_id"], python=platform.python_version(),
                      torch=torch.__version__, torchvision=__import__("torchvision").__version__,
                      ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), classes={0: "FREE", 1: "OCCUPIED"}, test_used=False)
    data = str(work_root / "trainval.yaml")
    if resume_from:
        checkpoint = Path(resume_from).resolve()
        previous = checkpoint.parent.parent
        prior_meta = json.loads((previous / "provenance.json").read_text())
        if prior_meta["bundle_id"] != bundle["bundle_id"]:
            raise ValueError("Resume checkpoint belongs to another dataset export")
        if checkpoint.name != "last.pt":
            raise ValueError("Resume an interrupted run from last.pt, not best.pt")
        model = YOLO(str(checkpoint))
        if model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None:
            raise ValueError("This checkpoint is finalized; it cannot resume interrupted optimizer state")
        if previous != run.resolve():
            if run.exists():
                raise FileExistsError("Resume destination exists; use a new run name")
            shutil.copytree(previous, run)
        model = YOLO(str(run / "weights/last.pt"))
        options = dict(resume=True, data=data, device=0, batch=batch, workers=workers,
                       save_dir=str(run), save_period=1)
    else:
        if run.exists():
            raise FileExistsError("Choose a new RUN_NAME; existing experiment preserved")
        model = YOLO(str(bundle_root / "weights/yolo26n.pt"))
        options = dict(data=data, epochs=epochs, patience=8, imgsz=640, batch=batch, device=0,
                       workers=workers, amp=True, cache=False, seed=42, deterministic=True,
                       optimizer="AdamW", lr0=.001, warmup_epochs=3, mosaic=0.0,
                       close_mosaic=0, scale=.2, max_det=300, plots=True, save=True,
                       save_period=1, project=str(work_root / "runs"), name=run_name,
                       save_dir=str(run), verbose=False)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "provenance.json", provenance)
    (run / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    def progress(trainer):
        write_json(run / "progress.json", dict(completed_epochs=trainer.epoch + 1,
                   target_epochs=trainer.epochs, last_checkpoint="weights/last.pt",
                   note="Local session checkpoint; download/save notebook outputs to retain after VM removal"))
    model.add_callback("on_model_save", progress)
    try:
        metrics = model.train(**options)
        best = Path(model.trainer.best)
        write_json(run / "validation_metrics.json", {k: float(v) for k, v in metrics.results_dict.items()})
        write_json(run / "model_info.json", dict(best_sha256=sha256(best), bytes=best.stat().st_size,
                   parameters=sum(p.numel() for p in YOLO(str(best)).model.parameters())))
        return best
    finally:
        saved = archive_run(run, work_root)
        print(f"Results archive (if runtime survives): {saved}")


def count_metrics(records):
    if not records:
        raise ValueError("No count predictions")
    n = len(records)
    free_errors = [r["predicted_free"] - int(r["free"]) for r in records]
    occupied_errors = [r["predicted_occupied"] - int(r["occupied"]) for r in records]
    return dict(images=n, free_mae=sum(map(abs, free_errors)) / n,
                occupied_mae=sum(map(abs, occupied_errors)) / n,
                total_mae=sum(abs(a+b) for a, b in zip(free_errors, occupied_errors)) / n,
                total_bias=sum(a+b for a, b in zip(free_errors, occupied_errors)) / n,
                class_count_mae=sum(abs(a)+abs(b) for a, b in zip(free_errors, occupied_errors)) / (2*n))


def evaluate_counts(weights, work_root, run, split="val", thresholds=(.25, .4, .5, .6)):
    from ultralytics import YOLO
    work_root, run = Path(work_root), Path(run)
    if split not in {"val", "test"} or not thresholds or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("Invalid evaluation settings")
    data = work_root / "data"
    rows = [r for r in read_rows(data / "manifest.csv") if r["split"] == split]
    lookup = {str((data / r["image"]).resolve()): r for r in rows}
    records = {threshold: [] for threshold in thresholds}
    model = YOLO(str(weights))
    results = model.predict(source=str(data / f"{split}.txt"), conf=min(thresholds),
                            imgsz=640, device=0, batch=8, rect=False, stream=True, verbose=False, max_det=300)
    seen = set()
    for index, result in enumerate(results, 1):
        key = str(Path(result.path).resolve())
        if key in seen or key not in lookup:
            raise ValueError("Unexpected prediction image")
        seen.add(key)
        row = lookup[key]
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for threshold in thresholds:
            selected = classes[scores >= threshold]
            records[threshold].append(dict(image=row["image"], free=int(row["free"]), occupied=int(row["occupied"]),
                                            predicted_free=int((selected == 0).sum()),
                                            predicted_occupied=int((selected == 1).sum())))
        if index % 200 == 0:
            print(f"Count evaluation {split}: {index}/{len(rows)}", flush=True)
    if seen != set(lookup):
        raise ValueError("Missing prediction images")
    scores = {str(t): count_metrics(group) for t, group in records.items()}
    write_json(run / f"{split}_count_metrics.json", scores)
    with (run / f"{split}_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["threshold", "image", "free", "occupied", "predicted_free", "predicted_occupied"])
        writer.writeheader()
        for threshold, group in records.items():
            writer.writerows(dict(threshold=threshold, **row) for row in group)
    if split == "val":
        chosen = min(thresholds, key=lambda t: (scores[str(t)]["class_count_mae"], t))
        write_json(run / "selected_threshold.json", dict(conf=chosen, selected_on="val",
                   criterion="mean absolute count error per class", weights_sha256=sha256(weights)))
    archive_run(run, work_root)
    return scores


def final_test(weights, work_root, run):
    from ultralytics import YOLO
    work_root, run = Path(work_root), Path(run)
    if (run / "test_metrics.json").exists():
        raise FileExistsError("Final test already recorded for this experiment")
    chosen = json.loads((run / "selected_threshold.json").read_text())
    if chosen["weights_sha256"] != sha256(weights) or chosen["selected_on"] != "val":
        raise ValueError("Freeze model and select threshold on validation first")
    metrics = YOLO(str(weights)).val(data=str(work_root / "evaluation.yaml"), split="test",
              imgsz=640, batch=8, device=0, workers=2, plots=True, project=str(run), name="final_test")
    write_json(run / "test_metrics.json", {k: float(v) for k, v in metrics.results_dict.items()})
    evaluate_counts(weights, work_root, run, "test", (chosen["conf"],))
    meta = json.loads((run / "provenance.json").read_text())
    write_json(run / "provenance.json", dict(meta, test_used=True))
    archive_run(run, work_root)
