"""Sequential camera holdout experiment; preserves previous data and models."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import yaml

from src.train_full import file_hash, write_status

VARIANTS = {
    "baseline": dict(scale=.2, degrees=0.0, translate=.1, perspective=0.0),
    "augmented": dict(scale=.7, degrees=15.0, translate=.15, perspective=.0005),
}
THRESHOLDS = (.05, .1, .25, .4, .5, .6, .75)


def completed_epoch_count(run):
    # Ultralytics may increment trainer.epoch before its final validation callback.
    # Saved CSV epochs are the authoritative count, including after early stopping.
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError("No completed epochs in results.csv")
    return max(int(row["epoch"]) for row in rows)


def partition(rows, camera):
    cameras = {r["camera"] for r in rows}
    if camera not in cameras or len(cameras) < 2:
        raise ValueError("Need a held-out camera and training cameras")
    groups = dict(train=[], val=[], holdout=[])
    seen = set()
    for row in rows:
        if row["image"] in seen:
            raise ValueError("Duplicate image in manifest")
        seen.add(row["image"])
        if row["split"] == "train" and row["camera"] != camera:
            groups["train"].append(row)
        elif row["split"] == "val":
            groups["holdout" if row["camera"] == camera else "val"].append(row)
    if any(not g for g in groups.values()):
        raise ValueError("Empty experiment partition")
    if {r["date"] for r in groups["train"]} & {r["date"] for role in ("val", "holdout") for r in groups[role]}:
        raise ValueError("Training/validation dates overlap")
    return groups


def safe_source(dataset, relative, prefix):
    path = (dataset / relative).resolve()
    if not path.is_relative_to((dataset / prefix).resolve()):
        raise ValueError("Manifest path outside expected data directory")
    return path


def prepare(dataset, output, camera, epochs, batch, workers):
    dataset, output = dataset.resolve(), output.resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Output must be separate from source dataset")
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        groups = partition(list(csv.DictReader(stream)), camera)
    base = Path("models/yolo26n.pt").resolve()
    protected = [base, dataset / "manifest.csv", Path("outputs/full_yolo26n_v1/weights/best.pt").resolve(),
                 Path("outputs/smoke_yolo26n/train/weights/best.pt").resolve()]
    protected_hashes = {str(p): file_hash(p) for p in protected if p.is_file()}
    if not base.is_file():
        raise FileNotFoundError(base)
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()
    write_status(output / "progress.json", dict(state="preparing", pid=os.getpid(), test_set_used=False))
    copied = []
    # Independent copies also isolate Ultralytics label caches and JPEG checks.
    for role, rows in groups.items():
        listed = []
        for row in rows:
            local = dict(row, role=role)
            for field, prefix in (("image", "images"), ("label", "labels")):
                source = safe_source(dataset, row[field], prefix)
                relative = Path(role) / prefix / source.relative_to(dataset / prefix)
                target = data / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                before = file_hash(source)
                shutil.copyfile(source, target)
                if file_hash(target) != before:
                    raise ValueError("Dataset copy verification failed")
                local[field] = relative.as_posix()
                local[field + "_sha256"] = before
            copied.append(local)
            listed.append((data / local["image"]).as_posix())
        (data / (role + ".txt")).write_text("\n".join(listed)+"\n", encoding="utf-8-sig")
        print(f"Prepared {role}: {len(rows)} images", flush=True)
    with (data / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(copied[0]))
        writer.writeheader()
        writer.writerows(copied)
    for name, val in (("trainval", "val.txt"), ("holdout", "holdout.txt")):
        cfg = dict(path=data.as_posix(), train="train.txt", val=val, names={0:"FREE", 1:"OCCUPIED"})
        (output / f"{name}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    experiment = dict(heldout_camera=camera, epochs=epochs, patience=8, batch=batch, workers=workers,
                      seed=42, imgsz=640, base_weights=str(base), protected_hashes=protected_hashes,
                      source_dataset=str(dataset), source_manifest_sha256=file_hash(dataset / "manifest.csv"),
                      prepared_manifest_sha256=file_hash(data / "manifest.csv"), variants=VARIANTS,
                      partitions={role:dict(images=len(rows), cameras=dict(Counter(r["camera"] for r in rows)),
                                            dates=len({r["date"] for r in rows})) for role,rows in groups.items()},
                      test_set_used=False, labels_corrected=False,
                      checkpoint_selection="val from training cameras only; heldout camera evaluated after training",
                      limitation="One camera holdout and one seed; development experiment, not final test")
    write_status(output / "experiment.json", experiment)
    shutil.copyfile(Path(__file__), output / "runner_snapshot.py")
    return experiment


def verify(output):
    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    for path, expected in experiment["protected_hashes"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f"Protected source changed: {path}")
    if file_hash(output / "data/manifest.csv") != experiment["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    with (output / "data/manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["split"] == "test" or (row["role"] == "train" and row["camera"] == experiment["heldout_camera"]):
            raise ValueError("Data leakage in prepared manifest")
        for field in ("image", "label"):
            if file_hash(output / "data" / row[field]) != row[field+"_sha256"]:
                raise ValueError("Prepared data changed")
    return experiment


def training_options(experiment, output, variant):
    return dict(data=str(output / "trainval.yaml"), epochs=experiment["epochs"], patience=experiment["patience"],
                imgsz=640, batch=experiment["batch"], device=0, workers=experiment["workers"],
                amp=True, cache=False, seed=42, deterministic=True, optimizer="AdamW", lr0=.001,
                warmup_epochs=3, mosaic=0.0, close_mosaic=0, max_det=300, plots=True,
                save=True, save_period=1, save_dir=str(output / variant), verbose=False,
                **experiment["variants"][variant])


def evaluate(model, output, run):
    import numpy as np
    from cloud.kaggle_runtime import count_metrics
    with (output / "data/manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = {}
    for role in ("val", "holdout"):
        thresholds = THRESHOLDS if role == "val" else tuple(sorted({.25, summary["selected_conf"]}))
        lookup = {str((output/"data"/r["image"]).resolve()):r for r in rows if r["role"]==role}
        records = {t:[] for t in thresholds}
        seen = set()
        for result in model.predict(source=str(output/"data"/(role+".txt")), imgsz=640, device=0,
                                    batch=4, rect=False, quantize=32, conf=min(thresholds), max_det=300,
                                    stream=True, verbose=False, save=False):
            key = str(Path(result.path).resolve())
            if key in seen or key not in lookup:
                raise ValueError("Unexpected evaluation prediction")
            seen.add(key)
            row = lookup[key]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidence = result.boxes.conf.cpu().numpy()
            for t in thresholds:
                selected = classes[confidence >= t]
                records[t].append(dict(image=row["image"], free=int(row["free"]), occupied=int(row["occupied"]),
                                       predicted_free=int(np.sum(selected==0)), predicted_occupied=int(np.sum(selected==1))))
        if seen != set(lookup):
            raise ValueError("Missing evaluation predictions")
        scores = {str(t):count_metrics(r) for t,r in records.items()}
        summary[role] = scores
        with (run / (role+"_counts.csv")).open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["threshold",*records[thresholds[0]][0].keys()])
            writer.writeheader()
            for t, group in records.items():
                writer.writerows(dict(threshold=t,**row) for row in group)
        if role == "val":
            summary["selected_conf"] = min(thresholds,key=lambda t:(scores[str(t)]["class_count_mae"],t))
        write_status(run / "count_metrics.json", summary)
    metrics = model.val(data=str(output/"holdout.yaml"), split="val", imgsz=640, device=0,
                        batch=4, workers=0, quantize=32, rect=False, conf=.001, max_det=300,
                        plots=True, save_dir=str(run/"holdout_validation"), verbose=False)
    summary["holdout_detection_metrics"] = {k:float(v) for k,v in metrics.results_dict.items()}
    summary["test_set_used"] = False
    write_status(run / "evaluation.json", summary)
    return summary


def worker(output, variant, resume):
    run = output / variant
    if run.exists() and not resume:
        raise FileExistsError(run)
    run.mkdir(exist_ok=resume)
    status = dict(state="initializing", variant=variant, pid=os.getpid(), completed_epochs=0, test_set_used=False)
    start = time.perf_counter()
    def progress(**items):
        status.update(items,elapsed_seconds=time.perf_counter()-start)
        write_status(run/"progress.json",status)
    try:
        progress(state="verifying_data")
        experiment = verify(output)
        os.environ["YOLO_CONFIG_DIR"] = str(run/"ultralytics_config")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["WANDB_MODE"] = "disabled"
        import torch
        import ultralytics
        from ultralytics import YOLO, settings
        from src.diagnose_models import configure_nms_budget
        configure_nms_budget()
        settings.update({k:False for k in ("wandb","clearml","comet","mlflow","neptune","dvc") if k in settings})
        settings.update({"weights_dir":str(Path("models").resolve())})
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(4)
        progress(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,ultralytics=ultralytics.__version__)
        if not (run/"training_complete.json").exists():
            options = training_options(experiment, output, variant)
            checkpoint = run/"weights/last.pt"
            if resume:
                if not checkpoint.exists():
                    raise RuntimeError("No resumable checkpoint; use a new experiment output")
                model = YOLO(str(checkpoint))
                if model.ckpt.get("epoch",-1)<0 or model.ckpt.get("optimizer") is None:
                    raise RuntimeError("Checkpoint finalized; inspect training logs before recovery")
                options = dict(resume=True, data=str(output/"trainval.yaml"), save_dir=str(run), device=0,
                               batch=experiment["batch"], workers=experiment["workers"], save_period=1)
            else:
                model = YOLO(experiment["base_weights"])
            def epoch_start(trainer):
                progress(state="training",current_epoch=trainer.epoch+1,target_epochs=trainer.epochs)
            def epoch_end(trainer):
                progress(state="training",completed_epochs=status["current_epoch"],epoch_seconds=float(trainer.epoch_time),
                         metrics={k:float(v) for k,v in trainer.metrics.items()},
                         best_checkpoint=str(trainer.best),last_checkpoint=str(trainer.last))
            model.add_callback("on_train_epoch_start",epoch_start)
            model.add_callback("on_fit_epoch_end",epoch_end)
            model.train(**options)
            write_status(run/"training_complete.json",dict(best_checkpoint=str(model.trainer.best),
                         best_sha256=file_hash(Path(model.trainer.best)),completed_epochs=completed_epoch_count(run)))
            del model
            torch.cuda.empty_cache()
        progress(state="evaluating",completed_epochs=completed_epoch_count(run))
        best = run/"weights/best.pt"
        summary = evaluate(YOLO(str(best)),output,run)
        verify(output)
        progress(state="completed",best_checkpoint=str(best),best_sha256=file_hash(best),
                 selected_conf=summary["selected_conf"])
    except BaseException as error:
        progress(state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                 error=str(error),traceback=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("data/pklot_yolo_v1"))
    parser.add_argument("--output",type=Path,default=Path("outputs/camera_experiment_v1"))
    parser.add_argument("--camera",default="PUCPR")
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--continue-run",action="store_true")
    parser.add_argument("--worker",choices=tuple(VARIANTS))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if args.worker:
        worker(output,args.worker,args.resume)
        return
    if args.epochs<1 or args.batch<1 or args.workers<0:
        raise ValueError("Invalid training settings")
    if args.continue_run:
        verify(output)
    else:
        prepare(args.dataset,output,args.camera,args.epochs,args.batch,args.workers)
    if args.prepare_only:
        write_status(output/"progress.json",dict(state="prepared",test_set_used=False))
        return
    queue=dict(state="running",pid=os.getpid(),completed=[],order=list(VARIANTS),test_set_used=False)
    child=None
    try:
        for variant in VARIANTS:
            run=output/variant
            prior=json.loads((run/"progress.json").read_text()) if (run/"progress.json").exists() else {}
            if prior.get("state")=="completed":
                queue["completed"].append(variant)
                continue
            queue.update(active_variant=variant)
            write_status(output/"progress.json",queue)
            command=[sys.executable,"-u","-m","src.train_camera_experiment","--output",str(output),"--worker",variant]
            if run.exists():
                command.append("--resume")
            with (output/(variant+".stdout.log")).open("a",encoding="utf-8") as stdout, (output/(variant+".stderr.log")).open("a",encoding="utf-8") as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONUTF8="1",PYTHONUNBUFFERED="1"))
                queue.update(worker_pid=child.pid)
                write_status(output/"progress.json",queue)
                code=child.wait()
            if code:
                raise RuntimeError(f"{variant} failed with exit code {code}; see its logs")
            queue["completed"].append(variant)
        comparison={variant:json.loads((output/variant/"evaluation.json").read_text()) for variant in VARIANTS}
        write_status(output/"comparison.json",dict(results=comparison,test_set_used=False))
        queue.update(state="completed",active_variant=None,worker_pid=None)
        write_status(output/"progress.json",queue)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        queue.update(state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        write_status(output/"progress.json",queue)
        raise


if __name__=="__main__":
    main()
