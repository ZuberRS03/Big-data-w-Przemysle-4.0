"""Full local YOLO training with persistent progress and resumable checkpoints."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import yaml


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_status(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_run(dataset, output, resume=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Training output must be separate from dataset")
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    fingerprint = file_hash(dataset / "manifest.csv")
    if resume:
        previous = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        if previous["manifest_sha256"] != fingerprint:
            raise ValueError("Dataset manifest changed since the interrupted training")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_status(output / "experiment.json", dict(dataset=str(dataset), manifest_sha256=fingerprint,
                     splits={key: value["images"] for key, value in summary["splits"].items()},
                     test_set_used=False, initialized_from="models/yolo26n.pt"))
    config = dict(path=dataset.as_posix(), train="train.txt", val="val.txt", names={0: "FREE", 1: "OCCUPIED"})
    config_path = output / "trainval.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path, summary


def train(args):
    output = args.output.resolve()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        raise ValueError("Invalid epochs, batch or workers")
    if args.resume and args.resume.resolve().parent.parent != output:
        raise ValueError("--resume checkpoint must belong to the specified --output run")
    config_path, summary = prepare_run(args.dataset, output, resume=bool(args.resume))
    started = time.perf_counter()
    status = dict(state="initializing", pid=os.getpid(), requested_epochs=args.epochs,
                  completed_epochs=0, output=str(output), test_set_used=False,
                  training_images=summary["splits"]["train"]["images"],
                  validation_images=summary["splits"]["val"]["images"])
    status_path = output / "progress.json"
    write_status(status_path, status)
    try:
        os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics_config")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["WANDB_MODE"] = "disabled"
        import torch
        import ultralytics
        from ultralytics import YOLO, settings
        settings.update({key: False for key in ("wandb", "clearml", "comet", "mlflow", "neptune", "dvc") if key in settings})
        settings.update({"weights_dir": str(Path("models").resolve())})
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; not falling back to CPU for full training")
        torch.set_num_threads(4)
        status.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                      ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                      python=platform.python_version(), batch=args.batch, workers=args.workers)
        epoch_times = []
        def on_epoch_start(trainer):
            status.update(state="training", current_epoch=trainer.epoch + 1, target_epochs=trainer.epochs)
            write_status(status_path, status)
        def on_epoch_end(trainer):
            completed = min(trainer.epoch + 1, trainer.epochs)
            if completed > status["completed_epochs"]:
                epoch_times.append(float(trainer.epoch_time))
            average = sum(epoch_times[-3:]) / max(1, len(epoch_times[-3:]))
            metrics = {key: float(value) if math.isfinite(float(value)) else None
                       for key, value in trainer.metrics.items()}
            status.update(state="training", completed_epochs=completed, target_epochs=trainer.epochs,
                          elapsed_seconds=time.perf_counter() - started, recent_epoch_seconds=average,
                          estimated_remaining_seconds=max(0, trainer.epochs - completed) * average,
                          metrics=metrics, last_checkpoint=str(trainer.last), best_checkpoint=str(trainer.best))
            write_status(status_path, status)
        model = YOLO(str(args.resume.resolve()) if args.resume else str(Path("models/yolo26n.pt").resolve()))
        if args.resume and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
            raise ValueError("Checkpoint is finalized; only interrupted training can resume")
        model.add_callback("on_train_epoch_start", on_epoch_start)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        if args.resume:
            options = dict(resume=True, data=str(config_path), device=0, batch=args.batch,
                           workers=args.workers, save_dir=str(output), save_period=1)
        else:
            options = dict(data=str(config_path), epochs=args.epochs, patience=8, imgsz=640,
                           batch=args.batch, device=0, workers=args.workers, amp=True, cache=False,
                           seed=42, deterministic=True, optimizer="AdamW", lr0=.001, warmup_epochs=3,
                           mosaic=0.0, close_mosaic=0, scale=.2, max_det=300, plots=True,
                           save=True, save_period=1, save_dir=str(output), verbose=False)
        metrics = model.train(**options)
        best = Path(model.trainer.best)
        write_status(output / "validation_metrics.json", {key: float(value) for key, value in metrics.results_dict.items()})
        status.update(state="completed", elapsed_seconds=time.perf_counter() - started,
                      completed_epochs=model.trainer.epoch + 1, estimated_remaining_seconds=0,
                      best_checkpoint=str(best), best_checkpoint_bytes=best.stat().st_size,
                      best_checkpoint_sha256=file_hash(best))
        write_status(status_path, status)
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      elapsed_seconds=time.perf_counter() - started, error=str(error),
                      traceback=traceback.format_exc())
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/pklot_yolo_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_yolo26n_v1"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    train(parser.parse_args())
