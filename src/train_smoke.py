"""Short GPU training and inference check; never evaluates the held-out test set."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import platform
import random
import time

import yaml


def sample_groups(rows: list[dict], split: str, per_group: int, seed: int) -> list[dict]:
    if per_group < 1:
        raise ValueError("per_group must be positive")
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[(row["camera"], row["weather"])].append(row)
    if not groups:
        raise ValueError(f"No examples in split {split}")
    rng = random.Random(seed)
    selected = []
    for key, group in sorted(groups.items()):
        candidates = sorted(group, key=lambda row: row["image"])
        rng.shuffle(candidates)
        selected.extend(candidates[:per_group])
    return sorted(selected, key=lambda row: row["image"])


def prepare_subset(dataset: Path, destination: Path, seed: int = 42) -> tuple[Path, list[dict], dict]:
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {"train": sample_groups(rows, "train", 60, seed),
                "val": sample_groups(rows, "val", 15, seed)}
    dates = {split: {r["date"] for r in group} for split, group in selected.items()}
    if dates["train"] & dates["val"]:
        raise ValueError("Training and validation dates overlap")
    if {r["image"] for r in selected["train"]} & {r["image"] for r in selected["val"]}:
        raise ValueError("Training and validation images overlap")
    destination.mkdir(parents=True)
    stats = {}
    for split, group in selected.items():
        (destination / f"{split}.txt").write_text(
            "".join((dataset / row["image"]).resolve().as_posix() + "\n" for row in group), encoding="utf-8")
        stats[split] = dict(images=len(group), dates=len(dates[split]),
                            groups=dict(Counter(f"{r['camera']}/{r['weather']}" for r in group)),
                            free=sum(int(r["free"]) for r in group),
                            occupied=sum(int(r["occupied"]) for r in group))
    config = dict(path=destination.resolve().as_posix(), train="train.txt", val="val.txt",
                  names={0: "FREE", 1: "OCCUPIED"})
    path = destination / "dataset.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with (destination / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected["train"] + selected["val"])
    return path, selected["val"], stats


def inference_check(weights: Path, dataset: Path, rows: list[dict], output: Path, imgsz: int):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import torch
    from ultralytics import YOLO

    groups = defaultdict(list)
    for row in rows:
        groups[(row["camera"], row["weather"])].append(row)
    samples = [group[len(group) // 2] for _, group in sorted(groups.items())]
    images = []
    for row in samples:
        with Image.open(dataset / row["image"]) as original:
            images.append(np.asarray(original.convert("RGB"))[:, :, ::-1].copy())
    output.mkdir()
    gpu_model = YOLO(str(weights))
    predictions = gpu_model.predict(images, imgsz=imgsz, device=0, conf=.25,
                                    batch=1, rect=False, verbose=False)
    counts, tiles = [], []
    for row, result in zip(samples, predictions):
        classes = result.boxes.cls.cpu().numpy().astype(int)
        free, occupied = int((classes == 0).sum()), int((classes == 1).sum())
        counts.append(dict(image=row["image"], predicted_free=free, predicted_occupied=occupied,
                           reference_free=int(row["free"]), reference_occupied=int(row["occupied"])))
        image = Image.fromarray(result.orig_img[:, :, ::-1])
        draw = ImageDraw.Draw(image)
        for cls, box in zip(classes, result.boxes.xyxy.cpu().numpy()):
            draw.rectangle(tuple(map(float, box)), outline="#00ff63" if cls == 0 else "#ff4747", width=2)
        image = ImageOps.expand(image, border=(0, 56, 0, 0), fill="#15202b")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=15)
        draw.text((10, 5), f"{row['camera']}/{row['weather']} | SMOKE MODEL | confidence >= 0.25", fill="white", font=font)
        draw.text((10, 29), f"Predicted FREE/OCCUPIED: {free}/{occupied} | XML: {row['free']}/{row['occupied']}", fill="white", font=font)
        image.save(output / f"{row['camera']}_{row['weather']}.jpg")
        tiles.append(ImageOps.pad(image, (640, 390), color="#15202b"))
    sheet = Image.new("RGB", (1920, 390 * ((len(tiles) + 2) // 3)), "#15202b")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3) * 640, (i // 3) * 390))
    sheet.save(output / "contact_sheet.jpg")
    benchmarks = {}
    # Single images, preloaded into memory; wall time includes preprocessing and postprocessing.
    for device in (0, "cpu"):
        print(f"Benchmarking single-image inference on {device}", flush=True)
        model = YOLO(str(weights))
        options = dict(imgsz=imgsz, device=device, conf=.25, rect=False, verbose=False, batch=1)
        for _ in range(3):
            model.predict(images[0], **options)
        times = []
        for _ in range(3):
            for image in images:
                if device == 0:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                model.predict(image, **options)
                if device == 0:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - started) * 1000)
        benchmarks[str(device)] = dict(samples=len(times), p50_ms=float(np.median(times)),
                                       p95_ms=float(np.percentile(times, 95)),
                                       mean_ms=float(np.mean(times)), batch=1, imgsz=imgsz,
                                       torch_threads=torch.get_num_threads(), precision="FP32",
                                       scope="preloaded BGR image -> predictions; no disk read, drawing or model load")
    return counts, benchmarks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/pklot_yolo_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke_yolo26n"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new --output; existing experiments are preserved")
    if args.epochs < 1 or args.imgsz < 32 or args.imgsz % 32 or args.batch < 1:
        raise ValueError("Positive epochs/batch and imgsz divisible by 32 required")
    args.output.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str((args.output / "ultralytics_config").resolve())
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this smoke test requires the configured NVIDIA GPU")
    torch.set_num_threads(4)
    dataset = args.dataset.resolve()
    subset_yaml, val_rows, subset_stats = prepare_subset(dataset, args.output / "subset")
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0), cpu=platform.processor(),
                       os=platform.platform(), cpu_logical_count=os.cpu_count())
    print(json.dumps(dict(environment=environment, subset=subset_stats), indent=2), flush=True)
    Path("models").mkdir(exist_ok=True)
    model = YOLO("models/yolo26n.pt")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = model.train(data=str(subset_yaml.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, device=0, workers=0, amp=False, cache=False,
                          project=str(args.output.resolve()), name="train", seed=42,
                          deterministic=True, optimizer="AdamW", lr0=.001, warmup_epochs=.5,
                          mosaic=0.0, close_mosaic=0, scale=.2, plots=True, save=True,
                          verbose=False, max_det=300)
    training_seconds = time.perf_counter() - started
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt")
    parameter_count = sum(p.numel() for p in YOLO(str(best)).model.parameters())
    counts, benchmarks = inference_check(best, dataset, val_rows, args.output / "predictions", args.imgsz)
    report = dict(purpose="Pipeline smoke test, not final model evaluation", environment=environment,
                  subset=subset_stats, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                  training_seconds=training_seconds, peak_allocated_vram_gib=peak_vram,
                  checkpoint=str(best.resolve()), checkpoint_bytes=best.stat().st_size,
                  parameters=parameter_count, validation_metrics={k: float(v) for k, v in metrics.results_dict.items()},
                  predictions=counts, inference_benchmarks=benchmarks, test_set_used=False,
                  raspberry_pi_measured=False)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
