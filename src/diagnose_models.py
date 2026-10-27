"""Read-only model diagnostics on validation and unlabelled personal photos."""

import argparse
import csv
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

THRESHOLDS = (.01, .05, .1, .25, .4, .5, .6, .75)


def configure_nms_budget():
    # Change only this diagnostic process, not installed files or the model.
    # A cold/loaded GPU can exceed the default 2s batch budget and silently
    # return empty detections for unprocessed images in that batch.
    from ultralytics.utils import nms
    nms.non_max_suppression = partial(nms.non_max_suppression, max_time_img=10.0)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validation_rows(dataset, subset_file):
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "val"]
    lookup = {str((dataset / r["image"]).resolve()): r for r in rows}
    subset = [str(Path(p).resolve()) for p in subset_file.read_text(encoding="utf-8-sig").splitlines() if p]
    if len(set(subset)) != len(subset) or not subset or not set(subset) <= lookup.keys():
        raise ValueError("Subset must contain unique validation images only")
    if len(lookup) != len(rows):
        raise ValueError("Duplicate validation image")
    return rows, [lookup[p] for p in subset]


def read_labels(dataset, row):
    labels = []
    for line in (dataset / row["label"]).read_text().splitlines():
        cls, x, y, w, h = map(float, line.split())
        width, height = int(row["width"]), int(row["height"])
        labels.append([int(cls), (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    if sum(x[0] == 0 for x in labels) != int(row["free"]) or sum(x[0] == 1 for x in labels) != int(row["occupied"]):
        raise ValueError("Labels disagree with manifest counts")
    return np.asarray(labels, dtype=float).reshape(-1, 5)


def overlaps(a, b):
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi-lo, 0).prod(axis=2)
    areas_a = np.maximum(a[:, 2:]-a[:, :2], 0).prod(axis=1)
    areas_b = np.maximum(b[:, 2:]-b[:, :2], 0).prod(axis=1)
    return inter / np.maximum(areas_a[:, None] + areas_b[None, :] - inter, 1e-9)


def box_errors(predictions, labels, threshold):
    """Confidence-ordered, class-agnostic 1:1 IoU>=.5 matching; not mAP."""
    pred = np.asarray(predictions, dtype=float).reshape(-1, 6)
    pred = pred[pred[:, 1] >= threshold]
    pred = pred[np.argsort(-pred[:, 1], kind="stable")]
    ious = overlaps(pred[:, 2:], labels[:, 1:])
    used, correct, wrong, duplicate, unlocalized = set(), 0, 0, 0, 0
    for p, row in zip(pred, ious):
        eligible = [int(i) for i in np.argsort(-row) if row[i] >= .5 and int(i) not in used]
        if eligible:
            match = eligible[0]
            used.add(match)
            if int(p[0]) == int(labels[match, 0]):
                correct += 1
            else:
                wrong += 1
        elif len(row) and row.max() >= .5:
            duplicate += 1
        else:
            unlocalized += 1
    return dict(correct=correct, wrong_class=wrong, duplicate=duplicate,
                unmatched_prediction=unlocalized, missed=len(labels)-len(used),
                predicted=len(pred), reference=len(labels))


def aggregate(records):
    n = len(records)
    ef = np.array([r["predicted_free"]-r["free"] for r in records])
    eo = np.array([r["predicted_occupied"]-r["occupied"] for r in records])
    totals = {key: sum(r[key] for r in records) for key in
              ("correct", "wrong_class", "duplicate", "unmatched_prediction", "missed", "predicted", "reference")}
    return dict(images=n, free_mae=float(abs(ef).mean()), occupied_mae=float(abs(eo).mean()),
                class_count_mae=float((abs(ef)+abs(eo)).mean()/2), total_mae=float(abs(ef+eo).mean()),
                free_bias=float(ef.mean()), occupied_bias=float(eo.mean()),
                exact_both_fraction=float(((ef == 0) & (eo == 0)).mean()),
                diagnostic_precision=totals["correct"]/max(1, totals["predicted"]),
                diagnostic_recall=totals["correct"]/max(1, totals["reference"]), **totals)


def evaluate(rows, predictions, labels, output, name):
    all_records, scores, groups = [], {}, {}
    for threshold in THRESHOLDS:
        records = []
        for row in rows:
            pred = np.asarray(predictions[row["image"]], dtype=float).reshape(-1, 6)
            classes = pred[pred[:, 1] >= threshold, 0]
            records.append(dict(image=row["image"], camera=row["camera"], weather=row["weather"],
                                free=int(row["free"]), occupied=int(row["occupied"]), threshold=threshold,
                                predicted_free=int((classes == 0).sum()), predicted_occupied=int((classes == 1).sum()),
                                **box_errors(pred, labels[row["image"]], threshold)))
        scores[str(threshold)] = aggregate(records)
        groups[str(threshold)] = {key: aggregate([r for r in records if r["camera"]+"/"+r["weather"] == key])
                                 for key in sorted({r["camera"]+"/"+r["weather"] for r in records})}
        all_records.extend(records)
    with (output / (name+"_counts.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    chosen = min(scores, key=lambda t: (scores[t]["class_count_mae"], float(t)))
    result = dict(scores=scores, by_camera_weather=groups, lowest_count_mae_threshold=float(chosen))
    write_json(output / (name+"_metrics.json"), result)
    return result, all_records


def panel(path, boxes, title, size=(640, 410)):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    with Image.open(path) as source:
        img = source.convert("RGB")
    draw = ImageDraw.Draw(img)
    for cls, *xyxy in boxes:
        draw.rectangle(tuple(xyxy), outline="#00ed73" if cls == 0 else "#ff4545", width=2)
    img = ImageOps.contain(img, (size[0], size[1]-48))
    tile = Image.new("RGB", size, "#15202b")
    tile.paste(img, ((size[0]-img.width)//2, 48+(size[1]-48-img.height)//2))
    ImageDraw.Draw(tile).text((8, 5), title, fill="white", font=ImageFont.load_default(size=15))
    return tile


def save_sheet(tiles, path, columns=2):
    from PIL import Image
    sheet = Image.new("RGB", (640*columns, 410*((len(tiles)+columns-1)//columns)), "#15202b")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns)*640, (i//columns)*410))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/pklot_yolo_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics_v1"))
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Output must be separate from dataset")
    rows, subset = validation_rows(dataset, Path("outputs/smoke_yolo26n/subset/val.txt"))
    output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    started = time.perf_counter()
    status = dict(state="initializing", test_set_used=False)
    def progress(**values):
        status.update(values, elapsed_seconds=time.perf_counter()-started)
        write_json(output / "progress.json", status)
        print(json.dumps(status), flush=True)
    progress()
    import torch
    import ultralytics
    from ultralytics import YOLO
    configure_nms_budget()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    weights = dict(smoke=Path("outputs/smoke_yolo26n/train/weights/best.pt").resolve(),
                   full=Path("outputs/full_yolo26n_v1/weights/best.pt").resolve())
    hashes = {key: digest(path) for key, path in weights.items()}
    manifest_hash = digest(dataset / "manifest.csv")
    labels = {r["image"]: read_labels(dataset, r) for r in rows}
    report = dict(test_set_used=False, weights={k:dict(path=str(v), sha256=hashes[k]) for k,v in weights.items()},
                  manifest_sha256=manifest_hash, validation_images=len(rows), resolution_subset_images=len(subset),
                  settings=dict(thresholds=THRESHOLDS, device=0, rect=False, quantize=32, max_det=300, batch=4, nms_max_time_img=10.0),
                  environment=dict(torch=torch.__version__, ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0)),
                  matching="Diagnostic only, not mAP: confidence-ordered class-agnostic greedy 1:1 matching at IoU >= 0.5",
                  limitations=["PKLot labels may cover only part of visible parking. Unlabelled visible slots count as unmatched predictions.",
                               "Personal photos have no ground truth: detection counts are not accuracy measurements.",
                               "Same cameras occur in training and validation; this does not measure generalization to unseen cameras.",
                               "Timing is observed during diagnostics, not a controlled device benchmark."],
                  evaluations={}, own_photos=[])
    geometries = np.concatenate([labels[r["image"]][:, 3:5]-labels[r["image"]][:, 1:3] for r in rows])
    report["reference_box_size_original_pixels"] = dict(width_percentiles=np.percentile(geometries[:,0], [10,50,90]).tolist(),
                                                          height_percentiles=np.percentile(geometries[:,1], [10,50,90]).tolist())
    own = sorted(p for p in Path("data/my_photo").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    own_panels = {p.name: [] for p in own}
    threshold_panels = {p.name: [] for p in own}
    for model_name, weight in weights.items():
        model = YOLO(str(weight))
        for size in (640, 960, 1280):
            group = rows if size == 640 else subset
            name = f"{model_name}_{size}"
            source = output / (name+"_images.txt")
            # Ultralytics uses the Windows locale unless a UTF-8 BOM is present.
            source.write_text("".join((dataset/r["image"]).as_posix()+"\n" for r in group), encoding="utf-8-sig")
            lookup = {str((dataset/r["image"]).resolve()): r for r in group}
            predictions, speeds = {}, []
            progress(state="predicting", configuration=name, completed_images=0, images=len(group))
            with (output / (name+"_predictions.jsonl")).open("w", encoding="utf-8") as stream:
                for i, result in enumerate(model.predict(source=str(source), imgsz=size, conf=min(THRESHOLDS),
                           device=0, batch=4, rect=False, quantize=32, stream=True, verbose=False, max_det=300, save=False), 1):
                    row = lookup[str(Path(result.path).resolve())]
                    if row["image"] in predictions:
                        raise ValueError("Duplicate model result")
                    pred = np.column_stack([result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()]).tolist()
                    predictions[row["image"]] = pred
                    speeds.append(result.speed)
                    stream.write(json.dumps(dict(image=row["image"], predictions=pred))+"\n")
                    if i % 200 == 0:
                        progress(completed_images=i)
            if set(predictions) != {r["image"] for r in group}:
                raise ValueError("Missing predictions")
            progress(state="scoring", completed_images=len(group))
            metrics, records = evaluate(group, predictions, labels, output, name)
            metrics["max_det_hits"] = sum(len(p) >= 300 for p in predictions.values())
            metrics["observed_mean_ms_per_image"] = {k:float(np.mean([s[k] for s in speeds])) for k in speeds[0]}
            report["evaluations"][name] = metrics
            if size == 640:
                subset_metrics, _ = evaluate(subset, predictions, labels, output, name+"_subset")
                report["evaluations"][name+"_subset"] = subset_metrics
                worst = sorted([r for r in records if r["threshold"] == .25],
                               key=lambda r: abs(r["predicted_free"]-r["free"])+abs(r["predicted_occupied"]-r["occupied"]), reverse=True)
                chosen = []
                for r in worst:
                    if r["camera"] not in {x["camera"] for x in chosen}:
                        chosen.append(r)
                tiles = []
                for r in chosen:
                    tiles.append(panel(dataset/r["image"], labels[r["image"]],
                                       f"GT {r['camera']}/{r['weather']} | FREE {r['free']} OCC {r['occupied']}\nGreen=FREE, red=OCCUPIED"))
                    boxes = [[p[0], *p[2:]] for p in predictions[r["image"]] if p[1] >= .25]
                    tiles.append(panel(dataset/r["image"], boxes,
                                       f"{name} conf=.25 | FREE {r['predicted_free']} OCC {r['predicted_occupied']}\nWorst count error in this camera"))
                save_sheet(tiles, output/(name+"_worst.jpg"))
                write_json(output/(name+"_worst.json"), chosen)
            # Personal images: same preprocessing and grid, qualitative comparison only.
            for path in own:
                result = model.predict(str(path.resolve()), imgsz=size, conf=min(THRESHOLDS), device=0,
                                       rect=False, quantize=32, verbose=False, max_det=300, save=False)[0]
                pred = np.column_stack([result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()]).tolist()
                for threshold in THRESHOLDS:
                    selected = [p for p in pred if p[1] >= threshold]
                    free = sum(int(p[0]) == 0 for p in selected)
                    occupied = sum(int(p[0]) == 1 for p in selected)
                    report["own_photos"].append(dict(image=path.name, model=model_name, imgsz=size,
                                                    threshold=threshold, free=free, occupied=occupied))
                    title = f"{model_name} {size} conf={threshold} | FREE {free} OCC {occupied}\nGreen=FREE, red=OCCUPIED; no ground truth"
                    if threshold == .25:
                        own_panels[path.name].append(panel(path, [[p[0], *p[2:]] for p in selected], title))
                    if model_name == "full" and size == 640 and threshold in (.01, .05, .1, .25, .5, .75):
                        threshold_panels[path.name].append(panel(path, [[p[0], *p[2:]] for p in selected], title))
                write_json(output/f"{path.stem}_{name}_predictions.json", pred)
            write_json(output/"report.json", report)
        del model
        torch.cuda.empty_cache()
    for path in own:
        # Columns are smoke/full, rows are 640/960/1280.
        tiles = own_panels[path.name]
        save_sheet([tiles[i] for i in (0,3,1,4,2,5)], output/f"{path.stem}_models.jpg")
        save_sheet(threshold_panels[path.name], output/f"{path.stem}_thresholds.jpg")
    if {key:digest(path) for key,path in weights.items()} != hashes or digest(dataset/"manifest.csv") != manifest_hash:
        raise RuntimeError("Source fingerprint changed during evaluation")
    report["source_fingerprints_unchanged"] = True
    report["elapsed_seconds"] = time.perf_counter()-started
    write_json(output/"report.json", report)
    progress(state="completed")


if __name__ == "__main__":
    main()
