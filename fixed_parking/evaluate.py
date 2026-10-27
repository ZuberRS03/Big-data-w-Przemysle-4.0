"""Audit validation mistakes, then evaluate a frozen model on held-out dates."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from fixed_parking.core import load_layout
from fixed_parking.train import metrics
from src.pklot import read_annotations
from src.train_full import file_hash, write_status

SOURCE = Path("data/pklot_yolo_v1")
RAW = Path("data/PKLot/PKLot")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def check_spaces(layout, spaces):
    by_id = {s.identifier: s for s in spaces}
    if len(by_id) != len(spaces) or set(by_id) != {s["id"] for s in layout["slots"]}:
        raise ValueError("Missing or duplicate slot IDs")
    labels = []
    for slot in layout["slots"]:
        space = by_id[slot["id"]]
        expected = np.asarray(slot["polygon"]) * layout["reference_size"]
        actual = np.asarray(space.polygon)
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError("Camera geometry changed")
        if space.occupied not in (0, 1) or space.issues:
            raise ValueError("Invalid XML annotation")
        labels.append(space.occupied)
    return labels


def select_test(rows, camera):
    selected = [r for r in rows if r["camera"] == camera and r["split"] == "test"]
    development = {r["date"] for r in rows if r["split"] in ("train", "val")}
    if not selected or development & {r["date"] for r in selected}:
        raise ValueError("Empty test or overlapping development/test dates")
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image in manifest")
    return selected


def error_sheet(records, layout, output, maximum=24):
    records = records[:maximum]
    if not records:
        return None
    sheet = Image.new("RGB", (1280, 290 * ((len(records) + 3) // 4)), "white")
    slots = {s["id"]: s for s in layout["slots"]}
    for i, r in enumerate(records):
        polygon = np.asarray(slots[r["slot_id"]]["polygon"]) * layout["reference_size"]
        with Image.open(SOURCE / r["image"]) as source:
            im = source.convert("RGB")
        ImageDraw.Draw(im).polygon([tuple(p) for p in polygon], outline="yellow", width=3)
        lo = np.maximum(polygon.min(0) - 35, 0).astype(int)
        hi = np.minimum(polygon.max(0) + 35, im.size).astype(int)
        crop = im.crop(tuple(map(int, (*lo, *hi))))
        crop.thumbnail((310, 225))
        panel = Image.new("RGB", (320, 290), "white")
        panel.paste(crop, ((320 - crop.width) // 2, 60))
        title = (f"{Path(r['image']).stem} / slot {r['slot_id']}\n"
                 f"XML={r['reference']} pred={r['predicted']} pOcc={r['probability']:.4f}")
        ImageDraw.Draw(panel).text((5, 5), title, fill="black", font_size=13)
        sheet.paste(panel, ((i % 4) * 320, (i // 4) * 290))
    sheet.save(output / "errors.jpg")
    return "errors.jpg"


def audit(run):
    report = json.loads((run / "report.json").read_text())
    if file_hash(run / "best.pt") != report["model_sha256"]:
        raise ValueError("Model changed")
    layout = load_layout(run / "layout.json")
    labels = np.load(run / "val_labels.npy")
    probabilities = np.load(run / "val_probabilities.npy")
    rows = read_rows(run / "val_frames.csv")
    n = len(layout["slots"])
    if len(labels) != len(rows) * n or len(probabilities) != len(labels):
        raise ValueError("Invalid validation cache length")
    output = run / "validation_audit"
    output.mkdir(exist_ok=False)
    records = []
    for index in np.flatnonzero(labels != (probabilities >= report["threshold"])):
        frame, slot = divmod(int(index), n)
        row = rows[frame]
        original = check_spaces(layout, read_annotations(RAW / row["source_xml"]))
        if original[slot] != labels[index]:
            raise ValueError("Cached label differs from original XML")
        records.append(dict(image=row["image"], source_xml=row["source_xml"], date=row["date"],
                            weather=row["weather"], slot_id=layout["slots"][slot]["id"],
                            reference=int(labels[index]), predicted=int(probabilities[index] >= report["threshold"]),
                            probability=float(probabilities[index]), xml_sha256=file_hash(RAW / row["source_xml"])))
    write_rows(output / "errors.csv", records)
    preview = error_sheet(records, layout, output)
    write_status(output / "audit.json", dict(error_count=len(records), error_sheet=preview,
                 threshold=report["threshold"], labels_changed=False, visual_review_completed=False))
    print(json.dumps(records), flush=True)


def final_test(run):
    audit_info = json.loads((run / "validation_audit/audit.json").read_text())
    if not audit_info["visual_review_completed"]:
        raise ValueError("Complete visual validation review before test")
    training = json.loads((run / "report.json").read_text())
    expected = {"best.pt": training["model_sha256"], "layout.json": training["dataset"]["layout_sha256"]}
    if file_hash(SOURCE / "manifest.csv") != training["dataset"]["source_manifest_sha256"]:
        raise ValueError("Manifest changed")
    for name, digest in expected.items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Changed {name}")
    layout = load_layout(run / "layout.json")
    output = run / "final_test"
    output.mkdir(exist_ok=False)  # Never silently repeat the final test.
    frozen = output / "frozen"
    frozen.mkdir()
    paths = [run / "best.pt", run / "layout.json"] + [Path("fixed_parking") / f for f in ("core.py", "model.py", "train.py", "evaluate.py")]
    hashes = {}
    for path in paths:
        shutil.copy2(path, frozen / path.name)
        hashes[str(path)] = file_hash(path)
        if file_hash(frozen / path.name) != hashes[str(path)]:
            raise ValueError("Snapshot copy mismatch")
    write_status(output / "frozen_protocol.json", dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        camera=training["dataset"]["camera"], threshold=training["threshold"], hashes=hashes,
        manifest_sha256=training["dataset"]["source_manifest_sha256"], device="cpu",
        scope="All held-out test frames for UFPR05, original labels, no tuning or corrections",
        validation_review_completed=True))
    started = time.perf_counter()
    try:
        from fixed_parking.model import Predictor
        predictor = Predictor(frozen / "best.pt", "cpu")
        if predictor.threshold != training["threshold"]:
            raise ValueError("Threshold mismatch")
        rows = select_test(read_rows(SOURCE / "manifest.csv"), training["dataset"]["camera"])
        write_rows(output / "test_frames.csv", rows)
        all_labels, all_probabilities, frames, records, inputs = [], [], [], [], []
        n = len(layout["slots"])
        for i, row in enumerate(rows):
            labels = check_spaces(layout, read_annotations(RAW / row["source_xml"]))
            if sum(labels) != int(row["occupied"]) or n - sum(labels) != int(row["free"]):
                raise ValueError("Manifest count mismatch")
            digest = file_hash(SOURCE / row["image"])
            if digest != file_hash(RAW / row["source_image"]):
                raise ValueError("Converted image differs from original")
            inputs.append(dict(image=row["image"], image_sha256=digest, xml_sha256=file_hash(RAW / row["source_xml"])))
            with Image.open(SOURCE / row["image"]) as image:
                result, _ = predictor.predict(np.asarray(image.convert("RGB")), layout)
            probabilities = [s["occupied_probability"] for s in result["slots"]]
            wrong = sum(y != int(s["occupied"]) for y, s in zip(labels, result["slots"]))
            frames.append(dict(image=row["image"], date=row["date"], weather=row["weather"],
                reference_free=n - sum(labels), predicted_free=result["free"], wrong_slots=wrong))
            for slot, y, probability in zip(layout["slots"], labels, probabilities):
                records.append(dict(image=row["image"], date=row["date"], weather=row["weather"], slot_id=slot["id"],
                    reference=y, predicted=int(probability >= predictor.threshold), probability=probability))
            all_labels.extend(labels)
            all_probabilities.extend(probabilities)
            if (i + 1) % 100 == 0:
                write_status(output / "progress.json", dict(state="evaluating", completed=i + 1, total=len(rows)))
                print(f"Test: {i + 1}/{len(rows)}", flush=True)
        y, p = np.asarray(all_labels), np.asarray(all_probabilities)
        scores = metrics(y, p, predictor.threshold, n)
        precision, recall = np.array(scores["precision"]), np.array(scores["recall"])
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
        scores.update(f1=f1.tolist(), macro_f1=float(f1.mean()), days=len({r["date"] for r in rows}),
            wrong_slots=sum(f["wrong_slots"] for f in frames),
            exact_slot_frame_fraction=sum(f["wrong_slots"] == 0 for f in frames) / len(frames),
            maximum_count_error=max(abs(f["predicted_free"] - f["reference_free"]) for f in frames))
        groups = {}
        for key in ("weather", "date"):
            groups[key] = {}
            for value in sorted({r[key] for r in rows}):
                mask = np.repeat([r[key] == value for r in rows], n)
                groups[key][value] = metrics(y[mask], p[mask], predictor.threshold, n)
        errors = [r for r in records if r["reference"] != r["predicted"]]
        for name, data in (("frame_predictions", frames), ("slot_predictions", records), ("errors", errors)):
            write_rows(output / (name + ".csv"), data)
        write_status(output / "input_hashes.json", inputs)
        preview = error_sheet(errors, layout, output)
        for path, digest in hashes.items():
            if file_hash(Path(path)) != digest:
                raise ValueError(f"Changed during test: {path}")
        report = dict(metrics=scores, by_group=groups, threshold=predictor.threshold, camera=predictor.camera,
            model_sha256=training["model_sha256"], error_sheet=preview, elapsed_seconds=time.perf_counter() - started,
            all_test_frames_evaluated=True, test_set_used=True, labels_changed=False, parameters_changed=False,
            scope="Known camera, held-out days; temporally correlated frames are not independent observations")
        write_status(output / "report.json", report)
        write_status(output / "progress.json", dict(state="completed", completed=len(rows), total=len(rows)))
        print(json.dumps(report), flush=True)
    except BaseException as error:
        write_status(output / "progress.json", dict(state="failed", error=str(error), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "test"))
    parser.add_argument("--run", type=Path, default=Path("outputs/fixed_parking_ufpr05_v1"))
    args = parser.parse_args()
    (audit if args.stage == "audit" else final_test)(args.run.resolve())


if __name__ == "__main__":
    main()
