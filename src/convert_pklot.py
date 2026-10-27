"""Convert full PKLot frames to a YOLO detection dataset with grouped splits."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

from src.pklot import ParkingSpace, read_annotations


@dataclass(frozen=True)
class Frame:
    relative: Path
    camera: str
    weather: str
    day: str
    width: int
    height: int
    labels: tuple[str, ...]
    free: int
    occupied: int
    date_source: str = "folder"


def capture_day(relative: Path) -> tuple[str, str]:
    """Use a valid filename date to recover malformed source folder dates."""
    folder = relative.parts[2]
    try:
        folder_day = date.fromisoformat(folder).isoformat()
    except ValueError:
        folder_day = None
    try:
        filename_day = date.fromisoformat(relative.stem.split("_")[0]).isoformat()
    except ValueError:
        filename_day = None
    if folder_day and filename_day and folder_day != filename_day:
        raise ValueError("Folder and filename dates disagree")
    if folder_day:
        return folder_day, "folder"
    if filename_day:
        return filename_day, "filename_invalid_folder"
    raise ValueError("Neither folder nor filename contains a valid date")


def yolo_box(space: ParkingSpace, width: int, height: int) -> str:
    """Enclose the original polygon in an axis-aligned, normalized box.

    Reject invalid/out-of-frame geometry instead of silently clipping it.
    """
    if width <= 0 or height <= 0 or space.occupied not in (0, 1) or space.issues:
        raise ValueError("Invalid dimensions or incomplete annotation")
    if len(space.polygon) < 3:
        raise ValueError("Missing polygon")
    if not all(math.isfinite(v) for point in space.polygon for v in point):
        raise ValueError("Non-finite coordinates")
    xs, ys = zip(*space.polygon)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Degenerate or out-of-frame box")
    values = ((left + right) / (2 * width), (top + bottom) / (2 * height),
              (right - left) / width, (bottom - top) / height)
    return f"{space.occupied} " + " ".join(f"{value:.8f}" for value in values)


def scan_frames(root: Path) -> tuple[list[Frame], list[dict]]:
    frames, excluded = [], []
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError("No images found; point --data-root at the full-frame PKLot directory")
    label_keys = set()
    image_keys = {p.with_suffix("") for p in images}
    for index, path in enumerate(images, 1):
        relative = path.relative_to(root)
        reasons = set()
        detail = ""
        if len(relative.parts) != 4:
            reasons.add("unexpected_layout")
        else:
            camera, weather, day, _ = relative.parts
            try:
                day, date_source = capture_day(relative)
            except ValueError as error:
                reasons.add("invalid_date")
                detail = str(error)
        xml = path.with_suffix(".xml")
        spaces = []
        if not xml.is_file():
            reasons.add("missing_xml")
        else:
            try:
                spaces = read_annotations(xml)
                if not spaces:
                    reasons.add("empty_annotation")
                reasons.update(issue for space in spaces for issue in space.issues)
            except (ET.ParseError, ValueError, OSError) as error:
                reasons.add("unreadable_xml")
                detail = str(error)
        labels = ()
        if not reasons:
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF orientation would change coordinates")
                    image.verify()
                labels = tuple(yolo_box(space, width, height) for space in spaces)
                if len(set(labels)) != len(labels):
                    reasons.add("duplicate_yolo_label")
            except (ValueError, OSError) as error:
                reasons.add("invalid_image_or_box")
                detail = str(error)
        if reasons:
            excluded.append(dict(image=relative.as_posix(), reasons=";".join(sorted(reasons)), detail=detail))
        else:
            label_key = relative.with_suffix(".txt").as_posix().casefold()
            if label_key in label_keys:
                raise ValueError(f"Multiple images map to the same label: {relative}")
            label_keys.add(label_key)
            frames.append(Frame(relative, camera, weather, day, width, height, labels,
                                sum(s.occupied == 0 for s in spaces), sum(s.occupied == 1 for s in spaces), date_source))
        if index % 1000 == 0:
            print(f"Checked {index}/{len(images)} source frames", flush=True)
    for xml in sorted(root.rglob("*.xml")):
        if xml.with_suffix("") not in image_keys:
            excluded.append(dict(image=xml.relative_to(root).as_posix(), reasons="missing_image", detail=""))
    return frames, excluded


def split_frames(frames: list[Frame], seed: int = 42, val_fraction: float = .15,
                 test_fraction: float = .15, holdout_camera: str | None = None) -> dict[str, str]:
    if not all(math.isfinite(x) and 0 < x < 1 for x in (val_fraction, test_fraction)):
        raise ValueError("Split fractions must be finite and between 0 and 1")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    cameras = {f.camera for f in frames}
    if holdout_camera and holdout_camera not in cameras:
        raise ValueError(f"Holdout camera not present: {holdout_camera}")
    pool = [f for f in frames if f.camera != holdout_camera]
    days = sorted({f.day for f in pool})
    random.Random(seed).shuffle(days)
    n_val = max(1, round(len(days) * val_fraction))
    n_test = 0 if holdout_camera else max(1, round(len(days) * test_fraction))
    if len(days) <= n_val + n_test:
        raise ValueError("Not enough distinct days for nonempty train/val/test splits")
    day_splits = {day: "val" if i < n_val else "test" if i < n_val + n_test else "train"
                  for i, day in enumerate(days)}
    assignments = {f.relative.as_posix(): "test" if f.camera == holdout_camera else day_splits[f.day]
                   for f in frames}
    if set(assignments.values()) != {"train", "val", "test"}:
        raise ValueError("All three splits must contain images")
    return assignments


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_previews(output: Path, frames: list[Frame], assignments: dict[str, str]):
    """Draw exported labels (read back from disk), not source polygons."""
    preview_dir = output / "previews"
    preview_dir.mkdir()
    tiles = []
    for camera in sorted({f.camera for f in frames}):
        for split in ("train", "val", "test"):
            candidates = [f for f in frames if f.camera == camera and assignments[f.relative.as_posix()] == split]
            if not candidates:
                continue
            frame = candidates[len(candidates) // 2]
            with Image.open(output / "images" / frame.relative) as original:
                image = original.convert("RGB")
            draw = ImageDraw.Draw(image)
            labels = (output / "labels" / frame.relative.with_suffix(".txt")).read_text().splitlines()
            for line in labels:
                cls, x, y, w, h = map(float, line.split())
                box = ((x - w / 2) * frame.width, (y - h / 2) * frame.height,
                       (x + w / 2) * frame.width, (y + h / 2) * frame.height)
                draw.rectangle(box, outline="#00ff63" if cls == 0 else "#ff4747", width=2)
            image = ImageOps.expand(image, border=(0, 36, 0, 0), fill="#15202b")
            ImageDraw.Draw(image).text((10, 8), f"{split} | {frame.relative.as_posix()} | FREE {frame.free}, OCCUPIED {frame.occupied}",
                                       fill="white", font=ImageFont.load_default(size=14))
            image.save(preview_dir / f"{camera}_{split}.jpg")
            tiles.append(ImageOps.pad(image, (640, 380), color="#15202b"))
    sheet = Image.new("RGB", (1920, 380 * math.ceil(len(tiles) / 3)), "#15202b")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3) * 640, (i // 3) * 380))
    sheet.save(preview_dir / "contact_sheet.jpg")


def convert(root: Path, output: Path, seed: int = 42, val_fraction: float = .15,
            test_fraction: float = .15, holdout_camera: str | None = None) -> dict:
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Source and destination must not contain each other")
    if output.exists():
        raise FileExistsError(f"Destination already exists; choose a new --output: {output}")
    frames, excluded = scan_frames(root)
    assignments = split_frames(frames, seed, val_fraction, test_fraction, holdout_camera)
    for split in ("train", "val", "test"):
        selected = [f for f in frames if assignments[f.relative.as_posix()] == split]
        if not sum(f.free for f in selected) or not sum(f.occupied for f in selected):
            raise ValueError(f"Split {split} does not contain both classes; review day allocation")
    required = sum((root / f.relative).stat().st_size for f in frames) + 100 * 1024**2
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < required:
        raise OSError(f"Not enough disk space; need approximately {required / 1024**3:.2f} GiB")
    output.mkdir()  # Refuse concurrent writers and never overwrite an existing dataset.
    marker = output / "INCOMPLETE"
    marker.write_text("Conversion in progress. Do not train until this marker is gone.\n")
    manifest = []
    for index, frame in enumerate(frames, 1):
        image_path = output / "images" / frame.relative
        label_path = output / "labels" / frame.relative.with_suffix(".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / frame.relative, image_path)
        label_path.write_text("\n".join(frame.labels) + "\n", encoding="utf-8")
        manifest.append(dict(source_image=frame.relative.as_posix(),
                             source_xml=frame.relative.with_suffix(".xml").as_posix(),
                             image=image_path.relative_to(output).as_posix(),
                             label=label_path.relative_to(output).as_posix(),
                             split=assignments[frame.relative.as_posix()], camera=frame.camera,
                             weather=frame.weather, date=frame.day, date_source=frame.date_source, width=frame.width,
                             height=frame.height, free=frame.free, occupied=frame.occupied))
        if index % 1000 == 0:
            print(f"Exported {index}/{len(frames)} images and labels", flush=True)
    write_csv(output / "manifest.csv", manifest, list(manifest[0]))
    write_csv(output / "excluded.csv", excluded, ["image", "reasons", "detail"])
    corrections = [dict(image=f.relative.as_posix(), folder_date=f.relative.parts[2], date=f.day)
                   for f in frames if f.date_source != "folder"]
    write_csv(output / "date_corrections.csv", corrections, ["image", "folder_date", "date"])
    stats = {}
    for split in ("train", "val", "test"):
        rows = [r for r in manifest if r["split"] == split]
        (output / f"{split}.txt").write_text("".join(f"./{r['image']}\n" for r in rows), encoding="utf-8")
        stats[split] = dict(images=len(rows), days=len({r["date"] for r in rows}),
                            free=sum(r["free"] for r in rows), occupied=sum(r["occupied"] for r in rows),
                            cameras=dict(Counter(r["camera"] for r in rows)),
                            weather=dict(Counter(r["weather"] for r in rows)))
    summary = dict(format="YOLO detect: axis-aligned xywh, normalized", names={0: "FREE", 1: "OCCUPIED"},
                   source=str(root), seed=seed, val_day_fraction=val_fraction,
                   test_day_fraction=None if holdout_camera else test_fraction,
                   holdout_camera=holdout_camera, images=len(frames), excluded=len(excluded),
                   recovered_dates=len(corrections),
                   exclusion_reasons=dict(Counter(reason for row in excluded for reason in row["reasons"].split(";"))),
                   splits=stats)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_previews(output, frames, assignments)
    config = dict(path=output.as_posix(), train="train.txt", val="val.txt", test="test.txt",
                  names={0: "FREE", 1: "OCCUPIED"})
    # Publish the training entry point only after all other output is complete.
    (output / "dataset.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    marker.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/PKLot/PKLot"))
    parser.add_argument("--output", type=Path, default=Path("data/pklot_yolo_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.15)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--holdout-camera", choices=["PUCPR", "UFPR04", "UFPR05"])
    args = parser.parse_args()
    convert(args.data_root, args.output, args.seed, args.val_fraction, args.test_fraction, args.holdout_camera)


if __name__ == "__main__":
    main()
