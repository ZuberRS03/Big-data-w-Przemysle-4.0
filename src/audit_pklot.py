"""Audit full PKLot images and XML; write reproducible metadata and previews.

Run from the repository root: uv run --frozen --no-sync python -m src.audit_pklot
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.pklot import read_annotations


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preview(image_path, xml_path, destination, title):
    spaces = read_annotations(xml_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    for space in spaces:
        if not space.polygon:
            continue
        color = {0: "#00ff63", 1: "#ff4747", None: "#ffcf33"}[space.occupied]
        draw.line(list(space.polygon) + [space.polygon[0]], fill=color, width=2)
        x = sum(p[0] for p in space.polygon) / len(space.polygon)
        y = sum(p[1] for p in space.polygon) / len(space.polygon)
        draw.text((x, y), space.identifier, font=font, fill="white", stroke_width=1, stroke_fill="black", anchor="mm")
    counts = Counter(s.occupied for s in spaces)
    image = ImageOps.expand(image, border=(0, 58, 0, 0), fill="#15202b")
    draw = ImageDraw.Draw(image)
    draw.text((12, 5), title, font=font, fill="white")
    draw.text((12, 31), f"XML labels (not predictions) | FREE: {counts[0]} | OCCUPIED: {counts[1]} | UNKNOWN: {counts[None]}", font=font, fill="white")
    image.save(destination)
    return image


def audit(data_root: Path, output: Path):
    if not data_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_root}")
    image_paths = sorted(p for p in data_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    xml_paths = sorted(data_root.rglob("*.xml"))
    if not image_paths or not xml_paths:
        raise ValueError("Expected full parking images and XML files; check --data-root")
    output.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    images = {p.relative_to(data_root).with_suffix("").as_posix(): p for p in image_paths}
    annotations = {p.relative_to(data_root).with_suffix("").as_posix(): p for p in xml_paths}
    rows, issues = [], []
    totals, geometries, dimensions = Counter(), Counter(), Counter()
    samples = defaultdict(list)
    by_camera = defaultdict(Counter)
    groups = defaultdict(set)
    for index, key in enumerate(sorted(images.keys() | annotations.keys()), 1):
        image_path, xml_path = images.get(key), annotations.get(key)
        parts = Path(key).parts
        camera, weather, date = (parts[:3] if len(parts) >= 4 else ("unknown", "unknown", "unknown"))
        row = dict(image=image_path.relative_to(data_root).as_posix() if image_path else "",
                   xml=xml_path.relative_to(data_root).as_posix() if xml_path else "",
                   camera=camera, weather=weather, date=date, width="", height="",
                   free=0, occupied=0, unknown=0, spaces=0, fallback_geometry=0, invalid_geometry=0,
                   out_of_bounds=0, readable_image=False, readable_xml=False)
        def issue(kind, identifier="", detail=""):
            issues.append(dict(path=key, space_id=identifier, kind=kind, detail=detail))
        if image_path is None:
            issue("missing_image")
        else:
            try:
                with Image.open(image_path) as image:
                    image.load()  # Decode every image, not just its header.
                    row["width"], row["height"] = image.size
                row["readable_image"] = True
                dimensions[f'{row["width"]}x{row["height"]}'] += 1
            except (OSError, ValueError) as error:
                issue("unreadable_image", detail=str(error))
        if xml_path is None:
            issue("missing_xml")
        else:
            try:
                spaces = read_annotations(xml_path)
                row["readable_xml"] = True
                if not spaces:
                    issue("empty_annotation")
                for space in spaces:
                    row[{0: "free", 1: "occupied", None: "unknown"}[space.occupied]] += 1
                    row["spaces"] += 1
                    geometries[space.geometry_source] += 1
                    row["fallback_geometry"] += space.geometry_source == "rotatedRect"
                    row["invalid_geometry"] += space.geometry_source == "invalid"
                    for kind in space.issues:
                        issue(kind, space.identifier)
                    if row["readable_image"] and any(
                        x < 0 or y < 0 or x > row["width"] or y > row["height"] for x, y in space.polygon
                    ):
                        row["out_of_bounds"] += 1
                        issue("out_of_bounds", space.identifier)
                if row["readable_image"] and spaces:
                    samples[(camera, weather)].append((key, image_path, xml_path))
            except (OSError, ET.ParseError, ValueError) as error:
                issue("unreadable_xml", detail=str(error))
        for field in ("free", "occupied", "unknown", "spaces", "fallback_geometry", "invalid_geometry", "out_of_bounds"):
            totals[field] += row[field]
            by_camera[camera][field] += row[field]
        by_camera[camera]["images"] += image_path is not None
        by_camera[camera]["xml"] += xml_path is not None
        groups[camera].add(date)
        rows.append(row)
        if index % 1000 == 0:
            print(f"Audited {index} files/pairs", flush=True)
    write_csv(output / "inventory.csv", rows, list(rows[0]))
    write_csv(output / "issues.csv", issues, ["path", "space_id", "kind", "detail"])
    selected, tiles = [], []
    for (camera, weather), candidates in sorted(samples.items()):
        key, image_path, xml_path = candidates[len(candidates) // 2]
        destination = preview_dir / f"{camera}_{weather}.jpg"
        image = preview(image_path, xml_path, destination, key)
        tiles.append(ImageOps.pad(image, (640, 390), color="#15202b"))
        selected.append(dict(source=key, preview=destination.relative_to(output).as_posix()))
    # Include a rotatedRect-only annotation in a separate preview if present.
    fallback_row = next((r for r in rows if r["fallback_geometry"] and r["readable_image"] and r["readable_xml"]), None)
    if fallback_row:
        destination = preview_dir / "rotated_rect_fallback.jpg"
        preview(data_root / fallback_row["image"], data_root / fallback_row["xml"], destination, fallback_row["image"])
        selected.append(dict(source=fallback_row["image"], preview=destination.relative_to(output).as_posix()))
    unknown_row = max((r for r in rows if r["readable_image"] and r["readable_xml"]), key=lambda r: r["unknown"], default=None)
    if unknown_row and unknown_row["unknown"]:
        destination = preview_dir / "unknown_occupancy.jpg"
        preview(data_root / unknown_row["image"], data_root / unknown_row["xml"], destination, unknown_row["image"])
        selected.append(dict(source=unknown_row["image"], preview=destination.relative_to(output).as_posix()))
    if tiles:
        contact = Image.new("RGB", (1920, 390 * ((len(tiles) + 2) // 3)), "#15202b")
        for index, tile in enumerate(tiles):
            contact.paste(tile, ((index % 3) * 640, (index // 3) * 390))
        contact.save(output / "contact_sheet.jpg")
    summary = dict(data_root=str(data_root.resolve()), images=len(image_paths), xml_files=len(xml_paths),
                   paired_files=len(images.keys() & annotations.keys()),
                   readable_images=sum(r["readable_image"] for r in rows),
                   readable_xml=sum(r["readable_xml"] for r in rows), totals=dict(totals),
                   images_with_unknown_occupancy=sum(bool(r["image"]) and r["unknown"] > 0 for r in rows),
                   fully_labelled_valid_images=sum(
                       r["readable_image"] and r["readable_xml"] and r["spaces"] > 0
                       and r["unknown"] == 0 and r["invalid_geometry"] == 0 and r["out_of_bounds"] == 0
                       for r in rows
                   ),
                   geometry_sources=dict(geometries), dimensions=dict(dimensions),
                   issues=dict(Counter(i["kind"] for i in issues)),
                   by_camera={k: dict(v, days=len(groups[k])) for k, v in sorted(by_camera.items())},
                   previews=selected)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Audyt PKLot", "", f"Źródło: `{data_root.as_posix()}`", "",
             f"Zdjęcia: **{len(image_paths)}**; XML: **{len(xml_paths)}**; pary: **{summary['paired_files']}**.",
             f"Poprawnie odczytane obrazy: **{summary['readable_images']}**; XML: **{summary['readable_xml']}**.", "",
             f"Zdjęcia z nieznanym statusem co najmniej jednego miejsca: **{summary['images_with_unknown_occupancy']}**.",
             f"Pary z kompletnymi statusami i poprawną geometrią: **{summary['fully_labelled_valid_images']}**.", "",
             "| Ujęcie | Zdjęcia | Dni | FREE | OCCUPIED | Bez statusu / niepoprawny status |",
             "|---|---:|---:|---:|---:|---:|"]
    for camera, counts in summary["by_camera"].items():
        lines.append(f"| {camera} | {counts['images']} | {counts['days']} | {counts['free']} | {counts['occupied']} | {counts['unknown']} |")
    lines += ["", "## Geometria", "", *[f"- {k}: {v}" for k, v in geometries.items()],
              "", "## Problemy i braki", "", *[f"- {k}: {v}" for k, v in summary["issues"].items()],
              "", "Liczniki problemów dotyczą wystąpień; kilka problemów może dotyczyć tego samego miejsca.",
              "", "## Interpretacja i kolejny etap", "",
              "Podglądy pokazują adnotacje XML, nie predykcje modelu. Zielony: FREE, czerwony: OCCUPIED, żółty: nieznany status.",
              "Kontur ma pierwszeństwo. Gdy go brakuje, geometria jest odtwarzana z rotatedRect; osobny podgląd pozwala ją sprawdzić.",
              "Nieznany status nie jest zamieniany na FREE. Przed konwersją trzeba ustalić politykę dla takich zdjęć: pominięcie samych etykiet w detektorze może uczyć traktowania tych miejsc jako tła.",
              "Najpierw zatwierdzić geometrię i obsługę braków, następnie przygotować podział według dni oraz konwersję do YOLO. Zachować osobny test na niewidzianym ujęciu.",
              "Liczby dotyczą pełnych zdjęć i XML. Katalog PKLotSegmented nie jest skanowany.", "",
              "## Pliki", "", "- `inventory.csv`: metadane i liczności dla każdego zdjęcia/XML.",
              "- `issues.csv`: wszystkie wykryte problemy z identyfikatorem miejsca.",
              "- `summary.json`: podsumowanie do dalszego przetwarzania.",
              "- `contact_sheet.jpg` i `previews/`: wizualna kontrola adnotacji."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "previews"}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/PKLot/PKLot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pklot_audit"))
    args = parser.parse_args()
    audit(args.data_root, args.output)


if __name__ == "__main__":
    main()
