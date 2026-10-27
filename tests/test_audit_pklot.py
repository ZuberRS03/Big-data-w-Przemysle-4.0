import csv
import json

from PIL import Image

from src.audit_pklot import audit


def test_audit_pairing_unknown_labels_and_previews(tmp_path):
    root = tmp_path / "data"
    day = root / "camera" / "Sunny" / "2026-10-24"
    day.mkdir(parents=True)
    for name in ("paired", "no_xml"):
        Image.new("RGB", (100, 100)).save(day / f"{name}.jpg")
    xml = '<parking><space id="1"><contour><Point x="10" y="10"/><Point x="50" y="10"/><Point x="50" y="50"/></contour></space></parking>'
    (day / "paired.xml").write_text(xml, encoding="utf-8")
    (day / "no_image.xml").write_text(xml.replace('id="1"', 'id="1" occupied="0"'), encoding="utf-8")
    output = tmp_path / "output"
    audit(root, output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 2
    assert summary["xml_files"] == 2
    assert summary["paired_files"] == 1
    assert summary["issues"] == {"missing_image": 1, "missing_xml": 1, "missing_occupancy": 1}
    assert summary["images_with_unknown_occupancy"] == 1
    assert summary["fully_labelled_valid_images"] == 0
    assert summary["totals"]["free"] == 1
    assert summary["totals"]["unknown"] == 1
    with (output / "inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3
    for filename in ("contact_sheet.jpg", "previews/camera_Sunny.jpg", "previews/unknown_occupancy.jpg"):
        with Image.open(output / filename) as image:
            image.load()
