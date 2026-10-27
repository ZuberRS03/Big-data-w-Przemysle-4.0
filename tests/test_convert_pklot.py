import csv
from pathlib import Path

from PIL import Image
import pytest
import yaml

from src.convert_pklot import Frame, capture_day, convert, scan_frames, split_frames, yolo_box
from src.pklot import ParkingSpace


def frame(camera, day):
    return Frame(Path(camera) / "Sunny" / day / "image.jpg", camera, "Sunny", day,
                 100, 100, (), 1, 1)


def frames():
    return [frame(camera, f"2026-10-{day:02}") for camera in ("A", "B") for day in range(1, 11)]


def test_polygon_envelope_and_normalization():
    space = ParkingSpace("1", 1, ((10, 20), (30, 10), (50, 40), (20, 60)), "contour", ())
    assert yolo_box(space, 100, 200) == "1 0.30000000 0.17500000 0.40000000 0.25000000"


def test_full_image_box_and_class_zero():
    space = ParkingSpace("1", 0, ((0, 0), (100, 0), (100, 50), (0, 50)), "contour", ())
    assert yolo_box(space, 100, 50) == "0 0.50000000 0.50000000 1.00000000 1.00000000"


@pytest.mark.parametrize("polygon", [((0, 0), (110, 0), (10, 10)),
                                    ((-1, 0), (10, 0), (10, 10)),
                                    ((0, 0), (0, 5), (0, 10)),
                                    ((float("nan"), 0), (10, 0), (10, 10))])
def test_bad_boxes_rejected(polygon):
    with pytest.raises(ValueError):
        yolo_box(ParkingSpace("1", 0, polygon, "contour", ()), 100, 100)


def test_unknown_is_not_converted():
    with pytest.raises(ValueError):
        yolo_box(ParkingSpace("1", None, ((0, 0), (10, 0), (10, 10)), "contour", ()), 100, 100)


def test_split_is_repeatable_and_keeps_dates_together_across_cameras():
    samples = frames()
    result = split_frames(samples, seed=42)
    assert result == split_frames(list(reversed(samples)), seed=42)
    assert result != split_frames(samples, seed=7)
    assert set(result.values()) == {"train", "val", "test"}
    for day in {f.day for f in samples}:
        assert len({result[f.relative.as_posix()] for f in samples if f.day == day}) == 1


def test_camera_holdout_is_absent_from_training_and_validation():
    samples = frames()
    result = split_frames(samples, holdout_camera="B")
    assert all((result[f.relative.as_posix()] == "test") == (f.camera == "B") for f in samples)
    assert {result[f.relative.as_posix()] for f in samples if f.camera == "A"} == {"train", "val"}


@pytest.mark.parametrize("kwargs", [{"val_fraction": 0}, {"test_fraction": float("nan")},
                                    {"val_fraction": .6, "test_fraction": .5}, {"holdout_camera": "C"}])
def test_invalid_splits_rejected(kwargs):
    with pytest.raises(ValueError):
        split_frames(frames(), **kwargs)


def test_too_few_days_rejected():
    with pytest.raises(ValueError):
        split_frames([frame("A", "2026-10-01")])


def test_original_dataset_date_typo_recovered_from_filename():
    relative = Path("UFPR05/Sunny/2013-14-16/2013-04-16_06_30_00.jpg")
    assert capture_day(relative) == ("2013-04-16", "filename_invalid_folder")


@pytest.mark.parametrize("relative", ["A/Sunny/2013-04-15/2013-04-16_06_30_00.jpg",
                                      "A/Sunny/2013-14-16/image.jpg"])
def test_ambiguous_or_missing_dates_rejected(relative):
    with pytest.raises(ValueError):
        capture_day(Path(relative))


def test_recovered_date_is_used_for_grouping(tmp_path):
    write_sample(tmp_path, "UFPR05", "2013-14-16", "2013-04-16_06_30_00")
    selected, excluded = scan_frames(tmp_path)
    assert not excluded
    assert selected[0].day == "2013-04-16"
    assert selected[0].date_source == "filename_invalid_folder"


def write_sample(root, camera, day, name="image", unknown=False):
    directory = root / camera / "Sunny" / day
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / f"{name}.jpg"
    Image.new("RGB", (100, 100), color=(60, 100, 140)).save(image)
    xml = '<parking><space id="1" occupied="0"><contour><point x="10" y="10"/><point x="40" y="10"/><point x="40" y="40"/></contour></space><space id="2" occupied="1"><contour><Point x="50" y="50"/><Point x="90" y="50"/><Point x="90" y="90"/></contour></space></parking>'
    if unknown:
        xml = xml.replace(' occupied="1"', '')
    image.with_suffix(".xml").write_text(xml, encoding="utf-8")
    return image


def test_conversion_roundtrip_exclusion_and_source_preservation(tmp_path):
    root = tmp_path / "źródło danych"
    for day in range(1, 11):
        write_sample(root, "A", f"2026-10-{day:02}")
    unknown = write_sample(root, "A", "2026-10-01", "unknown", unknown=True)
    missing = write_sample(root, "A", "2026-10-01", "missing")
    missing.with_suffix(".xml").unlink()
    original_image, original_xml = unknown.read_bytes(), unknown.with_suffix(".xml").read_bytes()
    output = tmp_path / "wynik YOLO"
    summary = convert(root, output)
    assert summary["images"] == 10
    assert summary["excluded"] == 2
    assert summary["exclusion_reasons"] == {"missing_xml": 1, "missing_occupancy": 1}
    config = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert Path(config["path"]) == output
    assert config["names"] == {0: "FREE", 1: "OCCUPIED"}
    seen = set()
    for split in ("train", "val", "test"):
        entries = (output / config[split]).read_text(encoding="utf-8").splitlines()
        assert entries
        for entry in entries:
            assert entry.startswith("./images/")
            assert entry not in seen
            seen.add(entry)
            image = output / entry
            relative = image.relative_to(output / "images")
            assert image.read_bytes() == (root / relative).read_bytes()
            labels = (output / "labels" / relative.with_suffix(".txt")).read_text().splitlines()
            assert len(labels) == 2
            # Recover pixel bounds from the serialized label, independently.
            for line, expected in zip(labels, ((0, 10, 10, 40, 40), (1, 50, 50, 90, 90))):
                cls, x, y, w, h = map(float, line.split())
                assert (cls, (x - w / 2) * 100, (y - h / 2) * 100,
                        (x + w / 2) * 100, (y + h / 2) * 100) == pytest.approx(expected)
    with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(seen) == 10
    assert not (output / "INCOMPLETE").exists()
    assert (output / "previews/contact_sheet.jpg").exists()
    assert unknown.read_bytes() == original_image
    assert unknown.with_suffix(".xml").read_bytes() == original_xml
    with pytest.raises(FileExistsError):
        convert(root, output)


def test_reject_destination_inside_source(tmp_path):
    with pytest.raises(ValueError, match="contain each other"):
        convert(tmp_path, tmp_path / "output")


def test_bad_geometry_excludes_entire_image(tmp_path):
    image = write_sample(tmp_path, "A", "2026-10-01")
    xml = image.with_suffix(".xml")
    xml.write_text(xml.read_text().replace('x="90"', 'x="150"'))
    selected, excluded = scan_frames(tmp_path)
    assert selected == []
    assert excluded[0]["reasons"] == "invalid_image_or_box"
