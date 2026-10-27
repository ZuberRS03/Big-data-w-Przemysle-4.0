import csv

import pytest
import yaml

from src.train_smoke import prepare_subset, sample_groups


def rows():
    return [dict(image=f"images/{camera}/{weather}/{split}_{i}.jpg", split=split,
                 camera=camera, weather=weather, date=f"2026-10-{1 if split == 'train' else 2:02}",
                 free="1", occupied="2")
            for split in ("train", "val", "test") for camera in ("A", "B")
            for weather in ("Sunny", "Rainy") for i in range(20)]


def test_subset_is_repeatable_diverse_and_preserves_split():
    selected = sample_groups(rows(), "train", 3, 42)
    assert len(selected) == 12
    assert selected == sample_groups(list(reversed(rows())), "train", 3, 42)
    assert {r["split"] for r in selected} == {"train"}
    assert len({(r["camera"], r["weather"]) for r in selected}) == 4


def test_subset_config_cannot_select_test_set(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    records = rows()
    with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config_path, selected, stats = prepare_subset(dataset, tmp_path / "subset")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "test" not in config
    assert stats["train"]["images"] == 80
    assert stats["val"]["images"] == 60
    assert {r["split"] for r in selected} == {"val"}
    assert "_test_" not in config_path.with_name("train.txt").read_text()
    records[0]["date"] = "2026-10-02"
    with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="dates overlap"):
        prepare_subset(dataset, tmp_path / "bad")
