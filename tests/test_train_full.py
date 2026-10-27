import json

import pytest
import yaml

from src.train_full import prepare_run, write_status


def test_full_training_preserves_data_and_excludes_test_config(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "manifest.csv").write_text("original manifest")
    (dataset / "summary.json").write_text(json.dumps({"splits": {s: {"images": 10} for s in ("train", "val", "test")}}))
    before = (dataset / "manifest.csv").read_bytes()
    output = tmp_path / "run"
    config, _ = prepare_run(dataset, output)
    contents = yaml.safe_load(config.read_text())
    assert "test" not in contents
    assert contents["path"] == dataset.as_posix()
    assert (dataset / "manifest.csv").read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_run(dataset, output)
    prepare_run(dataset, output, resume=True)
    (dataset / "manifest.csv").write_text("changed data")
    with pytest.raises(ValueError, match="manifest changed"):
        prepare_run(dataset, output, resume=True)


def test_progress_written_as_complete_json(tmp_path):
    path = tmp_path / "progress.json"
    write_status(path, {"state": "training", "completed_epochs": 1})
    write_status(path, {"state": "completed", "completed_epochs": 30})
    assert json.loads(path.read_text())["state"] == "completed"
    assert not path.with_suffix(".json.tmp").exists()
