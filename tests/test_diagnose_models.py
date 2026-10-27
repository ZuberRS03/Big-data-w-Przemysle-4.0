import csv

import numpy as np
import pytest

from src.diagnose_models import aggregate, box_errors, validation_rows


def test_duplicate_does_not_hide_missed_place():
    gt = np.array([[0, 0, 0, 10, 10], [1, 20, 20, 30, 30]])
    got = box_errors([[0,.9,0,0,10,10], [0,.8,0,0,10,10]], gt, .25)
    assert got == dict(correct=1, wrong_class=0, duplicate=1, unmatched_prediction=0,
                       missed=1, predicted=2, reference=2)


def test_class_confusion_is_not_a_correct_detection():
    gt = np.array([[1, 0, 0, 10, 10]])
    got = box_errors([[0,.9,0,0,10,10], [1,.1,0,0,10,10]], gt, .25)
    assert got["wrong_class"] == 1 and got["correct"] == 0 and got["missed"] == 0


def test_empty_predictions_and_empty_labels():
    assert box_errors([], np.array([[0,0,0,1,1]]), .25)["missed"] == 1
    assert box_errors([[0,.9,0,0,1,1]], np.empty((0,5)), .25)["unmatched_prediction"] == 1


def test_total_counts_cannot_cancel_class_errors():
    row = dict(free=5, occupied=5, predicted_free=2, predicted_occupied=8,
               correct=7, wrong_class=3, duplicate=0, unmatched_prediction=0, missed=0, predicted=10, reference=10)
    got = aggregate([row])
    assert got["total_mae"] == 0 and got["class_count_mae"] == 3
    assert got["diagnostic_precision"] == .7


def test_subset_rejects_test_images(tmp_path):
    with (tmp_path/"manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "split"])
        writer.writeheader()
        writer.writerows([dict(image="a.jpg", split="val"), dict(image="b.jpg", split="test")])
    subset = tmp_path/"subset.txt"
    subset.write_text(str(tmp_path/"b.jpg"))
    with pytest.raises(ValueError, match="validation images only"):
        validation_rows(tmp_path, subset)
