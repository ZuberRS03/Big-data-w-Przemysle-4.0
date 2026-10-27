from types import SimpleNamespace

import pytest

from fixed_parking.evaluate import check_spaces, select_test


def test_test_selection_rejects_leakage_and_duplicate_images():
    rows = [dict(camera="UFPR05", split="train", date="a", image="train"),
            dict(camera="UFPR05", split="test", date="b", image="test"),
            dict(camera="UFPR04", split="test", date="b", image="other")]
    assert select_test(rows, "UFPR05") == [rows[1]]
    with pytest.raises(ValueError, match="overlapping"):
        select_test(rows + [dict(camera="UFPR04", split="val", date="b", image="val")], "UFPR05")
    with pytest.raises(ValueError, match="Duplicate"):
        select_test(rows + [rows[0]], "UFPR05")


def test_geometry_and_label_order_are_checked():
    layout = dict(reference_size=[100, 100], slots=[dict(id="a", polygon=[[0, 0], [1, 0], [1, 1]]),
                                                  dict(id="b", polygon=[[0, 0], [1, 1], [0, 1]])])
    a = SimpleNamespace(identifier="a", occupied=1, polygon=((0, 0), (100, 0), (100, 100)), issues=[])
    b = SimpleNamespace(identifier="b", occupied=0, polygon=((0, 0), (100, 100), (0, 100)), issues=[])
    assert check_spaces(layout, [b, a]) == [1, 0]
    b.polygon = ((0, 0), (99, 100), (0, 100))
    with pytest.raises(ValueError, match="geometry"):
        check_spaces(layout, [a, b])


def test_absent_class_is_excluded_from_group_balanced_accuracy():
    from fixed_parking.report import display_metrics
    raw = dict(by_group=dict(date={"empty": dict(confusion_matrix=[[99, 1], [0, 0]], balanced_accuracy=.495)}))
    derived = display_metrics(raw)
    assert derived["by_group"]["date"]["empty"]["balanced_accuracy"] == .99
    assert derived["by_group"]["date"]["empty"]["recall"] == [.99, None]
    assert raw["by_group"]["date"]["empty"]["balanced_accuracy"] == .495
