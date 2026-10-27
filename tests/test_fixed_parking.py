import copy

import numpy as np
import pytest

from fixed_parking.core import crops, validate_layout
from fixed_parking.train import metrics, select_frames


def layout():
    return dict(schema=1,reference_size=[100,100],slots=[dict(id="A",polygon=[[.1,.1],[.9,.1],[.5,.9]])])


def test_crop_masks_neighboring_background_and_has_expected_shape():
    image=np.full((100,100,3),255,np.uint8)
    crop=crops(image,layout())
    assert crop.shape==(1,64,64,3)
    assert (crop[0,20,32]==255).all()
    assert (crop[0,-1,0]==127).all()


def test_reject_mismatched_aspect_ratio_and_duplicate_ids():
    with pytest.raises(ValueError,match="Proporcje"):
        crops(np.zeros((100,200,3),np.uint8),layout())
    bad=layout(); bad["slots"].append(copy.deepcopy(bad["slots"][0]))
    with pytest.raises(ValueError,match="unikalne"):
        validate_layout(bad)


def test_self_intersection_and_out_of_bounds_are_rejected():
    bad=layout(); bad["slots"][0]["polygon"]=[[0,0],[1,1],[1,0],[0,1]]
    with pytest.raises(ValueError,match="wypukłym"):
        validate_layout(bad)
    bad["slots"][0]["polygon"][0]=[-.1,0]
    with pytest.raises(ValueError,match="zakresu"):
        validate_layout(bad)


def test_counts_and_accuracy_are_separate():
    result=metrics(np.array([0,1,0,1]),np.array([.9,.1,.1,.9]),.5,2)
    assert result["accuracy"]==.5
    assert result["occupied_count_mae"]==0
    assert result["confusion_matrix"]==[[1,1],[1,1]]


def test_split_preserves_days_and_does_not_use_test():
    rows=[dict(camera="C",split=s,date=d) for s,d in (("train","1"),("val","2"),("test","3"))]
    assert set(select_frames(rows,"C"))=={"train","val"}
    rows[1]["date"]="1"
    with pytest.raises(ValueError,match="overlapping"):
        select_frames(rows,"C")
