from pathlib import Path

import pytest

from src.train_camera_experiment import VARIANTS, completed_epoch_count, partition, safe_source, training_options


def rows():
    return [dict(image=f"{camera}_{split}.jpg",camera=camera,split=split,date=str(day))
            for camera in ("PUCPR","UFPR04","UFPR05") for split,day in (("train",1),("val",2),("test",3))]


def test_holdout_never_used_for_training_or_checkpoint_selection():
    groups=partition(rows(),"PUCPR")
    assert len(groups["train"])==2 and len(groups["val"])==2 and len(groups["holdout"])==1
    assert all(r["camera"]!="PUCPR" for key in ("train","val") for r in groups[key])
    assert all(r["camera"]=="PUCPR" for r in groups["holdout"])
    assert all(r["split"]!="test" for group in groups.values() for r in group)


def test_partition_rejects_duplicate_or_leaking_dates():
    with pytest.raises(ValueError,match="Duplicate"):
        partition(rows()+[rows()[0]],"PUCPR")
    changed=rows()
    changed[1]["date"]="1"
    with pytest.raises(ValueError,match="dates overlap"):
        partition(changed,"PUCPR")


def test_source_cannot_escape_image_tree(tmp_path):
    with pytest.raises(ValueError,match="outside"):
        safe_source(tmp_path,"../other.jpg","images")


def test_only_augmentation_and_output_differ_between_variants():
    experiment=dict(epochs=30,patience=8,batch=8,workers=2,variants=VARIANTS)
    a=training_options(experiment,Path("run"),"baseline")
    b=training_options(experiment,Path("run"),"augmented")
    differences={key for key in a if a[key]!=b[key]}
    assert differences=={"save_dir","scale","degrees","translate","perspective"}
    assert a["seed"]==b["seed"]==42
    assert a["data"]==str(Path("run/trainval.yaml"))


def test_epoch_count_comes_from_completed_csv_rows(tmp_path):
    (tmp_path / "results.csv").write_text(" epoch, loss\n1,0.9\n2,0.8\n3,0.7\n")
    assert completed_epoch_count(tmp_path) == 3
    (tmp_path / "results.csv").write_text("epoch,loss\n")
    with pytest.raises(ValueError, match="No completed epochs"):
        completed_epoch_count(tmp_path)
