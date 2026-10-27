"""Review a completed camera experiment and predict personal photos on GPU."""

import argparse
from collections import defaultdict
import csv
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.diagnose_models import configure_nms_budget, panel, read_labels, write_json
from src.train_camera_experiment import completed_epoch_count, verify
from src.train_full import file_hash


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k.strip():v for k,v in row.items()} for row in csv.DictReader(stream)]


def draw_full(path, predictions, title, target):
    with Image.open(path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for p in predictions:
        draw.rectangle(tuple(p[2:]), outline="#00ed73" if p[0]==0 else "#ff4545", width=2)
    result = Image.new("RGB", (image.width,image.height+60), "#15202b")
    result.paste(image,(0,60))
    ImageDraw.Draw(result).text((8,6), title, font=ImageFont.load_default(size=16), fill="white")
    result.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,default=Path("outputs/camera_experiment_v1"))
    parser.add_argument("--output",type=Path,default=Path("outputs/camera_experiment_review_v1"))
    args = parser.parse_args()
    experiment, out = args.experiment.resolve(), args.output.resolve()
    if json.loads((experiment/"progress.json").read_text())["state"]!="completed":
        raise ValueError("Experiment has not completed")
    out.mkdir(parents=True,exist_ok=False)
    print("Verifying data and model fingerprints",flush=True)
    verify(experiment)
    comparison = json.loads((experiment/"comparison.json").read_text())
    rows=read_csv(experiment/"data/manifest.csv")
    by_group=defaultdict(list)
    for row in rows:
        if row["role"] in ("val","holdout"):
            by_group[(row["role"],row["camera"],row["weather"])].append(row)
    samples=[sorted(g,key=lambda r:r["image"])[len(g)//2] for _,g in sorted(by_group.items())]
    models={name:experiment/name/"weights/best.pt" for name in ("baseline","augmented")}
    models["previous_full"]=Path("outputs/full_yolo26n_v1/weights/best.pt").resolve()
    training={}
    for name in ("baseline","augmented"):
        run=experiment/name
        progress=json.loads((run/"progress.json").read_text())
        assert file_hash(models[name])==progress["best_sha256"]
        epochs=read_csv(run/"results.csv")
        best=max(epochs,key=lambda r:float(r["metrics/mAP50-95(B)"]))
        training[name]=dict(completed_epochs=completed_epoch_count(run),best_epoch=int(best["epoch"]),
                            training_seconds=float(epochs[-1]["time"]),
                            training_and_evaluation_seconds=progress["elapsed_seconds"],
                            selected_conf=comparison["results"][name]["selected_conf"])
        # Repair only derived epoch metadata, keeping the old value and reason.
        for filename in ("progress.json","training_complete.json"):
            path=run/filename
            obj=json.loads(path.read_text())
            if obj["completed_epochs"]!=training[name]["completed_epochs"]:
                obj["completed_epochs_before_correction"]=obj["completed_epochs"]
                obj["completed_epochs"]=training[name]["completed_epochs"]
                obj["epoch_count_source"]="results.csv; final validation increments trainer.epoch"
                write_json(path,obj)
    os.environ["YOLO_CONFIG_DIR"]=str(out/"ultralytics_config")
    os.environ["YOLO_AUTOINSTALL"]="false"
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    configure_nms_budget()
    photos=sorted(p for p in Path("data/my_photo").iterdir() if p.suffix.lower() in {".jpg",".png",".jpeg"})
    report=dict(training=training,comparison=comparison,own_photos=[],sample_checks=[],test_set_used=False,
                weights={k:dict(path=str(v),sha256=file_hash(v)) for k,v in models.items()},
                caveat="previous_full saw PUCPR during training; only shown on personal photos, not as an unseen-camera baseline")
    photo_tiles=defaultdict(list)
    sample_tiles={r["image"]:[panel(experiment/"data"/r["image"],read_labels(experiment/"data",r),
                                  f"GT {r['camera']}/{r['weather']}\nFREE {r['free']} | OCCUPIED {r['occupied']}")] for r in samples}
    for name,weight in models.items():
        print(f"Predicting personal photos and checking samples: {name}",flush=True)
        model=YOLO(str(weight))
        assert model.names=={0:"FREE",1:"OCCUPIED"}
        selected_conf=training[name]["selected_conf"] if name in training else .6
        for path in photos:
            result=model.predict(str(path.resolve()),imgsz=640,device=0,rect=False,quantize=32,
                                 conf=.25,max_det=300,verbose=False,save=False)[0]
            pred=np.column_stack([result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.xyxy.cpu().numpy()]).tolist()
            write_json(out/f"{path.stem}_{name}_boxes.json",pred)
            for threshold in sorted({.25,selected_conf}):
                keep=[p for p in pred if p[1]>=threshold]
                free=sum(p[0]==0 for p in keep)
                occupied=sum(p[0]==1 for p in keep)
                filename=f"{path.stem}_{name}_conf{str(threshold).replace('.','_')}.jpg"
                title=f"{name} | conf {threshold} | FREE {free} | OCCUPIED {occupied}\nGreen=FREE, red=OCCUPIED"
                draw_full(path,keep,title,out/filename)
                report["own_photos"].append(dict(image=path.name,model=name,conf=threshold,free=free,occupied=occupied,preview=filename))
                if threshold==.25:
                    photo_tiles[path.name].append(panel(path,[[p[0],*p[2:]] for p in keep],title))
        if name in training:
            expected={}
            for role in ("val","holdout"):
                expected.update({r["image"]:r for r in read_csv(experiment/name/(role+"_counts.csv")) if float(r["threshold"])==.25})
            for row in samples:
                path=experiment/"data"/row["image"]
                result=model.predict(str(path),imgsz=640,device=0,rect=False,quantize=32,conf=.25,
                                     max_det=300,verbose=False,save=False)[0]
                pred=np.column_stack([result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.xyxy.cpu().numpy()]).tolist()
                free=sum(p[0]==0 for p in pred)
                occupied=sum(p[0]==1 for p in pred)
                reference=expected[row["image"]]
                report["sample_checks"].append(dict(model=name,image=row["image"],role=row["role"],free=free,occupied=occupied,
                      matches_saved_counts=free==int(reference["predicted_free"]) and occupied==int(reference["predicted_occupied"])))
                sample_tiles[row["image"]].append(panel(path,[[p[0],*p[2:]] for p in pred],
                                                       f"{name} | conf .25\nFREE {free} | OCCUPIED {occupied}"))
        del model
        torch.cuda.empty_cache()
    for photo,tiles in photo_tiles.items():
        sheet=Image.new("RGB",(640,410*3),"#15202b")
        for i,tile in enumerate(tiles):
            sheet.paste(tile,(0,i*410))
        sheet.save(out/(Path(photo).stem+"_comparison.jpg"))
    for role in ("val","holdout"):
        selected=[r for r in samples if r["role"]==role]
        sheet=Image.new("RGB",(1920,410*len(selected)),"#15202b")
        for i,row in enumerate(selected):
            for j,tile in enumerate(sample_tiles[row["image"]]):
                sheet.paste(tile,(j*640,i*410))
        sheet.save(out/(role+"_sample_comparison.jpg"))
    for key,info in report["weights"].items():
        assert file_hash(models[key])==info["sha256"]
    report["weights_unchanged"]=True
    write_json(out/"report.json",report)
    print(json.dumps(dict(training=training,own_photos=report["own_photos"],
                         sample_checks_match=all(r["matches_saved_counts"] for r in report["sample_checks"])),indent=2),flush=True)


if __name__=="__main__":
    main()
