"""Prepare ROI data and train a fixed-camera occupancy classifier."""
import argparse
import csv
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
from PIL import Image

from fixed_parking.core import PREPROCESS, SIZE, crops, validate_layout
from src.pklot import read_annotations
from src.train_full import file_hash, write_status


def select_frames(rows,camera):
    groups={s:[r for r in rows if r["camera"]==camera and r["split"]==s] for s in ("train","val")}
    if not all(groups.values()) or {r["date"] for r in groups["train"]}&{r["date"] for r in groups["val"]}:
        raise ValueError("Empty split or overlapping train/validation dates")
    return groups


def metrics(labels,probabilities,threshold,slots):
    predicted=probabilities>=threshold
    matrix=np.zeros((2,2),dtype=np.int64)
    np.add.at(matrix,(labels.astype(int),predicted.astype(int)),1)
    recall=np.diag(matrix)/np.maximum(matrix.sum(1),1)
    precision=np.diag(matrix)/np.maximum(matrix.sum(0),1)
    errors=predicted.reshape(-1,slots).sum(1)-labels.reshape(-1,slots).sum(1)
    return dict(accuracy=float((labels==predicted).mean()),balanced_accuracy=float(recall.mean()),
                precision=precision.tolist(),recall=recall.tolist(),confusion_matrix=matrix.tolist(),
                occupied_count_mae=float(abs(errors).mean()),free_count_mae=float(abs(errors).mean()),
                exact_count_fraction=float((errors==0).mean()),frames=len(errors),slots=len(labels))


def prepare(output,camera):
    source=Path("data/pklot_yolo_v1").resolve()
    raw=Path("data/PKLot/PKLot").resolve()
    with (source/"manifest.csv").open(encoding="utf-8-sig",newline="") as stream:
        groups=select_frames(list(csv.DictReader(stream)),camera)
    ref=groups["train"][0]
    annotations=read_annotations(raw/ref["source_xml"])
    layout=dict(schema=1,name=camera,camera=camera,reference_size=[int(ref["width"]),int(ref["height"])],
                slots=[dict(id=s.identifier,polygon=[[x/int(ref["width"]),y/int(ref["height"])] for x,y in s.polygon]) for s in annotations])
    validate_layout(layout)
    expected={s.identifier:s.polygon for s in annotations}
    write_status(output/"layout.json",layout)
    config=Path("fixed_parking/configs")
    config.mkdir(exist_ok=True)
    config_file=config/(camera.lower()+".json")
    if config_file.exists() and json.loads(config_file.read_text())!=layout:
        raise FileExistsError("Existing camera configuration differs")
    if not config_file.exists(): write_status(config_file,layout)
    info=dict(camera=camera,preprocess=PREPROCESS,test_set_used=False,labels_corrected=False,
              source_manifest_sha256=file_hash(source/"manifest.csv"),layout_sha256=file_hash(output/"layout.json"),splits={})
    for split,rows in groups.items():
        n=len(rows)*len(annotations)
        x=np.lib.format.open_memmap(output/f"{split}_crops.npy",mode="w+",dtype=np.uint8,shape=(n,SIZE,SIZE,3))
        y=np.lib.format.open_memmap(output/f"{split}_labels.npy",mode="w+",dtype=np.uint8,shape=(n,))
        for i,row in enumerate(rows):
            spaces=read_annotations(raw/row["source_xml"])
            if {s.identifier:s.polygon for s in spaces}!=expected or any(s.occupied not in (0,1) or s.issues for s in spaces):
                raise ValueError("Camera geometry changed or labels incomplete; use a separate calibration")
            values={s.identifier:s.occupied for s in spaces}
            with Image.open(source/row["image"]) as image:
                batch=crops(np.asarray(image.convert("RGB")),layout)
            start=i*len(annotations)
            x[start:start+len(batch)]=batch
            y[start:start+len(batch)]=[values[s["id"]] for s in layout["slots"]]
            if i%200==0: print(f"Preparing {split}: {i}/{len(rows)} frames",flush=True)
        x.flush(); y.flush()
        info["splits"][split]=dict(frames=len(rows),crops=n,days=len({r["date"] for r in rows}),
                                  free=int((y==0).sum()),occupied=int((y==1).sum()))
        with (output/f"{split}_frames.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        del x,y
    write_status(output/"dataset.json",info)
    # Demo images selected by weather and filename, never by model performance.
    samples=[]
    for weather in sorted({r["weather"] for r in groups["val"]}):
        candidates=sorted([r for r in groups["val"] if r["weather"]==weather],key=lambda r:r["image"])
        samples.append(candidates[len(candidates)//2])
    write_status(output/"demo_samples.json",samples)
    return info,layout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/fixed_parking_ufpr05_v1"))
    p.add_argument("--camera",default="UFPR05")
    p.add_argument("--epochs",type=int,default=12)
    args=p.parse_args()
    if args.epochs<1: raise ValueError("Positive epochs required")
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    status=dict(state="preparing",test_set_used=False)
    def update(**kw):
        status.update(kw,elapsed_seconds=time.perf_counter()-started)
        write_status(output/"progress.json",status)
    update()
    try:
        info,layout=prepare(output,args.camera)
        import torch
        from torch import nn
        from fixed_parking.model import OccupancyNet, normalize, Predictor
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        model=OccupancyNet().cuda()
        data={}
        for split in ("train","val"):
            data[split]=(torch.from_numpy(np.load(output/f"{split}_crops.npy")).cuda(),
                         torch.from_numpy(np.load(output/f"{split}_labels.npy").astype(np.int64)).cuda())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scaler=torch.amp.GradScaler("cuda")
        counts=torch.bincount(data["train"][1],minlength=2).float()
        criterion=nn.CrossEntropyLoss(weight=counts.sum()/(2*counts))
        best=float("inf"); stale=0; history=[]
        update(state="training",gpu=torch.cuda.get_device_name(0),parameters=sum(p.numel() for p in model.parameters()))
        for epoch in range(1,args.epochs+1):
            start=time.perf_counter(); model.train(); total_loss=0
            x,y=data["train"]
            indices=torch.randperm(len(y),device="cuda")
            for part in indices.split(256):
                batch=normalize(x[part]); target=y[part]
                flip=torch.rand(len(part),device="cuda")<.5
                batch[flip]=batch[flip].flip(-1)
                gain=torch.rand((len(part),1,1,1),device="cuda")*.4+.8
                offset=(torch.rand((len(part),1,1,1),device="cuda")-.5)*.3
                batch=(batch*gain+offset).clamp(-1,1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    loss=criterion(model(batch),target)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total_loss+=float(loss.detach())*len(part)
            model.eval(); probabilities=[]; val_loss=0
            with torch.inference_mode():
                vx,vy=data["val"]
                for start_index in range(0,len(vy),512):
                    logits=model(normalize(vx[start_index:start_index+512]))
                    target=vy[start_index:start_index+512]
                    val_loss+=float(criterion(logits,target))*len(target)
                    probabilities.extend(logits.softmax(1)[:,1].cpu().tolist())
            val_loss/=len(vy)
            scores=metrics(vy.cpu().numpy(),np.array(probabilities),.5,len(layout["slots"]))
            record=dict(epoch=epoch,train_loss=total_loss/len(y),val_loss=val_loss,seconds=time.perf_counter()-start,**scores)
            history.append(record); write_status(output/"history.json",history)
            checkpoint=dict(architecture="OccupancyNet_v1",preprocess=PREPROCESS,input_size=SIZE,camera=args.camera,
                            threshold=.5,epoch=epoch,state_dict={k:v.cpu().clone() for k,v in model.state_dict().items()})
            torch.save(checkpoint,output/"last.pt")
            if val_loss<best:
                best=val_loss; stale=0; torch.save(checkpoint,output/"best.pt")
            else: stale+=1
            update(completed_epochs=epoch,epoch_seconds=record["seconds"],validation=scores)
            print(json.dumps(record),flush=True)
            if stale>=4: break
        checkpoint=torch.load(output/"best.pt",weights_only=True)
        model.load_state_dict(checkpoint["state_dict"]); model.eval()
        with torch.inference_mode():
            prob=np.concatenate([model(normalize(part)).softmax(1)[:,1].cpu().numpy() for part in data["val"][0].split(512)])
        labels=data["val"][1].cpu().numpy()
        choices={str(round(t,2)):metrics(labels,prob,t,len(layout["slots"])) for t in np.arange(.1,.91,.05)}
        threshold=min(choices,key=lambda t:(-choices[t]["balanced_accuracy"],choices[t]["occupied_count_mae"],abs(float(t)-.5)))
        checkpoint["threshold"]=float(threshold); torch.save(checkpoint,output/"best.pt")
        np.save(output/"val_probabilities.npy",prob)
        report=dict(dataset=info,architecture="OccupancyNet_v1",parameters=sum(p.numel() for p in model.parameters()),
                    best_epoch=checkpoint["epoch"],completed_epochs=len(history),threshold=float(threshold),
                    validation=choices[threshold],threshold_candidates=choices,test_set_used=False,
                    model_sha256=file_hash(output/"best.pt"),model_bytes=(output/"best.pt").stat().st_size,
                    training_settings=dict(seed=42,batch=256,optimizer="AdamW",lr=.001,weight_decay=.0001,
                                           class_weighted_loss=True,max_epochs=args.epochs,patience=4,
                                           selection="minimum validation weighted cross-entropy",
                                           augmentation="horizontal flip, brightness and contrast"))
        write_status(output/"report.json",report)
        del data,model; torch.cuda.empty_cache()
        predictor=Predictor(output/"best.pt","cpu")
        demos=[]
        for row in json.loads((output/"demo_samples.json").read_text()):
            with Image.open(Path("data/pklot_yolo_v1")/row["image"]) as img:
                image=np.asarray(img.convert("RGB"))
            results,annotated=predictor.predict(image,layout)
            filename="demo_"+row["weather"].lower()+".jpg"; annotated.save(output/filename)
            demos.append(dict(image=row["image"],preview=filename,reference_free=int(row["free"]),
                              reference_occupied=int(row["occupied"]),**results))
        write_status(output/"demo_results.json",demos)
        update(state="completed",best_epoch=checkpoint["epoch"],threshold=float(threshold),
               validation=choices[threshold],model=str(output/"best.pt"))
    except BaseException as error:
        update(state="failed",error=str(error),traceback=traceback.format_exc()); raise


if __name__=="__main__": main()
