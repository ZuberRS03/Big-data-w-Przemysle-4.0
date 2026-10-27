"""Small offline CNN; no network downloads or pretrained weights required."""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fixed_parking.core import PREPROCESS, SIZE, annotate, crops


class OccupancyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in ((3,16),(16,32),(32,64)):
            layers.extend([nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2)])
        self.features=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(1024,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,2))

    def forward(self,x):
        return self.head(self.features(x))


def normalize(x):
    return x.float().permute(0,3,1,2).contiguous()/127.5-1.0


class Predictor:
    def __init__(self, weights, device="cpu"):
        self.device=torch.device(device)
        checkpoint=torch.load(Path(weights),map_location="cpu",weights_only=True)
        if checkpoint["architecture"]!="OccupancyNet_v1" or checkpoint["preprocess"]!=PREPROCESS or checkpoint["input_size"]!=SIZE:
            raise ValueError("Niezgodny format modelu.")
        self.model=OccupancyNet().to(self.device).eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.threshold=checkpoint["threshold"]
        self.camera=checkpoint["camera"]
        torch.set_num_threads(4)

    @torch.inference_mode()
    def predict(self,image,layout):
        if self.device.type=="cuda": torch.cuda.synchronize()
        start=time.perf_counter()
        batch=crops(image,layout)
        scores=[]
        for part in np.array_split(batch,max(1,(len(batch)+127)//128)):
            p=self.model(normalize(torch.from_numpy(part).to(self.device))).softmax(1)[:,1]
            scores.extend(p.cpu().tolist())
        records=[dict(id=slot["id"],occupied=p>=self.threshold,occupied_probability=p,
                      confidence=p if p>=self.threshold else 1-p) for slot,p in zip(layout["slots"],scores)]
        if self.device.type=="cuda": torch.cuda.synchronize()
        ms=(time.perf_counter()-start)*1000
        occupied=sum(r["occupied"] for r in records)
        return dict(free=len(records)-occupied,occupied=occupied,total=len(records),slots=records,
                    threshold=self.threshold,analysis_ms=ms,trained_camera=self.camera),annotate(image,layout,records)
