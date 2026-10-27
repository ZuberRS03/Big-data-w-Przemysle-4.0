"""Analyze one image with an explicit fixed-camera configuration."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fixed_parking.core import load_layout
from fixed_parking.model import Predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--layout",type=Path,default=Path("fixed_parking/configs/ufpr05.json"))
    p.add_argument("--model",type=Path,default=Path("outputs/fixed_parking_ufpr05_v1/best.pt"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError("Wybierz nowy katalog wyjściowy.")
    layout=load_layout(args.layout)
    with Image.open(args.image) as source: image=np.asarray(source.convert("RGB"))
    result,annotated=Predictor(args.model,args.device).predict(image,layout)
    args.output.mkdir(parents=True)
    annotated.save(args.output/"result.jpg")
    (args.output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="slots"},indent=2,ensure_ascii=False))


if __name__=="__main__": main()
