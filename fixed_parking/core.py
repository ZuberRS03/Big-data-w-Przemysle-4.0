"""Shared, label-free geometry and preprocessing for training and inference."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 64
PREPROCESS = "masked_polygon_letterbox_rgb64_gray127_v1"


def validate_layout(layout):
    if layout.get("schema") != 1 or not isinstance(layout.get("slots"), list) or not 1 <= len(layout["slots"]) <= 300:
        raise ValueError("Konfiguracja musi zawierać od 1 do 300 miejsc (schema=1).")
    width, height = layout.get("reference_size", [0, 0])
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Nieprawidłowy rozmiar obrazu referencyjnego.")
    ids=set()
    for slot in layout["slots"]:
        identifier=slot.get("id")
        if not isinstance(identifier,str) or not identifier.strip() or len(identifier)>30 or identifier in ids:
            raise ValueError("Miejsca muszą mieć unikalne identyfikatory tekstowe.")
        ids.add(identifier)
        points=np.asarray(slot.get("polygon"),dtype=np.float32)
        if points.ndim!=2 or points.shape[1]!=2 or not 3<=len(points)<=12 or not np.isfinite(points).all() or np.any(points<0) or np.any(points>1):
            raise ValueError("Wierzchołki muszą być znormalizowane do zakresu 0–1.")
        if not cv2.isContourConvex(points) or cv2.contourArea(points)<1e-6:
            raise ValueError("Miejsce musi być wypukłym, nieprzecinającym się wielokątem.")
    return layout


def load_layout(path):
    return validate_layout(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def crops(image, layout):
    validate_layout(layout)
    image=np.asarray(image,dtype=np.uint8)
    if image.ndim!=3 or image.shape[2]!=3:
        raise ValueError("Wymagany jest obraz RGB.")
    h,w=image.shape[:2]
    rw,rh=layout["reference_size"]
    if abs((w/h)/(rw/rh)-1)>.01:
        raise ValueError("Proporcje zdjęcia nie pasują do konfiguracji. Wczytaj konfigurację właściwej kamery.")
    result=[]
    for slot in layout["slots"]:
        points=np.asarray(slot["polygon"])*[w,h]
        x1,y1=np.floor(points.min(axis=0)).astype(int)
        x2,y2=np.ceil(points.max(axis=0)).astype(int)
        x2,y2=min(w,x2),min(h,y2)
        if x2<=x1 or y2<=y1:
            raise ValueError("Miejsce jest zbyt małe w tym obrazie.")
        patch=image[y1:y2,x1:x2].copy()
        mask=np.zeros(patch.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.rint(points-[x1,y1]).astype(np.int32)],255)
        patch[mask==0]=127
        ratio=min(SIZE/patch.shape[1],SIZE/patch.shape[0])
        nw,nh=max(1,round(patch.shape[1]*ratio)),max(1,round(patch.shape[0]*ratio))
        resized=cv2.resize(patch,(nw,nh),interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
        tile=np.full((SIZE,SIZE,3),127,np.uint8)
        ox,oy=(SIZE-nw)//2,(SIZE-nh)//2
        tile[oy:oy+nh,ox:ox+nw]=resized
        result.append(tile)
    return np.stack(result)


def annotate(image, layout, records):
    result=Image.fromarray(np.asarray(image,dtype=np.uint8)).convert("RGB")
    draw=ImageDraw.Draw(result)
    font=ImageFont.load_default(size=max(12,round(result.width/90)))
    for slot,record in zip(layout["slots"],records,strict=True):
        points=[(x*result.width,y*result.height) for x,y in slot["polygon"]]
        color="#ff5353" if record["occupied"] else "#00db88"
        draw.line(points+[points[0]],fill=color,width=3)
        x,y=points[0]
        text=slot["id"]
        box=draw.textbbox((x,y),text,font=font)
        draw.rectangle(box,fill="#122538")
        draw.text((x,y),text,fill=color,font=font)
    return result
