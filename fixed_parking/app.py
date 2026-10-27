"""Local UI: configure fixed parking spaces and classify their occupancy."""
import argparse
import base64
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np
from PIL import Image

from fixed_parking.core import load_layout, validate_layout
from fixed_parking.model import Predictor


def encoded_image(image):
    buffer=io.BytesIO(); image.save(buffer,format="JPEG",quality=92)
    return "data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("outputs/fixed_parking_ufpr05_v1"))
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    args=parser.parse_args()
    run=args.run.resolve()
    predictor=Predictor(run/"best.pt",args.device)
    layout=load_layout(run/"layout.json")
    samples=json.loads((run/"demo_samples.json").read_text())
    report=json.loads((run/"report.json").read_text())
    root=Path(__file__).resolve().parent
    data=Path("data/pklot_yolo_v1").resolve()
    with Image.open(data/samples[0]["image"]) as image:
        predictor.predict(np.asarray(image.convert("RGB")),layout)

    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value,content_type="application/json; charset=utf-8"):
            body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff"); self.end_headers(); self.wfile.write(body)

        def allowed(self):
            hosts={f"127.0.0.1:{args.port}",f"localhost:{args.port}"}
            return self.headers.get("Host") in hosts and self.headers.get("Origin") in {None,*["http://"+h for h in hosts]}

        def do_GET(self):
            if not self.allowed(): return self.send(403,{"error":"Dostęp tylko lokalnie."})
            request=urlparse(self.path)
            if request.path=="/": return self.send(200,(root/"web.html").read_bytes(),"text/html; charset=utf-8")
            if request.path=="/api/meta":
                return self.send(200,dict(camera=predictor.camera,device=args.device,validation=report["validation"],
                     samples=[dict(index=i,weather=r["weather"],date=r["date"]) for i,r in enumerate(samples)]))
            if request.path=="/api/sample":
                try:
                    index=int(parse_qs(request.query).get("index",["0"])[0])
                    if not 0<=index<len(samples): raise ValueError("Nieprawidłowy przykład")
                    row=samples[index]
                    image_data="data:image/jpeg;base64,"+base64.b64encode((data/row["image"]).read_bytes()).decode("ascii")
                    return self.send(200,dict(image=image_data,layout=layout,
                                             reference=dict(free=int(row["free"]),occupied=int(row["occupied"]))))
                except (ValueError,IndexError) as error: return self.send(400,{"error":str(error)})
            self.send(404,{"error":"Nie znaleziono"})

        def do_POST(self):
            if not self.allowed(): return self.send(403,{"error":"Dostęp tylko lokalnie."})
            if self.path!="/api/predict": return self.send(404,{"error":"Nie znaleziono"})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<24_000_000: raise ValueError("Wczytaj mniejsze zdjęcie (maksymalnie około 15 MB).")
                payload=json.loads(self.rfile.read(length))
                current=validate_layout(payload["layout"])
                raw=base64.b64decode(payload["image"].split(",",1)[-1],validate=True)
                with Image.open(io.BytesIO(raw)) as source:
                    if source.width*source.height>25_000_000: raise ValueError("Zdjęcie ma zbyt dużą rozdzielczość.")
                    image=np.asarray(source.convert("RGB"))
                result,annotated=predictor.predict(image,current)
                folder=run/"analyses"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
                folder.mkdir(parents=True)
                annotated.save(folder/"result.jpg")
                (folder/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
                (folder/"layout.json").write_text(json.dumps(current,indent=2,ensure_ascii=False),encoding="utf-8")
                self.send(200,dict(**result,preview=encoded_image(annotated),saved_analysis=folder.name))
            except (ValueError,KeyError,TypeError,OSError) as error:
                self.send(400,{"error":str(error)})
            except Exception as error:
                print(f"Analysis error: {error}",flush=True)
                self.send(500,{"error":"Analiza nie powiodła się. Szczegóły zapisano w logu aplikacji."})

    server=HTTPServer(("127.0.0.1",args.port),Handler)
    print(f"Fixed parking UI: http://127.0.0.1:{args.port} | {args.device} | {predictor.camera}",flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__=="__main__": main()
