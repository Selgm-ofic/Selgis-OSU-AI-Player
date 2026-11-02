import cv2
import numpy as np
import torch
from pathlib import Path
from .data import SequenceDataset
from .train import load_policy


def preview(session_npz, weights, output, sequence=6, fps=60):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    data=SequenceDataset([session_npz],sequence,1)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,_=load_policy(weights,device)
    writer=cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*"mp4v"),fps,(960,720))
    with torch.inference_mode():
        for x,y,valid in data:
            frame=(x[-1,0].numpy()*255).astype(np.uint8)
            shown=cv2.cvtColor(cv2.resize(frame,(960,720),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
            out=model(x.unsqueeze(0).to(device)); px,py=out["xy"][0].cpu().numpy()
            gx,gy=y[:2].numpy(); pred=(int(px*959),int(py*719)); truth=(int(gx*959),int(gy*719))
            if float(valid) > .5:
                cv2.drawMarker(shown,truth,(0,255,0),cv2.MARKER_CROSS,24,2)
            cv2.drawMarker(shown,pred,(0,0,255),cv2.MARKER_TILTED_CROSS,24,2)
            label="green=visible target  red=policy"
            cv2.putText(shown,label,(14,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            cv2.putText(shown,f"tth={float(out['tth'][0]):.0f}ms present={float(out['present'][0]):.2f}",(14,56),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            writer.write(shown)
    writer.release(); print(f"saved preview -> {output}")
