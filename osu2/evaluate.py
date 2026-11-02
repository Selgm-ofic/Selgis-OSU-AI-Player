import json
from pathlib import Path
import numpy as np
import torch
from .data import SequenceDataset
from .train import load_policy


def evaluate(dataset_root, weights, sequence=6):
    index=json.loads((Path(dataset_root)/"index.json").read_text(encoding="utf-8"))
    held_out=index["sessions"][-1]
    data=SequenceDataset([held_out["path"]],sequence,4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,_=load_policy(weights,device); errors=[]
    with torch.inference_mode():
        for x,y,valid in data:
            if valid <= .5: continue
            pred=model(x.unsqueeze(0).to(device))["xy"][0].cpu()
            errors.append(float(torch.linalg.vector_norm(pred-y[:2])))
    print(f"held_out={held_out['name']} frames={len(errors)} "
          f"mean_xy_distance={np.mean(errors):.4f} p95={np.percentile(errors,95):.4f}")
