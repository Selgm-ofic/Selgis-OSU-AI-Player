"""Fine-tune an existing policy's aim without changing its action heads."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .fast_train import InMemoryPinnedDataset, load_policy
from .model import target_heatmap


def aim_metrics(model, dataset, device, batch_size):
    model.eval()
    distances = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            x = dataset.batch_images(slice(start, end)).to(device).float().div_(255)
            y = dataset.Y[start:end].to(device)
            valid = dataset.V[start:end].to(device) > 0.5
            if valid.any():
                pred = model(x, compute_action=False)["xy"]
                distances.append(torch.linalg.vector_norm(pred[valid].float() - y[valid, :2], dim=1).cpu())
    d = torch.cat(distances).numpy()
    return {"count": int(len(d)), "mean": float(d.mean()), "p95": float(np.percentile(d, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="data/models/unet_policy_v7_2_best_xy.pt")
    parser.add_argument("--dataset", default="data/dataset_v8")
    parser.add_argument("--out", default="data/models/unet_policy_v7_2_heatmap_ft.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    model, source = load_policy(args.weights, device)
    if not isinstance(source, dict) or source.get("arch") != "unet":
        raise ValueError("Aim fine-tuning requires a UNet checkpoint")
    # Safe aim-only adaptation: the shared encoder/GRU feeds timing and click
    # heads in the live policy.  Changing it while freezing those heads causes
    # representation drift and can make live play worse.  Only the heatmap
    # decoder is allowed to move here.
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("heatmap."))

    index = json.loads((Path(args.dataset) / "index.json").read_text(encoding="utf-8"))
    sessions = index["sessions"]
    holdout_names = ("Bass_Slut", "Embraced_by_the_Flame")
    holdouts = {s["name"]: InMemoryPinnedDataset([s["path"]], stride=4) for s in sessions if any(n in s["name"] for n in holdout_names)}
    if len(holdouts) != 2:
        raise RuntimeError("Expected both held-out sessions in the dataset index")
    train_paths = [s["path"] for s in sessions if s["name"] not in holdouts]
    train = InMemoryPinnedDataset(train_paths, stride=args.stride)
    print(f"device={device}, train_samples={len(train)}, held_out={list(holdouts)}", flush=True)

    def evaluate():
        metrics = {name: aim_metrics(model, ds, device, args.batch) for name, ds in holdouts.items()}
        score = float(np.mean([m["p95"] for m in metrics.values()]))
        return score, metrics

    baseline, baseline_metrics = evaluate()
    print(f"baseline p95={baseline:.5f}: {json.dumps(baseline_metrics)}", flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = baseline
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train))
        running = 0.0
        for indices in order.split(args.batch):
            x = train.batch_images(indices).to(device).float().div_(255)
            y = train.Y[indices].to(device)
            valid = train.V[indices].to(device) > 0.5
            if not bool(valid.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x, compute_action=False)
                heat = target_heatmap(y[valid, :2])
                hm = -(heat * F.log_softmax(out["heatmap"][valid].float(), dim=1)).sum(dim=1).mean()
                xy = F.smooth_l1_loss(out["xy"][valid].float(), y[valid, :2])
                loss = 0.3 * hm + 30.0 * xy
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(indices)
        scheduler.step()
        score, metrics = evaluate()
        row = {"epoch": epoch, "train_loss": running / len(train), "p95_score": score, "held_out": metrics, "seconds": round(time.perf_counter() - started, 2)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score:
            best_score = score
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = dict(source)
            checkpoint.update(state=model.state_dict(), version="v7.2-aim-ft", tag="best_heldout_aim", source_checkpoint=str(Path(args.weights).resolve()), source_dataset=str(Path(args.dataset).resolve()), aim_metrics=metrics)
            torch.save(checkpoint, out_path)
            print(f"saved {out_path}", flush=True)
    report = {"source": args.weights, "baseline": baseline_metrics, "baseline_p95_score": baseline, "best_p95_score": best_score, "history": history}
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"finished: baseline={baseline:.5f}, best={best_score:.5f}", flush=True)


if __name__ == "__main__":
    main()
