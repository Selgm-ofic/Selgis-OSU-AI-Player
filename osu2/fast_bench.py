"""Standalone Performance Benchmark for osu! Policy Training Pipeline.
Measures throughput (samples/sec), step latency (ms/step), and epoch duration across:
  1. Data Loading: Baseline vs Fast uint8 vs In-Memory Pinned Tensor.
  2. Training Step (Stage 1): Baseline vs Optimized (No Sync Barriers + Fused AdamW).
  3. Training Step (Stage 2): Baseline (Frozen Backbone Re-eval) vs Latent Caching.
  4. End-to-End Epoch Simulation.
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from osu2.data import SequenceDataset
from osu2.model import UNetTinyOsuPolicy, PolicyLoss, spatial_softmax


def load_dataset_metadata(root_path):
    root = Path(root_path)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    paths = [x["path"] for x in index["sessions"]]
    val_candidates = [p for p in paths if any(name in p for name in ["Bass_Slut", "Embraced_by_the_Flame", "Satori_De_Pon"])]
    val_path = val_candidates[0] if val_candidates else paths[-1]
    train_paths = [p for p in paths if p != val_path]
    return train_paths, val_path


def benchmark_data_loading(train_paths, sequence=6, stride=2, batch_size=128, num_test_batches=50):
    print("=" * 70)
    print("1. DATA LOADING BENCHMARK")
    print("=" * 70)

    # 1A. Baseline SequenceDataset + DataLoader
    ds_baseline = SequenceDataset(train_paths, sequence=sequence, stride=stride)
    dl_baseline = DataLoader(ds_baseline, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    t0 = time.perf_counter()
    count = 0
    for x, y, v in dl_baseline:
        count += len(x)
        if count >= batch_size * num_test_batches:
            break
    dt_baseline = time.perf_counter() - t0
    rate_baseline = count / dt_baseline
    epoch_baseline = len(ds_baseline) / rate_baseline
    print(f"[1A] Baseline DataLoader:")
    print(f"     Throughput: {rate_baseline:8.1f} samples/s | Est. Epoch: {epoch_baseline:6.2f}s")

    # 1B. Fast uint8 SequenceDataset (no float32 on CPU)
    class FastUint8Dataset(torch.utils.data.Dataset):
        def __init__(self, paths, seq=6, strd=2):
            self.sequence, self.rows, self.arrays = seq, [], []
            for path in paths:
                z = np.load(path)
                imgs = np.ascontiguousarray(z["images"], dtype=np.uint8)
                tgts = np.ascontiguousarray(z["targets"], dtype=np.float32)
                vld = np.ascontiguousarray(z["valid"], dtype=np.float32)
                slot = len(self.arrays)
                self.arrays.append((imgs, tgts, vld))
                n_frames = len(imgs)
                ends = set(range(seq - 1, n_frames, strd))
                if len(tgts) > 0 and tgts.shape[1] >= 4:
                    for ci in np.where(tgts[:, 3] > 0.25)[0]:
                        if ci >= seq - 1:
                            ends.add(int(ci))
                    for hi in np.where(tgts[:, 4] > 0.5)[0][::2]:
                        if hi >= seq - 1:
                            ends.add(int(hi))
                self.rows += [(slot, end) for end in sorted(ends)]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            slot, end = self.rows[index]
            imgs, tgts, vld = self.arrays[slot]
            return (torch.from_numpy(imgs[end - self.sequence + 1 : end + 1]),
                    torch.from_numpy(tgts[end]),
                    torch.tensor(vld[end]))

    ds_fast = FastUint8Dataset(train_paths, sequence, stride)
    dl_fast = DataLoader(ds_fast, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    t0 = time.perf_counter()
    count = 0
    for x, y, v in dl_fast:
        count += len(x)
        if count >= batch_size * num_test_batches:
            break
    dt_fast = time.perf_counter() - t0
    rate_fast = count / dt_fast
    epoch_fast = len(ds_fast) / rate_fast
    print(f"[1B] Fast uint8 DataLoader:")
    print(f"     Throughput: {rate_fast:8.1f} samples/s | Est. Epoch: {epoch_fast:6.2f}s | Speedup: {rate_fast/rate_baseline:.2f}x")

    # 1C. In-Memory Pinned Tensor Dataset
    t_build0 = time.perf_counter()
    all_x, all_y, all_v = [], [], []
    for path in train_paths:
        z = np.load(path)
        imgs, tgts, vld = z["images"], z["targets"], z["valid"]
        n_frames = len(imgs)
        ends = set(range(sequence - 1, n_frames, stride))
        if len(tgts) > 0 and tgts.shape[1] >= 4:
            for ci in np.where(tgts[:, 3] > 0.25)[0]:
                if ci >= sequence - 1:
                    ends.add(int(ci))
            for hi in np.where(tgts[:, 4] > 0.5)[0][::2]:
                if hi >= sequence - 1:
                    ends.add(int(hi))
        ends = np.array(sorted(ends), dtype=np.int32)
        offsets = np.arange(-sequence + 1, 1, dtype=np.int32)
        idx_mat = ends[:, None] + offsets
        all_x.append(imgs[idx_mat])
        all_y.append(tgts[ends].astype(np.float32))
        all_v.append(vld[ends].astype(np.float32))

    X_pin = torch.from_numpy(np.concatenate(all_x, axis=0)).pin_memory()
    Y_pin = torch.from_numpy(np.concatenate(all_y, axis=0)).pin_memory()
    V_pin = torch.from_numpy(np.concatenate(all_v, axis=0)).pin_memory()
    t_build = time.perf_counter() - t_build0
    N = len(X_pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perm = torch.randperm(N)

    t0 = time.perf_counter()
    for b in range(num_test_batches):
        idx = perm[b * batch_size : (b + 1) * batch_size]
        bx = X_pin[idx].to(device, non_blocking=True).to(dtype=torch.float16).div_(255.0)
        by = Y_pin[idx].to(device, non_blocking=True)
        bv = V_pin[idx].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_pin = time.perf_counter() - t0
    count_pin = batch_size * num_test_batches
    rate_pin = count_pin / dt_pin
    epoch_pin = N / rate_pin
    print(f"[1C] In-Memory Pinned Tensor (with CUDA transfer + GPU scaling):")
    print(f"     Throughput: {rate_pin:8.1f} samples/s | Est. Epoch: {epoch_pin:6.2f}s | Speedup: {rate_pin/rate_baseline:.2f}x")
    print(f"     (One-time memory allocation: {X_pin.nbytes / 1e6:.1f} MB in {t_build:.2f}s)")

    return {
        "baseline_rate": rate_baseline,
        "baseline_epoch": epoch_baseline,
        "fast_rate": rate_fast,
        "fast_epoch": epoch_fast,
        "pinned_rate": rate_pin,
        "pinned_epoch": epoch_pin,
        "total_samples": N,
        "X_pin": X_pin,
        "Y_pin": Y_pin,
        "V_pin": V_pin
    }


def benchmark_stage1(device, batch_size=128, iters=40):
    print("\n" + "=" * 70)
    print("2. STAGE 1 TRAINING STEP BENCHMARK (Backbone Active, Lambda=0.0)")
    print("=" * 70)

    model = UNetTinyOsuPolicy(hidden=160, spatial_channels=80, action_fusion="tc_roi", tth_scale_ms=500.0).to(device)
    criterion = PolicyLoss(tth_scale_ms=500.0, pos_weight=2.0, weighting="decoupled").to(device)

    # 2A. Baseline Training Step (Sync Barriers + Standard AdamW)
    for name, p in model.named_parameters():
        if name.startswith("action.") or name.startswith("roi_encoder."):
            p.requires_grad = False
        else:
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad] + list(criterion.parameters())
    opt_base = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scaler_base = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=2048.0)

    x = torch.randn(batch_size, 6, 2, 72, 96, device=device)
    y = torch.randn(batch_size, 6, device=device)
    v = torch.ones(batch_size, device=device)

    # Warmup
    for _ in range(3):
        opt_base.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x, roi_jitter=0.0, roi_gt=y[:, :2], roi_beta=0.0)
            loss, _ = criterion(pred, y, v, lambda_action=0.0)
        scaler_base.scale(loss).backward()
        scaler_base.unscale_(opt_base)
        active_params = [p for pg in opt_base.param_groups for p in pg["params"] if p.requires_grad]
        valid_grads = all(torch.isfinite(p.grad).all() for p in active_params if p.grad is not None)
        if valid_grads:
            gn = float(torch.nn.utils.clip_grad_norm_(active_params, 5.0))
        scaler_base.step(opt_base)
        scaler_base.update()
        lv = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt_base.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x, roi_jitter=0.0, roi_gt=y[:, :2], roi_beta=0.0)
            loss, _ = criterion(pred, y, v, lambda_action=0.0)
        scaler_base.scale(loss).backward()
        scaler_base.unscale_(opt_base)
        active_params = [p for pg in opt_base.param_groups for p in pg["params"] if p.requires_grad]
        valid_grads = all(torch.isfinite(p.grad).all() for p in active_params if p.grad is not None)
        if valid_grads:
            gn = float(torch.nn.utils.clip_grad_norm_(active_params, 5.0))
        scaler_base.step(opt_base)
        scaler_base.update()
        lv = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_base = time.perf_counter() - t0
    rate_base = (iters * batch_size) / dt_base
    ms_base = (dt_base / iters) * 1000.0
    print(f"[2A] Baseline Stage 1 (Full Sync Barriers + Standard AdamW):")
    print(f"     Throughput: {rate_base:8.1f} samples/s | Step Latency: {ms_base:6.2f} ms")

    # 2B. Optimized Training Step (Zero Sync Barriers + Fused AdamW)
    opt_opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4, fused=True)
    scaler_opt = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=2048.0)

    # Warmup
    for _ in range(3):
        opt_opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x, roi_jitter=0.0, roi_gt=y[:, :2], roi_beta=0.0)
            loss, _ = criterion(pred, y, v, lambda_action=0.0)
        scaler_opt.scale(loss).backward()
        scaler_opt.unscale_(opt_opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler_opt.step(opt_opt)
        scaler_opt.update()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_loss_tensor = torch.zeros((), device=device)
    for step in range(1, iters + 1):
        opt_opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x, roi_jitter=0.0, roi_gt=y[:, :2], roi_beta=0.0)
            loss, _ = criterion(pred, y, v, lambda_action=0.0)
        scaler_opt.scale(loss).backward()
        scaler_opt.unscale_(opt_opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler_opt.step(opt_opt)
        scaler_opt.update()
        total_loss_tensor += loss.detach()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_opt = time.perf_counter() - t0
    rate_opt = (iters * batch_size) / dt_opt
    ms_opt = (dt_opt / iters) * 1000.0
    print(f"[2B] Optimized Stage 1 (Zero CPU-GPU Sync + Fused AdamW):")
    print(f"     Throughput: {rate_opt:8.1f} samples/s | Step Latency: {ms_opt:6.2f} ms | Speedup: {rate_opt/rate_base:.2f}x")

    return {
        "stage1_baseline_rate": rate_base,
        "stage1_baseline_ms": ms_base,
        "stage1_opt_rate": rate_opt,
        "stage1_opt_ms": ms_opt,
    }


def benchmark_stage2(device, batch_size=128, iters=80):
    print("\n" + "=" * 70)
    print("3. STAGE 2 TRAINING STEP BENCHMARK (Action Heads Only, Lambda=1.0)")
    print("=" * 70)

    model = UNetTinyOsuPolicy(hidden=160, spatial_channels=80, action_fusion="tc_roi", tth_scale_ms=500.0).to(device)
    criterion = PolicyLoss(tth_scale_ms=500.0, pos_weight=2.0, weighting="decoupled").to(device)

    # Freeze backbone
    for name, p in model.named_parameters():
        if name.startswith("action.") or name.startswith("roi_encoder."):
            p.requires_grad = True
        else:
            p.requires_grad = False

    action_params = [p for p in model.parameters() if p.requires_grad] + list(criterion.parameters())
    opt_base = torch.optim.AdamW(action_params, lr=3e-4, weight_decay=1e-4)
    scaler_base = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=2048.0)

    x = torch.randn(batch_size, 6, 2, 72, 96, device=device)
    y = torch.randn(batch_size, 6, device=device)
    v = torch.ones(batch_size, device=device)

    # 3A. Baseline Stage 2: Re-computing full frozen backbone every step
    for _ in range(3):
        opt_base.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x, roi_jitter=0.08, roi_gt=y[:, :2], roi_beta=0.5)
            loss, _ = criterion(pred, y, v, lambda_action=1.0)
        scaler_base.scale(loss).backward()
        scaler_base.unscale_(opt_base)
        active_params = [p for pg in opt_base.param_groups for p in pg["params"] if p.requires_grad]
        valid_grads = all(torch.isfinite(p.grad).all() for p in active_params if p.grad is not None)
        if valid_grads:
            gn = float(torch.nn.utils.clip_grad_norm_(active_params, 5.0))
        scaler_base.step(opt_base)
        scaler_base.update()
        lv = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt_base.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x, roi_jitter=0.08, roi_gt=y[:, :2], roi_beta=0.5)
            loss, _ = criterion(pred, y, v, lambda_action=1.0)
        scaler_base.scale(loss).backward()
        scaler_base.unscale_(opt_base)
        active_params = [p for pg in opt_base.param_groups for p in pg["params"] if p.requires_grad]
        valid_grads = all(torch.isfinite(p.grad).all() for p in active_params if p.grad is not None)
        if valid_grads:
            gn = float(torch.nn.utils.clip_grad_norm_(active_params, 5.0))
        scaler_base.step(opt_base)
        scaler_base.update()
        lv = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_base = time.perf_counter() - t0
    rate_base = (iters * batch_size) / dt_base
    ms_base = (dt_base / iters) * 1000.0
    print(f"[3A] Baseline Stage 2 (Re-evaluating Full Frozen Backbone):")
    print(f"     Throughput: {rate_base:8.1f} samples/s | Step Latency: {ms_base:6.2f} ms")

    # 3B. Latent Caching Stage 2: Pre-extracting backbone features once
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        e1 = model.vision["enc1"](x_flat)
        e2 = model.vision["enc2"](e1)
        spatial = model.vision["enc3"](e2)
        _, C, Hs, Ws = spatial.shape
        enc = model.embed(spatial).reshape(b, t, -1)
        seq, _ = model.gru(enc)
        cached_last = model.gru_norm(seq[:, -1])
        last_map = spatial.reshape(b, t, C, Hs, Ws)[:, -1]
        ctx = model.context(cached_last)[:, :, None, None].expand(-1, -1, Hs, Ws)
        up = model.heatmap["up"](torch.cat((last_map, ctx), 1))
        _, C2, H2, W2 = e2.shape
        cached_skip = e2.reshape(b, t, C2, H2, W2)[:, -1]
        if up.shape[-2:] != cached_skip.shape[-2:]:
            up = F.interpolate(up, size=cached_skip.shape[-2:], mode="bilinear", align_corners=False)
        hm = model.heatmap["fuse"](torch.cat((up, cached_skip), 1))
        hm = F.interpolate(hm, size=(model.hm_h, model.hm_w), mode="bilinear", align_corners=False)
        cached_logits = hm.flatten(1)
        cached_xy, _ = spatial_softmax(cached_logits, model.grid_x, model.grid_y, hm_h=model.hm_h, hm_w=model.hm_w, temperature=0.5)
        cached_x_recent = x[:, -3:].reshape(b, -1, h, w)

    opt_opt = torch.optim.AdamW(action_params, lr=3e-4, weight_decay=1e-4, fused=True)
    scaler_opt = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=2048.0)

    # Warmup
    for _ in range(5):
        opt_opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            center = (1.0 - 0.5) * y[:, :2] + 0.5 * cached_xy
            c_norm = center * 2.0 - 1.0
            noise = (torch.randn_like(c_norm) * 0.08).clamp(-0.14, 0.14)
            c_norm = (c_norm + noise).clamp(-1.0, 1.0)
            grid = (model.base_patch_grid * 0.28).unsqueeze(0) + c_norm[:, None, None, :]
            grid = grid.clamp(-1.0, 1.0)
            feat_skip = F.grid_sample(cached_skip, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            feat_raw = F.grid_sample(cached_x_recent, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            patch = torch.cat([feat_skip, feat_raw], dim=1)
            local_feat = model.roi_encoder(patch)
            action_in = torch.cat([cached_last, local_feat], dim=1)
            a = model.action(action_in)
            pred = {
                "heatmap": cached_logits, "xy": cached_xy, "tth": torch.tanh(a[:, 0]) * model.tth_scale_ms,
                "click": torch.sigmoid(a[:, 1]), "hold": torch.sigmoid(a[:, 2]),
                "present": torch.sigmoid(a[:, 3]),
                "click_logit": a[:, 1], "hold_logit": a[:, 2], "present_logit": a[:, 3],
                "last": cached_last
            }
            loss, _ = criterion(pred, y, v, lambda_action=1.0)
        scaler_opt.scale(loss).backward()
        scaler_opt.unscale_(opt_opt)
        torch.nn.utils.clip_grad_norm_(action_params, 5.0)
        scaler_opt.step(opt_opt)
        scaler_opt.update()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt_opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            center = (1.0 - 0.5) * y[:, :2] + 0.5 * cached_xy
            c_norm = center * 2.0 - 1.0
            noise = (torch.randn_like(c_norm) * 0.08).clamp(-0.14, 0.14)
            c_norm = (c_norm + noise).clamp(-1.0, 1.0)
            grid = (model.base_patch_grid * 0.28).unsqueeze(0) + c_norm[:, None, None, :]
            grid = grid.clamp(-1.0, 1.0)
            feat_skip = F.grid_sample(cached_skip, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            feat_raw = F.grid_sample(cached_x_recent, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            patch = torch.cat([feat_skip, feat_raw], dim=1)
            local_feat = model.roi_encoder(patch)
            action_in = torch.cat([cached_last, local_feat], dim=1)
            a = model.action(action_in)
            pred = {
                "heatmap": cached_logits, "xy": cached_xy, "tth": torch.tanh(a[:, 0]) * model.tth_scale_ms,
                "click": torch.sigmoid(a[:, 1]), "hold": torch.sigmoid(a[:, 2]),
                "present": torch.sigmoid(a[:, 3]),
                "click_logit": a[:, 1], "hold_logit": a[:, 2], "present_logit": a[:, 3],
                "last": cached_last
            }
            loss, _ = criterion(pred, y, v, lambda_action=1.0)
        scaler_opt.scale(loss).backward()
        scaler_opt.unscale_(opt_opt)
        torch.nn.utils.clip_grad_norm_(action_params, 5.0)
        scaler_opt.step(opt_opt)
        scaler_opt.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_opt = time.perf_counter() - t0
    rate_opt = (iters * batch_size) / dt_opt
    ms_opt = (dt_opt / iters) * 1000.0
    print(f"[3B] Stage 2 with Latent Caching (Zero Backbone Forward/Backward):")
    print(f"     Throughput: {rate_opt:8.1f} samples/s | Step Latency: {ms_opt:6.2f} ms | Speedup: {rate_opt/rate_base:.2f}x")

    return {
        "stage2_baseline_rate": rate_base,
        "stage2_baseline_ms": ms_base,
        "stage2_opt_rate": rate_opt,
        "stage2_opt_ms": ms_opt,
    }


def print_summary_comparison(dl_res, s1_res, s2_res):
    N = dl_res["total_samples"]

    # In baseline: DataLoader (43.6s) + GPU step run sequentially on main thread.
    t_base_s1_step = N / s1_res["stage1_baseline_rate"]
    t_base_s1_total = dl_res["baseline_epoch"] + t_base_s1_step

    t_base_s2_step = N / s2_res["stage2_baseline_rate"]
    t_base_s2_total = dl_res["baseline_epoch"] + t_base_s2_step

    # In optimized Stage 1: Pinned DataLoader transfer + optimized step
    t_opt_s1_step = N / s1_res["stage1_opt_rate"]
    t_opt_s1_total = dl_res["pinned_epoch"] + t_opt_s1_step

    # In optimized Stage 2: Latent cached streaming + micro-step
    t_opt_s2_total = N / s2_res["stage2_opt_rate"]

    print("\n" + "=" * 70)
    print("4. COMPREHENSIVE END-TO-END EPOCH DURATION COMPARISON")
    print("=" * 70)
    print(f"Dataset Size: {N:,} sequences | Batch Size: 128 ({int(np.ceil(N/128))} batches/epoch)")
    print("-" * 70)
    print(f"{'Component / Stage':<35} | {'Baseline':<14} | {'Optimized':<14} | {'Speedup':<8}")
    print("-" * 70)
    print(f"{'Data Loading & Transfer':<35} | {dl_res['baseline_epoch']:6.2f} s/ep   | {dl_res['pinned_epoch']:6.2f} s/ep   | {dl_res['baseline_epoch']/dl_res['pinned_epoch']:5.1f}x")
    print(f"{'Stage 1 (Aim Backbone Active)':<35} | {t_base_s1_total:6.2f} s/ep   | {t_opt_s1_total:6.2f} s/ep   | {t_base_s1_total/t_opt_s1_total:5.1f}x")
    print(f"{'Stage 2 (Action Heads Only)':<35} | {t_base_s2_total:6.2f} s/ep   | {t_opt_s2_total:6.2f} s/ep   | {t_base_s2_total/t_opt_s2_total:5.1f}x")
    print("-" * 70)

    # 40 epochs total (20 Stage 1 + 20 Stage 2)
    ep1, ep2 = 20, 20
    base_full_time = ep1 * t_base_s1_total + ep2 * t_base_s2_total
    opt_full_time = ep1 * t_opt_s1_total + ep2 * t_opt_s2_total + 25.0  # +25s one-time latent extraction
    print(f"{'Full 40-Epoch Training Job':<35} | {base_full_time/60:6.1f} min     | {opt_full_time/60:6.1f} min     | {base_full_time/opt_full_time:5.1f}x")
    print("=" * 70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    train_paths, val_path = load_dataset_metadata(r"C:\Selgis OSU AI Player\data\dataset_v8")
    print(f"Initialized benchmark on device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Training sessions: {len(train_paths)} | Validation session: {Path(val_path).name}")

    dl_res = benchmark_data_loading(train_paths, sequence=6, stride=2, batch_size=128, num_test_batches=50)
    s1_res = benchmark_stage1(device, batch_size=128, iters=40)
    s2_res = benchmark_stage2(device, batch_size=128, iters=80)
    print_summary_comparison(dl_res, s1_res, s2_res)


if __name__ == "__main__":
    main()
