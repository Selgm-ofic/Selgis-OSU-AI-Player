"""Train PolicyGRU v3 (transformer over tracks) from actionlabel sessions."""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from policy.model import CURSOR_SPEED_SCALE, PolicyGRU, TTH_SCALE_MS
from policy.features import FEATURE_DIM, FEATURE_DIM_V3


def upgrade_v2_features(features):
    """Losslessly lift legacy track state into the v3 object layout.

    Width/height/confidence were not stored by v2, so conservative constants
    are supplied. Cursor y-velocity is recomputed later by SequenceDataset.
    """
    if features.shape[1] == FEATURE_DIM_V3:
        return features
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"unsupported feature dim {features.shape[1]}")
    out = np.zeros((len(features), FEATURE_DIM_V3), dtype=np.float32)
    for track_index in range(8):
        src = features[:, track_index * 6:(track_index + 1) * 6]
        dst = out[:, track_index * 14:(track_index + 1) * 14]
        present = np.abs(src[:, :5]).sum(axis=1) > 0
        dst[:, :4] = src[:, :4]
        dst[present, 4] = 0.08
        dst[present, 5] = 0.08
        dst[:, 6] = src[:, 4]
        dst[present, 7] = 0.90
        class_id = np.clip(np.rint(src[:, 5] * 5.0).astype(int), 0, 5)
        rows = np.flatnonzero(present)
        dst[rows, 8 + class_id[rows]] = 1.0
    out[:, -5] = features[:, -4]
    out[:, -4] = features[:, -3]
    out[:, -3] = features[:, -2]
    out[:, -2] = 0.0
    out[:, -1] = features[:, -1]
    return out


class SequenceDataset(Dataset):
    def __init__(self, sessions, seq_len=24, stride=2, training=True,
                 cursor_noise=0.08, seed=7):
        self.seq_len = seq_len
        self.training = training
        self.cursor_noise = float(cursor_noise)
        self.seed = int(seed)
        self.samples = []
        self.data = []
        for feature_path, target_path in sessions:
            features = upgrade_v2_features(
                np.load(feature_path).astype(np.float32))
            with np.load(target_path) as archive:
                targets = archive["targets"].astype(np.float32)
            n = min(len(features), len(targets))
            if n < seq_len:
                continue
            slot = len(self.data)
            self.data.append((features[:n], targets[:n]))
            self.samples.extend((slot, end) for end in range(seq_len, n + 1, stride))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        slot, end = self.samples[index]
        features, targets = self.data[slot]
        start = end - self.seq_len
        current_index = end - 1
        future_index = min(current_index + 1, len(targets) - 1)
        row = targets[current_index]
        sequence = features[start:end].copy()

        # Replace the teacher-forced cursor tail with a plausible off-trajectory
        # cursor history.  The policy must learn to correct this state instead
        # of copying the answer already present in the input.
        ideal = targets[start:end, 2:4].copy()
        rng = (np.random.default_rng() if self.training else
               np.random.default_rng(self.seed + index))
        end_error = np.clip(rng.normal(0.0, self.cursor_noise, 2), -0.18, 0.18)
        start_error = np.clip(end_error + rng.normal(0.0, 0.025, 2), -0.20, 0.20)
        alpha = np.linspace(0.0, 1.0, len(ideal), dtype=np.float32)[:, None]
        noisy_cursor = np.clip(ideal + start_error * (1.0 - alpha) + end_error * alpha,
                               0.0, 1.0)
        dt = np.diff(targets[start:end, 0], prepend=targets[start, 0]) / 1000.0
        if len(dt) > 1:
            dt[0] = dt[1]
        dt = np.maximum(dt, 0.001)
        velocity = np.zeros_like(noisy_cursor)
        if len(noisy_cursor) > 1:
            velocity[1:] = np.diff(noisy_cursor, axis=0) / dt[1:, None]
            velocity[0] = velocity[1]
        # v3 tail: [cursor_x, cursor_y, cursor_vx*0.1, cursor_vy*0.1, clicking]
        sequence[:, -5:-3] = noisy_cursor
        sequence[:, -3:-1] = velocity * 0.1

        desired_next = targets[future_index, 2:4]
        step_dt = max(0.001, (targets[future_index, 0] - targets[current_index, 0]) /
                      1000.0) if future_index != current_index else max(0.001, dt[-1])
        velocity_target = np.clip(
            (desired_next - noisy_cursor[-1]) / step_dt / CURSOR_SPEED_SCALE,
            -1.0, 1.0,
        )
        track_rows = sequence[-1, :8 * 14].reshape(8, 14)
        present = np.abs(track_rows).sum(axis=1) > 0
        target_slot = -1
        if np.any(present):
            if row[5] >= 0.5:
                # Active slider/spinner: follow the current path position.
                goal_xy = desired_next
            else:
                # Before a hit, the owner remains stable while the ideal cursor
                # approaches it. Pick the sample closest to its hit time.
                owner = int(row[6])
                limit = min(len(targets), current_index + 90)
                candidates = np.arange(current_index, limit)
                candidates = candidates[
                    (targets[candidates, 6].astype(int) == owner) &
                    (targets[candidates, 4] >= 0.0)
                ]
                goal_index = (int(candidates[np.argmin(targets[candidates, 4])])
                              if len(candidates) else future_index)
                goal_xy = targets[goal_index, 2:4]
            distances = np.linalg.norm(track_rows[:, :2] - goal_xy, axis=1)
            distances[~present] = np.inf
            candidate = int(np.argmin(distances))
            if distances[candidate] <= 0.25:
                target_slot = candidate
        target = np.asarray([
            velocity_target[0], velocity_target[1],
            np.clip(row[4] / TTH_SCALE_MS, -1.0, 1.0), row[5], target_slot,
        ], dtype=np.float32)
        return sequence, target


def discover_sessions(root):
    found = []
    pattern = os.path.join(root, "**", "policy_features.npy")
    for feature_path in glob.glob(pattern, recursive=True):
        folder = os.path.dirname(feature_path)
        target_path = os.path.join(folder, "policy_targets.npz")
        manifest_path = os.path.join(folder, "policy_manifest.json")
        if not os.path.exists(target_path) or not os.path.exists(manifest_path):
            continue
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("clock", {}).get("click_safe", False):
            found.append((feature_path, target_path,
                          manifest.get("split_group", folder)))
    return found


def split_by_map(sessions, val_fraction=0.2, seed=7):
    groups = sorted(set(item[2] for item in sessions))
    random.Random(seed).shuffle(groups)
    if len(groups) < 2:
        print("WARNING: only one beatmap group; validation is not cross-map")
        return sessions, sessions
    val_count = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:val_count])
    return ([x for x in sessions if x[2] not in val_groups],
            [x for x in sessions if x[2] in val_groups])


def loss_parts(pred, target, track_scores=None):
    cursor = F.smooth_l1_loss(pred[:, :2], target[:, :2])
    tth = F.mse_loss(pred[:, 2], target[:, 2])
    hold_weight = 1.0 + target[:, 3] * 4.0
    hold = F.binary_cross_entropy(pred[:, 3], target[:, 3], weight=hold_weight)
    selection = pred.new_zeros(())
    if track_scores is not None:
        slots = target[:, 4].long()
        if bool((slots >= 0).any()):
            selection = F.cross_entropy(track_scores, slots, ignore_index=-1)
    total = cursor * 4.0 + tth * 8.0 + hold * 0.5 + selection * 0.75
    return cursor, tth, hold, total, selection


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums = np.zeros(5, dtype=np.float64)
    count = 0
    for features, target in loader:
        pred, _, aux = model(features.to(device), return_aux=True)
        sums += np.asarray([float(x) for x in loss_parts(
            pred, target.to(device), aux["track_scores"])])
        count += 1
    return sums / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/recordings")
    ap.add_argument("--out", default="data/policy/policy_v3.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nice", action="store_true")
    args = ap.parse_args()

    if args.nice:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass

    sessions = discover_sessions(args.sessions)
    if not sessions:
        raise RuntimeError("no click-safe actionlabel sessions found")
    source_dims = sorted(set(int(np.load(fp, mmap_mode="r").shape[1])
                             for fp, _, _ in sessions))
    if any(dim not in (FEATURE_DIM, FEATURE_DIM_V3) for dim in source_dims):
        raise RuntimeError(f"unsupported policy feature dimensions: {source_dims}")
    feat_dim = FEATURE_DIM_V3
    print(f"feature_dim={feat_dim} closed-loop-v4 source_dims={source_dims}")
    train_sessions, val_sessions = split_by_map(sessions)
    paths = lambda rows: [(a, b) for a, b, _ in rows]
    train_ds = SequenceDataset(paths(train_sessions), args.seq_len, args.stride,
                               training=True)
    val_ds = SequenceDataset(paths(val_sessions), args.seq_len, max(args.stride, 4),
                             training=False)
    if not len(train_ds):
        raise RuntimeError("sessions are shorter than --seq-len")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyGRU(input_dim=feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"sessions={len(sessions)} train_windows={len(train_ds)} "
          f"val_windows={len(val_ds)} device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        report_every = max(1, len(train_loader) // 20)
        for batch_index, (features, target) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            pred, _, aux = model(features.to(device, non_blocking=True), return_aux=True)
            loss = loss_parts(pred, target.to(device, non_blocking=True),
                              aux["track_scores"])[3]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.nice:
                time.sleep(0.004)
            running += float(loss)
            batches += 1
            if batch_index == 1 or batch_index % report_every == 0 or batch_index == len(train_loader):
                print(f"train epoch {epoch}/{args.epochs} "
                      f"batch {batch_index}/{len(train_loader)} "
                      f"loss={running/max(1, batches):.5f}", flush=True)
        metrics = validate(model, val_loader, device)
        print(f"epoch {epoch:03d}: train={running/max(1,batches):.5f} "
              f"val={metrics[3]:.5f} cursor={metrics[0]:.5f} "
              f"tth={metrics[1]:.5f} hold={metrics[2]:.5f} "
              f"select={metrics[4]:.5f}")
        if metrics[3] < best:
            best = float(metrics[3])
            model.save(args.out)
    print(f"best={best:.5f} -> {args.out}")


if __name__ == "__main__":
    main()
