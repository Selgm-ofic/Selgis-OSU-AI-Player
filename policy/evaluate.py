"""Offline policy test: no capture, mouse movement or key injection."""
import argparse
import os
import time

import numpy as np

from policy.features import FEATURE_DIM_V3
from policy.model import (CURSOR_SPEED_SCALE, MAX_CURSOR_STEP, PolicyGRU,
                          TTH_SCALE_MS)


def closed_loop(model, features, targets, initial_offset=(0.0, 0.0)):
    n = min(len(features), len(targets))
    cursor = np.clip(targets[0, 2:4] + np.asarray(initial_offset), 0.0, 1.0)
    previous = cursor.copy()
    hidden = None
    predictions = []
    cursor_errors = []
    for index in range(n):
        feature = features[index].copy()
        dt = (1.0 / 30.0 if index == 0 else
              max(0.001, (targets[index, 0] - targets[index - 1, 0]) / 1000.0))
        velocity = (cursor - previous) / dt
        previous = cursor.copy()
        feature[-5:-3] = cursor
        feature[-3:-1] = velocity * 0.1
        out, hidden = model.act(feature, hidden)
        delta = out[:2] * CURSOR_SPEED_SCALE * min(dt, 0.05)
        delta = np.clip(delta, -MAX_CURSOR_STEP, MAX_CURSOR_STEP)
        cursor = np.clip(cursor + delta, 0.0, 1.0)
        desired = targets[min(index + 1, n - 1), 2:4]
        cursor_errors.append(np.abs(cursor - desired))
        predictions.append(out)
    return np.asarray(predictions), np.asarray(cursor_errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--policy", required=True)
    args = ap.parse_args()
    features = np.load(os.path.join(args.session, "policy_features.npy"))
    with np.load(os.path.join(args.session, "policy_targets.npz")) as archive:
        targets = archive["targets"]
    n = min(len(features), len(targets))
    model = PolicyGRU.load(args.policy)
    if features.shape[1] != FEATURE_DIM_V3:
        raise RuntimeError("closed-loop evaluation requires v3 object features")
    t0 = time.perf_counter()
    pred, cursor_error = closed_loop(model, features[:n], targets[:n], (.12, -.08))
    elapsed = time.perf_counter() - t0
    cursor_mae = np.mean(cursor_error)
    cursor_p95 = np.percentile(np.linalg.norm(cursor_error, axis=1), 95)
    true_tth = np.clip(targets[:n, 4], -TTH_SCALE_MS, TTH_SCALE_MS)
    tth_mae = np.mean(np.abs(pred[:, 2] * TTH_SCALE_MS - true_tth))
    near_mask = (targets[:n, 4] >= 0.0) & (targets[:n, 4] <= 250.0)
    near_tth_mae = (np.mean(np.abs(pred[near_mask, 2] * TTH_SCALE_MS -
                                   targets[:n, 4][near_mask]))
                    if np.any(near_mask) else float("nan"))
    hold_acc = np.mean((pred[:, 3] >= 0.5) == (targets[:n, 5] >= 0.5))
    print(f"frames={n} policy={elapsed*1000/max(1,n):.3f}ms "
          f"closed_loop_cursor_mae={cursor_mae:.4f} cursor_p95={cursor_p95:.4f} "
          f"tth_mae={tth_mae:.1f}ms "
          f"near_hit_mae={near_tth_mae:.1f}ms "
          f"hold_acc={hold_acc:.3f}")


if __name__ == "__main__":
    main()
