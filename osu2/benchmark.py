import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .data import SequenceDataset
from .model import TinyOsuPolicy
from .train import load_policy


def benchmark_latency(model, device, iterations=500):
    x_step = torch.rand(1, 1, 2, 72, 96, device=device)
    state = None
    latencies = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(30):
            out = model(x_step, state=state)
            state = out.get("state")
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iterations):
            t0 = time.perf_counter()
            out = model(x_step, state=state)
            state = out.get("state")
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(latencies)
    return {
        "mean_ms": float(np.mean(lat)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "fps": float(1000.0 / max(1e-5, np.mean(lat)))
    }


def benchmark_mechanics(model, dataset_path, device, temperature=1.0):
    index_file = Path(dataset_path) / "index.json"
    if not index_file.exists():
        return None
    index = json.loads(index_file.read_text(encoding="utf-8"))
    sessions = index.get("sessions", [])
    if not sessions:
        return None
    held_out = sessions[-1]
    data = SequenceDataset([held_out["path"]], sequence=6, stride=1)

    xy_errors = []
    timing_errors = []
    note_timing_errors = []
    eval_objs = set()
    judgments = {300: 0, 100: 0, 50: 0, 0: 0}
    slider_tracking_errors = []
    slider_frames = 0
    slider_hold_hits = 0
    true_clicks = 0
    pred_clicks = 0
    true_pos_clicks = 0

    state = None
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for x, y, valid in data:
            if float(valid) <= 0.5:
                continue
            x_step = x[-1:].unsqueeze(0).to(device)
            out = model(x_step, state=state, temperature=temperature)
            state = out.get("state")
            px, py = out["xy"][0].detach().cpu().numpy()
            target_x = float(y[0])
            target_y = float(y[1])
            dist = float(math.hypot(px - target_x, py - target_y))
            xy_errors.append(dist)

            tth_pred = float(out["tth"][0].detach().cpu())
            tth_gt = float(y[2])
            timing_err = abs(tth_pred - tth_gt)
            timing_errors.append(timing_err)

            is_click_target = bool(y[3] > 0.5) if len(y) >= 6 else bool(abs(tth_gt) <= 16.0)
            is_hold_target = bool(y[4] > 0.5) if len(y) >= 6 else bool(y[3] > 0.5)

            click_pred = bool(out["click"][0] > 0.45) if "click" in out else bool(abs(tth_pred) <= 20.0)
            hold_pred = bool(out["hold"][0] > 0.45) if "hold" in out else click_pred

            if is_click_target:
                true_clicks += 1
                if click_pred:
                    true_pos_clicks += 1
            if click_pred:
                pred_clicks += 1

            # Per-hit-object judgment evaluation at the peak frame
            is_peak = (float(y[3]) >= 0.85 and abs(tth_gt) <= 45.0) if len(y) >= 6 else (abs(tth_gt) <= 16.0)
            obj_idx = int(y[5]) if len(y) >= 6 else None
            if is_peak and (obj_idx is None or obj_idx not in eval_objs):
                if obj_idx is not None:
                    eval_objs.add(obj_idx)
                hit_timing_err = abs(tth_gt)
                note_timing_errors.append(hit_timing_err)
                if dist <= 0.08 and click_pred:
                    if hit_timing_err <= 30.0:
                        judgments[300] += 1
                    elif hit_timing_err <= 60.0:
                        judgments[100] += 1
                    elif hit_timing_err <= 100.0:
                        judgments[50] += 1
                    else:
                        judgments[0] += 1
                else:
                    judgments[0] += 1

            if is_hold_target:
                slider_frames += 1
                slider_tracking_errors.append(dist)
                if hold_pred:
                    slider_hold_hits += 1

    total_hits = sum(judgments.values())
    acc = (judgments[300] * 300 + judgments[100] * 100 + judgments[50] * 50) / max(1, total_hits * 300) * 100.0 if total_hits > 0 else 0.0
    ur = float(np.std(note_timing_errors) * 10.0) if note_timing_errors else 0.0
    click_precision = true_pos_clicks / max(1, pred_clicks)
    click_recall = true_pos_clicks / max(1, true_clicks)
    f1 = 2.0 * click_precision * click_recall / max(1e-5, click_precision + click_recall)
    hold_continuity = (slider_hold_hits / max(1, slider_frames)) * 100.0

    return {
        "session": held_out["name"],
        "mean_xy": float(np.mean(xy_errors)) if xy_errors else 0.0,
        "p95_xy": float(np.percentile(xy_errors, 95)) if xy_errors else 0.0,
        "accuracy_pct": acc,
        "unstable_rate": ur,
        "judgments": judgments,
        "click_precision": click_precision,
        "click_recall": click_recall,
        "click_f1": f1,
        "slider_hold_continuity_pct": hold_continuity,
        "slider_tracking_mean": float(np.mean(slider_tracking_errors)) if slider_tracking_errors else 0.0,
    }


def benchmark_spinners(model, device, episodes=10, use_engine=True):
    from .rl_lab import NoteEnv
    from .spinner import SpinnerEngine
    env = NoteEnv(kind="spinner")
    engine = SpinnerEngine(rpm=420.0) if use_engine else None
    total_revs = []
    cleared = 0
    state = None
    with torch.inference_mode():
        for _ in range(episodes):
            env.reset()
            if engine:
                engine.reset()
            revs = 0.0
            for _ in range(120):
                obs = env.fast_obs()
                x_step = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
                out = model(x_step, state=state)
                state = out.get("state")
                px, py = out["xy"][0].detach().cpu().numpy()
                click = float(out["click"][0].detach().cpu()) if "click" in out else 1.0
                hold = float(out["hold"][0].detach().cpu()) if "hold" in out else 1.0

                if engine and engine.is_active(env, obs=obs, current_time=env.time):
                    action = engine.step(env.time, (px, py), env)
                else:
                    action = [px, py, click, hold]

                _, _, done, info = env.step(action)
                for n in env.notes:
                    if n.get("kind") == "spinner":
                        revs = max(revs, float(n.get("revolutions", 0.0)))
                if done:
                    break
            total_revs.append(revs)
            if revs >= 1.0:
                cleared += 1
    mean_revs = float(np.mean(total_revs)) if total_revs else 0.0
    spinner_dur = 0.75
    rpm = mean_revs * (60.0 / spinner_dur)
    return {
        "episodes": episodes,
        "mean_revolutions": mean_revs,
        "estimated_rpm": rpm,
        "clear_rate_pct": (cleared / max(1, episodes)) * 100.0,
    }


def benchmark(weights=None, dataset_path=r"C:\Selgis OSU AI Player\data\dataset_v7", iterations=500, suite="all", temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(weights, device)[0] if weights else TinyOsuPolicy().to(device).eval()
    model.eval()

    model_name = Path(weights).name if weights else "Untrained_TinyOsuPolicy"
    print(f"============================================================")
    print(f"  OSU! NEURAL NETWORK BENCHMARK: {model_name} (tau={temperature})")
    print(f"  Device: {device} | Autocast: FP16")
    print(f"============================================================")

    if suite in ("latency", "all"):
        print("\n--- 1. LATENCY & THROUGHPUT (Streaming T=1) ---")
        lat = benchmark_latency(model, device, iterations)
        print(f"  Mean Latency: {lat['mean_ms']:.2f} ms")
        print(f"  P50 Latency:  {lat['p50_ms']:.2f} ms")
        print(f"  P95 Latency:  {lat['p95_ms']:.2f} ms")
        print(f"  P99 Latency:  {lat['p99_ms']:.2f} ms")
        print(f"  FPS Capacity: {lat['fps']:.1f} FPS (Target 60 FPS: {lat['fps']/60.0:.1f}x headroom)")

    if suite in ("mechanics", "all"):
        print("\n--- 2. CLICKS & SLIDERS MECHANICS BENCHMARK ---")
        mech = benchmark_mechanics(model, dataset_path, device, temperature=temperature)
        if mech:
            print(f"  Held-out Map:      {mech['session']}")
            print(f"  Mean XY Distance:  {mech['mean_xy']:.4f} (p95: {mech['p95_xy']:.4f})")
            print(f"  Simulated Acc:     {mech['accuracy_pct']:.1f}%")
            print(f"  Unstable Rate:     {mech['unstable_rate']:.1f} UR")
            print(f"  Judgments:         300: {mech['judgments'][300]} | 100: {mech['judgments'][100]} | 50: {mech['judgments'][50]} | Miss: {mech['judgments'][0]}")
            print(f"  Click Precision:   {mech['click_precision']*100.0:.1f}%")
            print(f"  Click Recall:      {mech['click_recall']*100.0:.1f}%")
            print(f"  Click F1-Score:    {mech['click_f1']*100.0:.1f}%")
            print(f"  Slider Hold Cont:  {mech['slider_hold_continuity_pct']:.1f}%")
            print(f"  Slider Track Err:  {mech['slider_tracking_mean']:.4f}")

        print("\n--- 3. SPINNERS MECHANICS BENCHMARK ---")
        spin = benchmark_spinners(model, device)
        print(f"  Mean Revolutions:  {spin['mean_revolutions']:.2f} revs")
        print(f"  Spin Speed:        {spin['estimated_rpm']:.1f} RPM")
        print(f"  Spinner Clear Rate:{spin['clear_rate_pct']:.1f}%")

    print("\n============================================================")

