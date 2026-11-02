import json, os, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import SequenceDataset
from .model import TinyOsuPolicy, UNetTinyOsuPolicy, PolicyLoss, policy_loss


def train(root, output=None, epochs=40, batch=128, sequence=6, stride=2, lr=1e-3, nice=False,
          hidden=128, spatial_channels=64, use_unet=False, use_kendall=True, pretrained=None,
          action_fusion="tc_roi", tth_scale_ms=500.0, pos_weight=2.0, weighting="hybrid",
          version="v8_5", method="sc_sja_fc"):
    index = json.loads((Path(root) / "index.json").read_text(encoding="utf-8"))
    paths = [x["path"] for x in index["sessions"]]
    if len(paths) < 2:
        raise RuntimeError("need at least two click-safe sessions")
    val_candidates = [p for p in paths if any(name in p for name in ["Bass_Slut", "Embraced_by_the_Flame", "Satori_De_Pon"])]
    val_path = val_candidates[0] if val_candidates else paths[-1]
    train_paths = [p for p in paths if p != val_path]
    train_ds = SequenceDataset(train_paths, sequence, stride)
    val_ds = SequenceDataset([val_path], sequence, max(4, stride))
    kwargs = dict(batch_size=batch, num_workers=0, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True, **kwargs)
    val_dl = DataLoader(val_ds, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ck = None
    if pretrained and Path(pretrained).exists():
        loaded, ck = load_policy(pretrained, device)
        model = loaded
    elif use_unet or method == "sc_sja_fc" or (version and (version.startswith("Semga") or "v8" in version)):
        model = UNetTinyOsuPolicy(hidden=hidden, spatial_channels=spatial_channels, action_fusion=action_fusion, tth_scale_ms=tth_scale_ms).to(device)
    else:
        model = TinyOsuPolicy(hidden=hidden, spatial_channels=spatial_channels, tth_scale_ms=tth_scale_ms).to(device)
    criterion = PolicyLoss(tth_scale_ms=tth_scale_ms, pos_weight=pos_weight, weighting=weighting).to(device)
    if criterion is not None and isinstance(ck, dict) and ck.get("log_vars") and len(ck["log_vars"]) == 5:
        criterion.log_vars.data.copy_(torch.tensor(ck["log_vars"], device=device))
        criterion.log_vars.data[4] = max(0.0, float(criterion.log_vars.data[4]))
    criterion_params = list(criterion.parameters()) if (criterion is not None and any(p.requires_grad for p in criterion.parameters())) else []
    stage1_epochs = max(2, epochs // 2)
    stage2_epochs = epochs - stage1_epochs
    if method == "two_stage_freeze":
        for name, p in model.named_parameters():
            if name.startswith("action.") or name.startswith("roi_encoder."):
                p.requires_grad = False
            else:
                p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad] + criterion_params
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage1_epochs, eta_min=1e-5)
    else:
        params = list(model.parameters()) + criterion_params
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    import numpy as np
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=2048.0)
    best = float("inf")
    best_xy = float("inf")
    best_tth = float("inf")
    best_click = -1.0
    if output is None or output in ["v8_5", "8.5", "v8.5", version]:
        models_dir = Path(root).resolve().parent / "models"
        if not models_dir.parent.exists():
            models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        if version.startswith("Semga"):
            output = str(models_dir / f"{version}.pt")
        else:
            output = str(models_dir / f"unet_policy_{version}.pt")
    elif not str(output).endswith(".pt"):
        out_p = Path(output)
        if version.startswith("Semga"):
            output = str(out_p / f"{version}.pt")
        else:
            output = str(out_p / f"unet_policy_{version}.pt")
    out_stem = str(Path(output).with_suffix(""))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} arch={'unet' if use_unet else 'cnn_gru'} fusion={action_fusion} tth_scale={tth_scale_ms} weighting={weighting} method={method} output={output}", flush=True)

    t1_epoch = max(2, int(epochs * 0.25))
    t2_epoch = max(t1_epoch + 2, int(epochs * 0.80))
    if method == "two_stage_freeze":
        print(f"=== Method TWO-STAGE-FREEZE: Stage 1 Aim (ep 1-{stage1_epochs}) | Stage 2 Action Heads (ep {stage1_epochs+1}-{epochs}) ===", flush=True)
    elif method == "sc_sja_fc":
        print(f"=== Method SC-SJA-FC: Phase 1 (ep 1-{t1_epoch}) | Phase 2 (ep {t1_epoch+1}-{t2_epoch}) | Phase 3 (ep {t2_epoch+1}-{epochs}) ===", flush=True)

    for epoch in range(1, epochs + 1):
        if method == "two_stage_freeze":
            if epoch <= stage1_epochs:
                sigma = 0.0
                beta_pred = 0.0
                lambda_action = 0.0
                print(f"--- [Two-Stage Freeze: Stage 1 Aim] Ep {epoch:03d}/{epochs} | Backbone Active | Lambda_action=0.0 ---", flush=True)
            else:
                if epoch == stage1_epochs + 1:
                    print(f"=== TWO-STAGE FREEZE: Transitioning to Stage 2 at epoch {epoch}. Freezing Aim Backbone 100%. Unfreezing Action Heads ===", flush=True)
                    for name, p in model.named_parameters():
                        if name.startswith("action.") or name.startswith("roi_encoder."):
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                    action_params = [p for p in model.parameters() if p.requires_grad] + criterion_params
                    trainable_count = sum(p.numel() for p in action_params)
                    total_count = sum(p.numel() for p in model.parameters())
                    print(f"Stage 2 Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.2f}%)", flush=True)
                    opt = torch.optim.AdamW(action_params, lr=3e-4, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage2_epochs, eta_min=1e-5)
                    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=2048.0)
                sigma = 0.08
                beta_pred = 0.5
                lambda_action = 1.0
                print(f"--- [Two-Stage Freeze: Stage 2 Action] Ep {epoch:03d}/{epochs} | Action Heads Only | Jitter sigma=0.08 | Beta=0.50 | Lambda_action=1.00 ---", flush=True)
        elif method == "sc_sja_fc":
            if epoch <= t1_epoch:
                phase = 1
                sigma = 0.0
                beta_pred = 0.0
                lambda_action = 0.0
            elif epoch <= t2_epoch:
                phase = 2
                progress = float(epoch - t1_epoch) / float(max(1, t2_epoch - t1_epoch))
                sigma = 0.08 * (progress ** 2)
                beta_pred = 0.5 * progress
                lambda_action = min(1.0, progress * 1.5)
            else:
                phase = 3
                sigma = 0.08
                beta_pred = 0.5
                lambda_action = 1.0
                for pg in opt.param_groups:
                    pg["lr"] = min(pg["lr"], 3e-5)
            print(f"--- [SC-SJA-FC Phase {phase}] Ep {epoch:03d}/{epochs} | Jitter sigma={sigma:.4f} | Beta={beta_pred:.2f} | Lambda_action={lambda_action:.2f} ---", flush=True)
        else:
            sigma = 0.0
            beta_pred = 0.0
            lambda_action = 1.0
            freeze_epoch = max(15, int(epochs * 0.65))
            if epoch == freeze_epoch and epochs >= 20:
                print(f"=== PHASE 2: Freezing Vision & Aim Backbone at epoch {epoch}. Fine-tuning Click Head only ===", flush=True)
                for name, p in model.named_parameters():
                    if not (name.startswith("action.") or name.startswith("roi_encoder.")):
                        p.requires_grad = False
        model.train()
        total = 0.0
        valid_train_samples = 0
        cur_lr = opt.param_groups[0]["lr"]
        grad_norms = []
        for step, (x, y, v) in enumerate(train_dl, 1):
            x, y, v = x.to(device, non_blocking=True), y.to(device, non_blocking=True), v.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(x, roi_jitter=sigma, roi_gt=y[:, :2], roi_beta=beta_pred)
                loss, _ = criterion(pred, y, v, lambda_action=lambda_action) if criterion is not None else policy_loss(pred, y, v)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            active_params = [p for pg in opt.param_groups for p in pg["params"] if p.requires_grad]
            valid_grads = all(torch.isfinite(p.grad).all() for p in active_params if p.grad is not None)
            if valid_grads:
                gn = float(torch.nn.utils.clip_grad_norm_(active_params, 5.0))
                if np.isfinite(gn):
                    grad_norms.append(gn)
            scaler.step(opt)
            scaler.update()
            lv = float(loss.detach().cpu())
            if np.isfinite(lv):
                total += lv * len(x)
                valid_train_samples += len(x)
            if step % 100 == 0 or step == len(train_dl):
                print(f"ep {epoch:03d}/{epochs} [{step}/{len(train_dl)}] loss={lv:.2f} lr={cur_lr:.1e}", flush=True)
            if nice:
                time.sleep(0.004)
        mean_train = total / max(1, valid_train_samples)
        mean_grad = float(np.mean(grad_norms)) if grad_norms else 0.0

        model.eval()
        val = 0.0
        count = 0
        raw_sums = {"hm": 0.0, "xy": 0.0, "tth": 0.0, "click": 0.0, "hold": 0.0}
        aim_dists = []
        timing_diffs = []
        timing_biases = []
        jitter_diffs = []
        click_tp = click_fp = click_fn = 0
        hold_tp = hold_fp = hold_fn = 0

        with torch.inference_mode():
            for x, y, v in val_dl:
                x, y, v = x.to(device), y.to(device), v.to(device)
                pred = model(x)
                loss, ldict = criterion(pred, y, v) if criterion is not None else policy_loss(pred, y, v)
                val += float(loss) * len(x)
                count += len(x)
                for k in raw_sums:
                    raw_sums[k] += float(ldict[k].detach().cpu()) * len(x)

                mask = (v > 0.5)
                if bool(mask.any()):
                    px = pred["xy"][mask].detach().cpu().numpy()
                    ty = y[mask, :2].detach().cpu().numpy()
                    dists = np.sqrt(np.sum((px - ty) ** 2, axis=-1))
                    aim_dists.extend(dists.tolist())
                    if len(px) > 1:
                        j = np.sqrt(np.sum(np.diff(px, axis=0) ** 2, axis=-1))
                        jitter_diffs.extend(j.tolist())

                    ptth = pred["tth"][mask].detach().cpu().numpy().flatten()
                    ytth = y[mask, 2].detach().cpu().numpy().flatten()
                    tdiff = ptth - ytth
                    timing_diffs.extend(tdiff.tolist())

                    if y.shape[1] >= 4 and "click_logit" in pred:
                        pcl = (torch.sigmoid(pred["click_logit"][mask]) > 0.5).detach().cpu().numpy().flatten()
                        ycl = (y[mask, 3] > 0.5).detach().cpu().numpy().flatten()
                        click_tp += int(np.sum((pcl == 1) & (ycl == 1)))
                        click_fp += int(np.sum((pcl == 1) & (ycl == 0)))
                        click_fn += int(np.sum((pcl == 0) & (ycl == 1)))
                        hit_clicks = (pcl == 1) & (ycl == 1)
                        if np.any(hit_clicks):
                            timing_biases.extend(tdiff[hit_clicks].tolist())

                    if y.shape[1] >= 5 and "hold_logit" in pred:
                        phl = (torch.sigmoid(pred["hold_logit"][mask]) > 0.5).detach().cpu().numpy().flatten()
                        yhl = (y[mask, 4] > 0.5).detach().cpu().numpy().flatten()
                        hold_tp += int(np.sum((phl == 1) & (yhl == 1)))
                        hold_fp += int(np.sum((phl == 1) & (yhl == 0)))
                        hold_fn += int(np.sum((phl == 0) & (yhl == 1)))

        score = val / max(1, count)
        raw_hm = raw_sums["hm"] / max(1, count)
        raw_xy = raw_sums["xy"] / max(1, count)
        raw_tth = raw_sums["tth"] / max(1, count)
        raw_click = raw_sums["click"] / max(1, count)
        raw_hold = raw_sums["hold"] / max(1, count)

        if criterion is not None:
            cv = criterion.log_vars.clamp(min=-4.0, max=4.0).detach().cpu().numpy()
            prec = np.exp(-cv)
            clamp_reached = any(cv <= -3.999)
            star = lambda i: "*" if cv[i] <= -3.999 else ""
            w_hm = prec[0] * raw_hm + cv[0]
            w_xy = prec[1] * (30.0 * raw_xy) + cv[1]
            w_tth = prec[2] * raw_tth + cv[2]
            w_click = prec[3] * raw_click + cv[3]
            w_hold = prec[4] * raw_hold + cv[4]
        else:
            prec = [0.3, 30.0, 15.0, 2.0, 1.0]
            w_hm, w_xy, w_tth, w_click, w_hold = 0.3 * raw_hm, 30.0 * raw_xy, 15.0 * raw_tth, 2.0 * raw_click, 1.0 * raw_hold
            clamp_reached = False
            star = lambda i: ""

        aim_arr = np.array(aim_dists) if aim_dists else np.array([0.0])
        aim_mean = float(np.mean(aim_arr))
        aim_med = float(np.median(aim_arr))
        aim_p95 = float(np.percentile(aim_arr, 95))
        aim_p99 = float(np.percentile(aim_arr, 99))
        p_02 = float(np.mean(aim_arr < 0.02) * 100.0)
        p_05 = float(np.mean(aim_arr < 0.05) * 100.0)
        p_10 = float(np.mean(aim_arr < 0.10) * 100.0)
        jitter_val = float(np.mean(jitter_diffs)) if jitter_diffs else 0.0

        t_arr = np.array(timing_diffs) if timing_diffs else np.array([0.0])
        abs_t = np.abs(t_arr)
        t_mae = float(np.mean(abs_t))
        t_med = float(np.median(abs_t))
        t_p95 = float(np.percentile(abs_t, 95))
        t_bias = float(np.mean(t_arr))
        p_10ms = float(np.mean(abs_t <= 10.0) * 100.0)
        p_20ms = float(np.mean(abs_t <= 20.0) * 100.0)
        p_50ms = float(np.mean(abs_t <= 50.0) * 100.0)

        cp = click_tp / max(1, click_tp + click_fp)
        cr = click_tp / max(1, click_tp + click_fn)
        cf1 = 2.0 * cp * cr / max(1e-5, cp + cr)
        cbias = float(np.mean(timing_biases)) if timing_biases else 0.0

        hp = hold_tp / max(1, hold_tp + hold_fp)
        hr = hold_tp / max(1, hold_tp + hold_fn)
        hf1 = 2.0 * hp * hr / max(1e-5, hp + hr)

        print(f"ep {epoch:03d}/{epochs}", flush=True)
        print(f"loss: train={mean_train:.2f} val={score:.2f} | lr={cur_lr:.1e} grad={mean_grad:.2f}\n", flush=True)
        print(f"raw val:\n  hm={raw_hm:.4f} xy={raw_xy:.4f} tth={raw_tth:.4f} click={raw_click:.3f} hold={raw_hold:.3f}\n", flush=True)
        print(f"weighted:\n  hm={w_hm:.2f} xy={w_xy:.2f} tth={w_tth:.2f} click={w_click:.2f} hold={w_hold:.2f}\n", flush=True)
        prec_line = f"precision:\n  hm={prec[0]:.2f}{star(0)} xy={prec[1]:.2f}{star(1)} tth={prec[2]:.2f}{star(2)} click={prec[3]:.2f}{star(3)} hold={prec[4]:.2f}{star(4)}"
        if clamp_reached:
            prec_line += "\n  * precision clamp reached"
        print(prec_line + "\n", flush=True)
        print(f"aim:\n  mean={aim_mean:.4f} med={aim_med:.4f} p95={aim_p95:.4f} p99={aim_p99:.4f}", flush=True)
        print(f"  <.02={p_02:.1f}% <.05={p_05:.1f}% <.10={p_10:.1f}%", flush=True)
        print(f"  jitter={jitter_val:.4f}\n", flush=True)
        print(f"timing:\n  TTH MAE={t_mae:.1f}ms med={t_med:.1f}ms p95={t_p95:.1f}ms bias={t_bias:+.1f}ms", flush=True)
        print(f"  ±10ms={p_10ms:.1f}% ±20ms={p_20ms:.1f}% ±50ms={p_50ms:.1f}%\n", flush=True)
        print(f"click:\n  P={cp:.3f} R={cr:.3f} F1={cf1:.3f}", flush=True)
        print(f"  early/late bias={cbias:+.1f}ms\n", flush=True)
        print(f"hold:\n  P={hp:.3f} R={hr:.3f} F1={hf1:.3f}\n", flush=True)
        current_metrics = {
            "epoch": epoch, "val_loss": score, "train_loss": mean_train,
            "xy_mean": aim_mean, "xy_p95": aim_p95,
            "tth_mae": t_mae, "tth_bias": t_bias,
            "click_p": cp, "click_r": cr, "click_f1": cf1,
            "hold_f1": hf1
        }
        def make_ckpt(tag, best_val):
            return {
                "state": model.state_dict(),
                "version": version if version else ("8.5" if ("v8_5" in out_stem or "8.5" in out_stem) else 2),
                "arch": "unet" if use_unet else "cnn_gru",
                "action_fusion": action_fusion,
                "tth_scale_ms": tth_scale_ms,
                "input": [2, 72, 96],
                "heatmap": [18, 24],
                "sequence": sequence,
                "hidden": hidden,
                "spatial_channels": spatial_channels,
                "val_loss": score,
                "tag": tag,
                "metrics": current_metrics,
                "log_vars": criterion.log_vars.detach().cpu().tolist() if criterion is not None else None
            }
        if np.isfinite(score) and score < best:
            best = score
            torch.save(make_ckpt("best_val", best), output)
            print(f"saved best_val -> {output}", flush=True)
        if np.isfinite(aim_p95) and aim_p95 < best_xy:
            best_xy = aim_p95
            p_xy = f"{out_stem}_best_xy.pt"
            torch.save(make_ckpt("best_xy", best_xy), p_xy)
            print(f"saved best_xy -> {p_xy}", flush=True)
        if np.isfinite(t_mae) and t_mae < best_tth:
            best_tth = t_mae
            p_tth = f"{out_stem}_best_tth.pt"
            torch.save(make_ckpt("best_tth", best_tth), p_tth)
            print(f"saved best_tth -> {p_tth}", flush=True)
        if np.isfinite(cf1) and cf1 > best_click and cf1 > 0.0:
            best_click = cf1
            p_click = f"{out_stem}_best_click.pt"
            torch.save(make_ckpt("best_click", best_click), p_click)
            print(f"saved best_click -> {p_click}", flush=True)
        if method == "two_stage_freeze" and epoch == stage1_epochs:
            p_stage1 = f"{out_stem}_stage1_aim.pt"
            torch.save(make_ckpt("stage1_aim", aim_p95), p_stage1)
            print(f"saved stage1_aim -> {p_stage1}", flush=True)
        Path(f"{out_stem}_metrics.json").write_text(json.dumps(current_metrics, indent=2), encoding="utf-8")
        scheduler.step()


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["state"] if isinstance(ck, dict) and "state" in ck else ck
    is_unet = (isinstance(ck, dict) and ck.get("arch") == "unet") or any(k.startswith("vision.enc") for k in state.keys())
    hm_hw = tuple(ck.get("heatmap", (18, 24))) if isinstance(ck, dict) else (18, 24)
    hidden = int(ck.get("hidden", 160 if is_unet else 128)) if isinstance(ck, dict) else (160 if is_unet else 128)
    spatial = int(ck.get("spatial_channels", 80 if is_unet else 64)) if isinstance(ck, dict) else (80 if is_unet else 64)
    if "action.weight" in state:
        action_in = state["action.weight"].shape[1]
    elif "action.tth_head.weight" in state:
        action_in = state["action.tth_head.weight"].shape[1]
    elif "action.click_head.weight" in state:
        action_in = state["action.click_head.weight"].shape[1]
    else:
        action_in = hidden
    default_fusion = "tc_roi" if action_in == (hidden + 96) else ("gru_attended" if action_in == (hidden + spatial) else "gru_only")
    action_fusion = ck.get("action_fusion", default_fusion) if isinstance(ck, dict) else default_fusion
    default_scale = 1200.0 if (not is_unet or action_fusion == "gru_attended") else 500.0
    tth_scale = float(ck.get("tth_scale_ms", default_scale)) if isinstance(ck, dict) else default_scale
    if is_unet:
        model = UNetTinyOsuPolicy(heatmap_hw=hm_hw, hidden=hidden, spatial_channels=spatial, action_fusion=action_fusion, tth_scale_ms=tth_scale)
    else:
        model = TinyOsuPolicy(heatmap_hw=hm_hw, hidden=hidden, spatial_channels=spatial, tth_scale_ms=tth_scale)
    state = dict(state)
    if "action.weight" in state:
        w, b = state["action.weight"], state["action.bias"]
        if w.shape[0] == 3:
            new_w = torch.zeros(4, w.shape[1], dtype=w.dtype)
            new_b = torch.zeros(4, dtype=b.dtype)
            new_w[0] = w[0]; new_w[2] = w[1]; new_w[3] = w[2]
            new_b[0] = b[0]; new_b[2] = b[1]; new_b[3] = b[2]
            w, b = new_w, new_b
        state["action.tth_head.weight"] = w[0:1]
        state["action.tth_head.bias"] = b[0:1]
        state["action.click_head.weight"] = w[1:2]
        state["action.click_head.bias"] = b[1:2]
        state["action.hold_head.weight"] = w[2:3]
        state["action.hold_head.bias"] = b[2:3]
        state["action.present_head.weight"] = w[3:4]
        state["action.present_head.bias"] = b[3:4]
        del state["action.weight"]
        del state["action.bias"]

    if "embed.2.weight" in state:
        ew = state["embed.2.weight"]
        target_in = model.embed[2].weight.shape[1]
        if ew.shape[1] * 4 == target_in:
            state["embed.2.weight"] = ew.repeat(1, 4) / 4.0

    model.load_state_dict(state)
    return model.to(device).eval(), ck


