"""Selgis ML training pipeline integration for osu! AI policy networks.

Integrates selgis.Trainer, SelgisConfig, and anomaly recovery mechanisms
with osu! SequenceDataset, UNetTinyOsuPolicy, TinyOsuPolicy, and PolicyLoss.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from selgis import Callback, SelgisConfig, Trainer

from .data import SequenceDataset
from .model import PolicyLoss, TinyOsuPolicy, UNetTinyOsuPolicy, policy_loss
from .train import load_policy

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")



def build_outputs_tensor(pred: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack multi-task network predictions into a 6D tensor aligned with target tensor shape.

    Dimension mapping:
      [0:2]: Target cursor coordinates (x, y)
      [2]:   Time-to-hit (tth) in milliseconds
      [3]:   Click activation logit
      [4]:   Hold activation logit
      [5]:   Presence activation logit
    """
    xy = pred["xy"]
    tth = pred["tth"].view(-1, 1)
    click_logit = pred["click_logit"].view(-1, 1)
    hold_logit = pred["hold_logit"].view(-1, 1)
    present_logit = pred.get("present_logit", torch.zeros_like(click_logit)).view(-1, 1)

    outputs = torch.cat([xy, tth, click_logit, hold_logit, present_logit], dim=-1)
    outputs.pred_dict = pred
    return outputs


def create_osu_forward_fn(
    criterion: nn.Module,
    lambda_action: float = 1.0,
) -> Callable[[nn.Module, Any], tuple[torch.Tensor, torch.Tensor]]:
    """Factory creating Selgis forward_fn closure bound to a PolicyLoss instance."""

    def forward_fn(model: nn.Module, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1]
            valid = batch[2] if len(batch) > 2 else torch.ones(x.shape[0], device=x.device)
        elif isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
            valid = batch.get("valid", torch.ones(x.shape[0], device=x.device))
        else:
            raise TypeError(f"Unsupported batch container: {type(batch)}")

        if model.training:
            roi_gt = y[:, :2] if (y is not None and y.shape[1] >= 2) else None
            try:
                pred = model(x, roi_gt=roi_gt)
            except TypeError:
                pred = model(x)
        else:
            pred = model(x)

        if isinstance(criterion, PolicyLoss):
            total_loss, loss_dict = criterion(pred, y, valid, lambda_action=lambda_action)
        elif criterion is not None:
            total_loss = criterion(pred, y, valid) if callable(criterion) else policy_loss(pred, y, valid)[0]
            loss_dict = {}
        else:
            total_loss, loss_dict = policy_loss(pred, y, valid)

        outputs = build_outputs_tensor(pred)
        outputs.loss_dict = loss_dict
        forward_fn.last_loss_dict = {
            k: float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v)
            for k, v in loss_dict.items()
        }
        return total_loss, outputs

    return forward_fn


def forward_fn(model: nn.Module, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Default forward function using model-attached or fallback PolicyLoss."""
    criterion = getattr(model, "criterion", None)
    if criterion is None:
        criterion = PolicyLoss().to(next(model.parameters()).device)
    fn = create_osu_forward_fn(criterion)
    return fn(model, batch)


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Evaluate coordinate error, click precision/recall/F1, and temporal hit error."""
    if preds is None or labels is None or len(preds) == 0 or len(labels) == 0:
        return {
            "xy_error": 0.0,
            "xy_mae": 0.0,
            "tth_error": 0.0,
            "click_accuracy": 0.0,
            "click_f1": 0.0,
            "click_precision": 0.0,
            "click_recall": 0.0,
            "hold_accuracy": 0.0,
            "hold_f1": 0.0,
        }

    preds_f = preds.float()
    labels_f = labels.float()

    xy_diff = preds_f[:, :2] - labels_f[:, :2]
    xy_dist = torch.norm(xy_diff, dim=-1)
    xy_error = float(xy_dist.mean().item())
    xy_mae = float(xy_diff.abs().mean().item())

    tth_diff = (preds_f[:, 2] - labels_f[:, 2]).abs()
    tth_error = float(tth_diff.mean().item())

    click_target = (labels_f[:, 3] > 0.5)
    click_pred = (torch.sigmoid(preds_f[:, 3]) > 0.5)
    click_acc = float((click_pred == click_target).float().mean().item()) * 100.0

    tp = float(((click_pred == 1) & (click_target == 1)).sum().item())
    fp = float(((click_pred == 1) & (click_target == 0)).sum().item())
    fn = float(((click_pred == 0) & (click_target == 1)).sum().item())
    click_prec = tp / max(tp + fp, 1e-6)
    click_rec = tp / max(tp + fn, 1e-6)
    click_f1 = float(2.0 * click_prec * click_rec / max(click_prec + click_rec, 1e-6))

    hold_target = (labels_f[:, 4] > 0.5)
    hold_pred = (torch.sigmoid(preds_f[:, 4]) > 0.5)
    hold_acc = float((hold_pred == hold_target).float().mean().item()) * 100.0

    h_tp = float(((hold_pred == 1) & (hold_target == 1)).sum().item())
    h_fp = float(((hold_pred == 1) & (hold_target == 0)).sum().item())
    h_fn = float(((hold_pred == 0) & (hold_target == 1)).sum().item())
    h_prec = h_tp / max(h_tp + h_fp, 1e-6)
    h_rec = h_tp / max(h_tp + h_fn, 1e-6)
    hold_f1 = float(2.0 * h_prec * h_rec / max(h_prec + h_rec, 1e-6))

    return {
        "xy_error": xy_error,
        "xy_mae": xy_mae,
        "tth_error": tth_error,
        "click_accuracy": click_acc,
        "click_f1": click_f1,
        "click_precision": click_prec,
        "click_recall": click_rec,
        "hold_accuracy": hold_acc,
        "hold_f1": hold_f1,
    }


class OsuMetricsCallback(Callback):
    """Monitors, formats, and logs osu! policy validation metrics per epoch."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history: list[dict[str, Any]] = []
        self.last_metrics: dict[str, float] = {}

    def on_evaluate(self, trainer: Any, metrics: dict[str, float]) -> None:
        self.last_metrics = dict(metrics)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        record = {"epoch": epoch + 1, **metrics}
        self.history.append(record)

        if not self.verbose:
            return

        train_loss = metrics.get("train_loss", 0.0)
        val_loss = metrics.get("loss", 0.0)
        xy_err = metrics.get("xy_error", 0.0)
        tth_err = metrics.get("tth_error", 0.0)
        click_f1 = metrics.get("click_f1", 0.0)
        click_acc = metrics.get("click_accuracy", 0.0)

        logger.info(
            "[Selgis] Epoch %03d | train_loss=%.4f | val_loss=%.4f | "
            "xy_err=%.4f | tth_err=%.1fms | click_acc=%.1f%% | click_f1=%.3f",
            epoch + 1,
            train_loss,
            val_loss,
            xy_err,
            tth_err,
            click_acc,
            click_f1,
        )


ComputeMetricsCallback = OsuMetricsCallback


class OsuCheckpointCallback(Callback):
    """Persists model checkpoints in the native dictionary format expected by load_policy."""

    def __init__(
        self,
        output_path: str | Path,
        version: str = "v8_5",
        sequence: int = 6,
    ):
        self.output_path = Path(output_path)
        self.out_stem = str(self.output_path.with_suffix(""))
        self.version = version
        self.sequence = sequence

        self.best_loss = float("inf")
        self.best_xy = float("inf")
        self.best_tth = float("inf")
        self.best_click = -1.0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_checkpoint_dict(
        self,
        trainer: Any,
        epoch: int,
        metrics: dict[str, float],
        tag: str,
    ) -> dict[str, Any]:
        model = trainer.model
        is_unet = isinstance(model, UNetTinyOsuPolicy)
        criterion = getattr(trainer, "criterion", None)
        log_vars = None
        if criterion is not None and hasattr(criterion, "log_vars"):
            log_vars = criterion.log_vars.detach().cpu().tolist()

        return {
            "state": model.state_dict(),
            "version": self.version,
            "arch": "unet" if is_unet else "cnn_gru",
            "action_fusion": getattr(model, "action_fusion", "tc_roi" if is_unet else "gru_only"),
            "tth_scale_ms": getattr(model, "tth_scale_ms", 500.0),
            "input": [2, 72, 96],
            "heatmap": [getattr(model, "hm_h", 18), getattr(model, "hm_w", 24)],
            "sequence": self.sequence,
            "hidden": getattr(model, "hidden", 160 if is_unet else 128),
            "spatial_channels": getattr(model, "spatial_channels", 80 if is_unet else 64),
            "val_loss": metrics.get("loss", float("inf")),
            "tag": tag,
            "metrics": metrics,
            "epoch": epoch + 1,
            "log_vars": log_vars,
        }

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        loss = metrics.get("loss", float("inf"))
        xy_err = metrics.get("xy_error", float("inf"))
        tth_err = metrics.get("tth_error", float("inf"))
        click_f1 = metrics.get("click_f1", -1.0)

        if loss < self.best_loss:
            self.best_loss = loss
            ckpt = self._make_checkpoint_dict(trainer, epoch, metrics, "best_val")
            torch.save(ckpt, self.output_path)

        if xy_err < self.best_xy:
            self.best_xy = xy_err
            ckpt = self._make_checkpoint_dict(trainer, epoch, metrics, "best_xy")
            torch.save(ckpt, f"{self.out_stem}_best_xy.pt")

        if click_f1 > self.best_click and click_f1 > 0.0:
            self.best_click = click_f1
            ckpt = self._make_checkpoint_dict(trainer, epoch, metrics, "best_click")
            torch.save(ckpt, f"{self.out_stem}_best_click.pt")

        if tth_err < self.best_tth:
            self.best_tth = tth_err
            ckpt = self._make_checkpoint_dict(trainer, epoch, metrics, "best_tth")
            torch.save(ckpt, f"{self.out_stem}_best_tth.pt")

        with open(f"{self.out_stem}_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch + 1, **metrics}, f, indent=2)

    def on_train_end(self, trainer: Any) -> None:
        if not self.output_path.exists():
            ckpt = self._make_checkpoint_dict(trainer, -1, {}, "final")
            torch.save(ckpt, self.output_path)


def resolve_session_path(session_info: dict, root: Path) -> Path:
    raw = Path(session_info["path"])
    if raw.exists():
        return raw
    candidate = root / raw.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Session path could not be resolved: {raw} or {candidate}")


def build_policy_model(
    hidden: int = 160,
    spatial_channels: int = 80,
    use_unet: bool = True,
    pretrained: str | None = None,
    action_fusion: str = "tc_roi",
    tth_scale_ms: float = 500.0,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any] | None]:
    """Instantiate either UNetTinyOsuPolicy or TinyOsuPolicy with optional checkpoint loading."""
    ck = None
    if pretrained and Path(pretrained).exists():
        model, ck = load_policy(pretrained, device=device)
        return model, ck

    dev = torch.device(device)
    if use_unet:
        model = UNetTinyOsuPolicy(
            hidden=hidden,
            spatial_channels=spatial_channels,
            action_fusion=action_fusion,
            tth_scale_ms=tth_scale_ms,
        ).to(dev)
    else:
        model = TinyOsuPolicy(
            hidden=hidden,
            spatial_channels=spatial_channels,
            tth_scale_ms=tth_scale_ms,
        ).to(dev)

    return model, ck


def train(
    root: str | Path,
    output: str | Path | None = None,
    epochs: int = 40,
    batch: int = 128,
    sequence: int = 6,
    stride: int = 2,
    lr: float = 1e-3,
    nice: bool = False,
    hidden: int = 160,
    spatial_channels: int = 80,
    use_unet: bool = True,
    use_kendall: bool = True,
    pretrained: str | None = None,
    action_fusion: str = "tc_roi",
    tth_scale_ms: float = 500.0,
    pos_weight: float = 2.0,
    weighting: str = "decoupled",
    version: str = "v8_5",
    method: str = "two_stage_freeze",
    stage1_epochs: int | None = None,
    stage2_epochs: int | None = None,
    fp16: bool | None = None,
    bf16: bool = False,
    nan_recovery: bool = True,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    device: str | None = None,
    callbacks: list[Callback] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute policy training via Selgis Trainer."""
    root_path = Path(root)
    index_file = root_path / "index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Dataset index file not found: {index_file}")

    index = json.loads(index_file.read_text(encoding="utf-8"))
    paths = [str(resolve_session_path(item, root_path)) for item in index["sessions"]]
    if not paths:
        raise RuntimeError("No sessions found in dataset index")

    if len(paths) == 1:
        val_path = paths[0]
        train_paths = paths
    else:
        val_candidates = [
            p for p in paths if any(name in p for name in ["Bass_Slut", "Embraced_by_the_Flame", "Satori_De_Pon"])
        ]
        val_path = val_candidates[0] if val_candidates else paths[-1]
        train_paths = [p for p in paths if p != val_path]

    train_ds = SequenceDataset(train_paths, sequence=sequence, stride=stride)
    val_ds = SequenceDataset([val_path], sequence=sequence, stride=max(4, stride))

    if device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    use_fp16 = (target_device.startswith("cuda") and fp16 is not False) if fp16 is None else fp16

    pin_mem = target_device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=pin_mem)

    model, ck = build_policy_model(
        hidden=hidden,
        spatial_channels=spatial_channels,
        use_unet=use_unet,
        pretrained=pretrained,
        action_fusion=action_fusion,
        tth_scale_ms=tth_scale_ms,
        device=target_device,
    )

    criterion = PolicyLoss(
        tth_scale_ms=tth_scale_ms,
        pos_weight=pos_weight,
        weighting=weighting,
    ).to(target_device)

    if isinstance(ck, dict) and ck.get("log_vars") and len(ck["log_vars"]) == 5:
        criterion.log_vars.data.copy_(torch.tensor(ck["log_vars"], device=target_device))
        criterion.log_vars.data[4] = max(0.0, float(criterion.log_vars.data[4]))

    if output is None or output in ["v8_5", "8.5", "v8.5", version]:
        models_dir = root_path.resolve().parent / "models"
        if not models_dir.parent.exists():
            models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        if version.startswith("Semga"):
            out_file = models_dir / f"{version}.pt"
        else:
            out_file = models_dir / f"unet_policy_{version}.pt"
    else:
        out_file = Path(output)
        if not str(out_file).endswith(".pt"):
            if version.startswith("Semga"):
                out_file = out_file / f"{version}.pt"
            else:
                out_file = out_file / f"unet_policy_{version}.pt"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    selgis_checkpoints_dir = out_file.parent / f"selgis_checkpoints_{out_file.stem}"

    selgis_config = SelgisConfig(
        batch_size=batch,
        eval_batch_size=batch,
        max_epochs=epochs,
        learning_rate=lr,
        grad_clip_norm=grad_clip_norm,
        nan_recovery=nan_recovery,
        fp16=use_fp16,
        bf16=bf16,
        weight_decay=1e-4,
        output_dir=str(selgis_checkpoints_dir),
        save_best_only=True,
        device=target_device,
        seed=seed,
    )

    forward_closure = create_osu_forward_fn(criterion)

    criterion_params = [p for p in criterion.parameters() if p.requires_grad]
    trainable_params = [p for p in model.parameters() if p.requires_grad] + criterion_params
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=selgis_config.weight_decay)

    pipeline_callbacks: list[Callback] = [
        OsuMetricsCallback(verbose=True),
        OsuCheckpointCallback(output_path=out_file, version=version, sequence=sequence),
    ]
    if callbacks:
        pipeline_callbacks.extend(callbacks)

    trainer = Trainer(
        model=model,
        config=selgis_config,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        criterion=criterion,
        optimizer=optimizer,
        forward_fn=forward_closure,
        compute_metrics=compute_metrics,
        callbacks=pipeline_callbacks,
    )

    metrics = trainer.train()
    return metrics
