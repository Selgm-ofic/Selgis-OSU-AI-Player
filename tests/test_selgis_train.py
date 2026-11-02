"""Unit and integration tests for Selgis ML training pipeline in osu! AI player."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from selgis import SelgisConfig, Trainer

from osu2.model import PolicyLoss, TinyOsuPolicy, UNetTinyOsuPolicy
from osu2.selgis_train import (
    OsuCheckpointCallback,
    OsuMetricsCallback,
    build_policy_model,
    compute_metrics,
    create_osu_forward_fn,
    forward_fn,
    train as selgis_train,
)
from osu2.train import load_policy


def generate_synthetic_session(file_path: Path, n_frames: int = 24) -> None:
    """Generate a valid synthetic session archive (.npz) conforming to SequenceDataset."""
    np.random.seed(42)
    images = np.random.randint(0, 255, (n_frames, 2, 72, 96), dtype=np.uint8)

    targets = np.zeros((n_frames, 6), dtype=np.float32)
    targets[:, 0] = np.linspace(0.2, 0.8, n_frames, dtype=np.float32)
    targets[:, 1] = np.linspace(0.3, 0.7, n_frames, dtype=np.float32)
    targets[:, 2] = np.linspace(200.0, -10.0, n_frames, dtype=np.float32)

    # Inject periodic clicks and holds safely within bounds
    if n_frames > 10:
        targets[10, 3] = 1.0
    if n_frames > 20:
        targets[20, 3] = 1.0
    if n_frames > 15:
        targets[12:min(16, n_frames), 4] = 1.0
    targets[:, 5] = 0.0

    valid = np.ones(n_frames, dtype=np.float32)
    stamps = np.linspace(0, 16.67 * n_frames, n_frames, dtype=np.float32)

    np.savez_compressed(
        file_path,
        images=images,
        targets=targets,
        valid=valid,
        stamps=stamps,
        version=6,
    )


class TestSelgisTrain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_selgis_")
        self.dataset_dir = Path(self.temp_dir) / "synthetic_dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.session_train_path = self.dataset_dir / "session_train.npz"
        self.session_val_path = self.dataset_dir / "session_val.npz"
        generate_synthetic_session(self.session_train_path, n_frames=24)
        generate_synthetic_session(self.session_val_path, n_frames=24)

        index_data = {
            "version": 2,
            "shape": [2, 72, 96],
            "channels": ["gray", "absdiff_16ms"],
            "sessions": [
                {"name": "session_train", "path": str(self.session_train_path), "frames": 24},
                {"name": "session_val", "path": str(self.session_val_path), "frames": 24},
            ],
        }
        (self.dataset_dir / "index.json").write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forward_fn_and_loss_backward(self):
        """Verify that forward_fn processes batches and enables gradient backpropagation."""
        batch_size = 3
        sequence = 6
        x = torch.randn(batch_size, sequence, 2, 72, 96)
        y = torch.tensor(
            [
                [0.4, 0.6, 120.0, 1.0, 0.0, 1.0],
                [0.2, 0.8, -20.0, 0.0, 1.0, 1.0],
                [0.5, 0.5, 50.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        valid = torch.ones(batch_size, dtype=torch.float32)
        batch = (x, y, valid)

        for use_unet in [False, True]:
            model, _ = build_policy_model(
                hidden=32,
                spatial_channels=16,
                use_unet=use_unet,
                action_fusion="gru_only",
                device="cpu",
            )
            criterion = PolicyLoss()
            fn = create_osu_forward_fn(criterion)

            loss, outputs = fn(model, batch)

            self.assertTrue(torch.is_tensor(loss))
            self.assertEqual(loss.dim(), 0)
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(loss.requires_grad)

            self.assertTrue(torch.is_tensor(outputs))
            self.assertEqual(tuple(outputs.shape), (batch_size, 6))

            model.zero_grad()
            loss.backward()

            has_gradients = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            self.assertTrue(has_gradients)

    def test_compute_metrics(self):
        """Verify metric computations for coordinate distance, TTH error, and click classification."""
        preds = torch.tensor(
            [
                [0.5, 0.5, 100.0, 2.5, -2.5, 0.0],
                [0.1, 0.9, -10.0, -3.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [0.5, 0.5, 120.0, 1.0, 0.0, 0.0],
                [0.1, 0.8, -10.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        metrics = compute_metrics(preds, labels)

        expected_keys = [
            "xy_error",
            "xy_mae",
            "tth_error",
            "click_accuracy",
            "click_f1",
            "click_precision",
            "click_recall",
            "hold_accuracy",
            "hold_f1",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertTrue(np.isfinite(metrics[key]))

        self.assertAlmostEqual(metrics["xy_error"], 0.05, delta=1e-4)
        self.assertAlmostEqual(metrics["tth_error"], 10.0, delta=1e-4)
        self.assertEqual(metrics["click_accuracy"], 100.0)
        self.assertAlmostEqual(metrics["click_f1"], 1.0, delta=1e-4)
        self.assertEqual(metrics["hold_accuracy"], 100.0)

    def test_selgis_trainer_two_epochs(self):
        """Instantiate selgis.Trainer directly and execute 2 epochs on synthetic data."""
        batch_size = 4
        x = torch.randn(8, 6, 2, 72, 96)
        y = torch.tensor([[0.5, 0.5, 50.0, 1.0, 0.0, 0.0]] * 8, dtype=torch.float32)
        valid = torch.ones(8, dtype=torch.float32)

        loader = DataLoader(TensorDataset(x, y, valid), batch_size=batch_size, shuffle=False)
        model = TinyOsuPolicy(hidden=32, spatial_channels=16)
        criterion = PolicyLoss()
        fwd = create_osu_forward_fn(criterion)

        output_dir = Path(self.temp_dir) / "selgis_trainer_out"
        config = SelgisConfig(
            max_epochs=2,
            learning_rate=1e-3,
            batch_size=batch_size,
            grad_clip_norm=1.0,
            nan_recovery=True,
            device="cpu",
            output_dir=str(output_dir),
            save_best_only=True,
        )

        metrics_cb = OsuMetricsCallback(verbose=False)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            eval_dataloader=loader,
            criterion=criterion,
            forward_fn=fwd,
            compute_metrics=compute_metrics,
            callbacks=[metrics_cb],
        )

        final_metrics = trainer.train()

        self.assertIn("train_loss", final_metrics)
        self.assertIn("loss", final_metrics)
        self.assertIn("xy_error", final_metrics)
        self.assertEqual(len(metrics_cb.history), 2)
        self.assertTrue(np.isfinite(final_metrics["loss"]))

    def test_end_to_end_pipeline_and_checkpoint_loading(self):
        """Run selgis_train pipeline end-to-end and reload resulting checkpoint with load_policy."""
        out_ckpt = Path(self.temp_dir) / "models" / "tiny_policy_test.pt"

        metrics = selgis_train(
            root=self.dataset_dir,
            output=out_ckpt,
            epochs=2,
            batch=4,
            sequence=6,
            stride=2,
            lr=1e-3,
            hidden=32,
            spatial_channels=16,
            use_unet=False,
            device="cpu",
            nan_recovery=True,
            grad_clip_norm=1.0,
        )

        self.assertIn("train_loss", metrics)
        self.assertTrue(out_ckpt.exists())

        loaded_model, ck = load_policy(str(out_ckpt), device="cpu")
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, TinyOsuPolicy)
        self.assertIsInstance(ck, dict)
        self.assertEqual(ck.get("arch"), "cnn_gru")

        test_input = torch.randn(1, 6, 2, 72, 96)
        with torch.no_grad():
            out = loaded_model(test_input)
        self.assertEqual(tuple(out["xy"].shape), (1, 2))
        self.assertIn("click", out)
        self.assertIn("tth", out)

    def test_unet_end_to_end_pipeline(self):
        """Run selgis_train with UNet architecture and verify model checkpoint validity."""
        out_ckpt = Path(self.temp_dir) / "models" / "unet_policy_test.pt"

        metrics = selgis_train(
            root=self.dataset_dir,
            output=out_ckpt,
            epochs=2,
            batch=4,
            sequence=6,
            stride=2,
            lr=1e-3,
            hidden=32,
            spatial_channels=16,
            use_unet=True,
            action_fusion="gru_only",
            device="cpu",
            nan_recovery=True,
            grad_clip_norm=1.0,
        )

        self.assertIn("train_loss", metrics)
        self.assertTrue(out_ckpt.exists())

        loaded_model, ck = load_policy(str(out_ckpt), device="cpu")
        self.assertIsInstance(loaded_model, UNetTinyOsuPolicy)
        self.assertEqual(ck.get("arch"), "unet")

        test_input = torch.randn(1, 6, 2, 72, 96)
        with torch.no_grad():
            out = loaded_model(test_input)
        self.assertEqual(tuple(out["xy"].shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
