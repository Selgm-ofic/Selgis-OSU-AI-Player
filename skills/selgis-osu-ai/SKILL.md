---
name: selgis-osu-ai
description: Operational guidelines, architectural specifications, tensor contracts, training workflows, and debugging procedures for the Selgis osu! AI Player codebase.
---

# Selgis osu! AI Player — Agent Operational Skill

This skill provides essential operational rules, tensor shapes, checkpoint schemas, and workflow commands for AI coding assistants and autonomous agents working in this repository.

---

## 1. Architecture & Repository Map

The codebase consists of three primary layers:

1. **R&D and Modern UNet Engine (`osu2/`)**:
   - `osu2/model.py`: UNet policy networks (`UNetTinyOsuPolicy`), temporal fusion modules (`tc_roi`, `gru_only`), action heads, and multi-task loss (`PolicyLoss`).
   - `osu2/selgis_train.py`: Integration with the **Selgis-ML** framework (`selgis.Trainer`, `SelgisConfig`, anomaly recovery, and callback pipelines).
   - `osu2/data.py`: `SequenceDataset` loader, `.npz` sequence parser, temporal difference calculations.
   - `osu2/live.py`: High-performance live inference loop.
   - `osu2/recording_gui.py`: Developer GUI for beatmap recording, Danser rendering, and ROI calibration.
   - `osu2/rl_lab.py`: Reinforcement learning testbed.

2. **Lightweight Runtime Client (`osu_bot/`)**:
   - `osu_bot/capture.py`: Windows DXGI Desktop Duplication (`dxcam`) and `mss` screen capture.
   - `osu_bot/controller.py`: Win32 `SendInput` virtual mouse movement and keyboard tapping (Z/X).
   - `osu_bot/live.py`: Lightweight Relax bot runtime.
   - `osu_bot/model.py`: Minimal policy loader (`load_policy`).

3. **Entrypoints & GUIs**:
   - `main.py`: Unified CLI entry point for training, evaluation, benchmarking, dataset building, and GUI launchers.
   - `normis_gui.py`: Clean desktop GUI (neutral dark-gray theme) with auto-detection of osu! windows and model selector.
   - `run_live.py`: CLI console for live gameplay inference.

---

## 2. Tensor & Data Schema Contracts

When modifying, feeding, or validating neural networks in this project, adhere strictly to these tensor contracts:

### Input Tensor (5D Tensor)
- **Shape**: `(B, T, C, H, W)` where:
  - `B`: Batch size (e.g. 128 for training, 1 for live inference).
  - `T`: Sequence length = 6 frames.
  - `C`: 2 channels:
    - `Channel 0`: Current grayscale frame normalized to `[0.0, 1.0]`.
    - `Channel 1`: Temporal difference frame `abs(frame_t - frame_{t-1})` normalized to `[0.0, 1.0]`.
  - `H`: Height = 72 pixels.
  - `W`: Width = 96 pixels.
- **Normalization**: Pixel values are divided by 255.0.

### Model Outputs (`pred` Dictionary)
The forward pass returns a dictionary containing:
- `xy`: `(B, 2)` cursor position normalized in range `[0.0, 1.0]` relative to playfield.
- `tth`: `(B,)` time-to-hit in milliseconds (scaled by `tth_scale_ms=500.0`).
- `click_logit`: `(B,)` click binary logit (activation: `sigmoid(logit) > threshold`).
- `hold_logit`: `(B,)` slider hold binary logit.
- `present_logit`: `(B,)` object presence logit.
- `heatmap`: `(B, 432)` or `(B, 18, 24)` spatial attention map.

---

## 3. Checkpoint Schema & Storage Protocol

Checkpoints (`.pt` files in `data/models/`) are loaded via `load_policy(path, device)` in `osu2/train.py` and `osu_bot/model.py`.

### Required Checkpoint Dictionary Schema
Never save a raw `torch.save(model.state_dict(), path)`. Always package the composite dictionary:
```python
checkpoint_dict = {
    "state": model.state_dict(),
    "version": "v7_2",         # Target architecture version tag
    "arch": "unet",            # "unet" or "cnn_gru"
    "action_fusion": "tc_roi", # "tc_roi", "gru_only", etc.
    "tth_scale_ms": 500.0,
    "input": [2, 72, 96],
}
torch.save(checkpoint_dict, output_path)
```

### Safety Rule for Weights
- **Never overwrite existing checkpoints** in `data/models/` during tests or ad-hoc training runs.
- Ad-hoc and test runs must output to explicit copy paths (e.g., `data/models/test_*.pt`).

---

## 4. Training Engine Protocol (Selgis-ML)

Training is driven by `main.py train` with `--engine selgis` (the default backend).

### Standard Training Commands
```bash
# Train UNet v7.2 policy on the combined subset dataset
python main.py train --engine selgis --version v7_2 --dataset data/dataset_v8_combined_subset --epochs 40 --batch 128

# Train UNet policy with Two-Stage Freeze transfer learning
python main.py train --engine selgis --version v8_5 --method two_stage_freeze --epochs 40 --batch 128
```

### Guardrail Best Practices
1. **Never edit the external library** `C:\Selgis ML2.0\selgis`. All hooks and callbacks must reside in `osu2/selgis_train.py` or runtime wrappers.
2. **In-Memory State Snapshots**: Set `state_storage='memory'` in `SelgisConfig` to prevent disk serialization latency during high-speed GPU training.
3. **AMP Unscaling Order**: Ensure `scaler.unscale_(optimizer)` is executed before calling gradient clipping (`clip_grad_norm_`).

---

## 5. Screen Capture & Windows Input Constraints

- **Capture Pipeline**: `osu_bot.capture.ScreenCapture` prioritizes DXGI Desktop Duplication (`dxcam`), which bypasses Windows desktop composition latency. It falls back to `mss` if DXGI is unavailable.
- **Safety Hotkeys**:
  - `F2`: Toggles AI control (play / pause).
  - `F10`: Emergency break (immediately releases all mouse/keyboard hooks).
- **Coordinate Space**: Normalized `xy` predictions in `[0.0, 1.0]` must be mapped to screen coordinates using the configured playfield bounding box (`playfield.left`, `playfield.top`, `playfield.right`, `playfield.bottom` in `config.yaml`).

---

## 6. Pre-Commit Verification Checklist

Before committing or pushing any modifications:
1. Verify no syntax or import regressions:
   ```bash
   python -m py_compile main.py normis_gui.py run_live.py osu2/*.py osu_bot/*.py
   ```
2. Execute the automated unit test suite:
   ```bash
   python -m unittest discover tests
   ```
   All 23 tests must pass.
3. Check `git status` to ensure temporary files, benchmark dumps, or media files (`.mp4`, `.npz`, `tesys/`) are not staged.
4. Ensure no emojis are included in commit messages, docstrings, or code comments.
