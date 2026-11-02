# Selgis osu! AI Player (v8 R&D Engine)

[![CI](https://github.com/ByIBBL/selgis-osu-ai-player/actions/workflows/ci.yml/badge.svg)](https://github.com/ByIBBL/selgis-osu-ai-player/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Comprehensive deep-learning player, training pipeline, and live inference assistant for osu! based on UNet spatial-temporal heatmap and action prediction architectures.

---

## Key Features

- **End-to-End Visual Policy**: Directly maps high-speed screen captures (DXGI Desktop Duplication) to cursor trajectory and hit timing (Z/X).
- **Dual Runtime Interfaces**:
  - **Normis Player GUI** (`run_player_gui.bat`): Lightweight, user-friendly desktop interface with drag-and-drop ROI calibration, model selector, sensitivity tuning, and themes.
  - **Dev & R&D Suite** (`run_gui.bat`): Full research environment with Danser synthetic playback, osu! Lazer integration, session recorder, and RL Lab.
- **Ultra-low Latency Inference**: TensorRT and PyTorch CUDA optimizations delivering sub-frame latency at 60–240 FPS.
- **Relax & Full-Auto Modes**: Choose between mouse aim assistance only (Relax) or fully autonomous keyboard tapping and aiming.

---

## Quick Start

### 1. Requirements
- Windows 10 / 11 (64-bit)
- Python 3.10, 3.11, or 3.12 (ensure "Add python.exe to PATH" is enabled)
- NVIDIA GPU with CUDA support recommended (GTX 1060+ or RTX series)

### 2. Installation
Run the automated environment installer:
```bat
install_dependencies.bat
```
This script detects your GPU, installs PyTorch with CUDA hardware acceleration, and configures all required dependencies from `requirements.txt`.

### 3. Launching
Choose your preferred interface:
- **Player GUI (Normis Edition)**: Double-click `run_player_gui.bat`
- **Research & Recording GUI**: Double-click `run_gui.bat`
- **Interactive CLI Setup**: Double-click `settings.bat`
- **One-Click Live Bot**: Double-click `run.bat`

---

## In-Game Controls & Hotkeys

| Hotkey | Action | Description |
| :--- | :--- | :--- |
| **F2** | **Toggle AI (Play / Pause)** | Toggles neural network control during gameplay. Bot always initializes in paused state. |
| **F10** | **Emergency Stop** | Immediately detaches mouse/keyboard hooks and restores complete manual control. |

---

## Playfield ROI Setup

To align the neural policy with your osu! playfield:
1. Launch osu! in Borderless or Windowed mode.
2. In the Player GUI (`run_player_gui.bat`), click **"Найти окно osu!"** (Auto-detect) or use **"Drag & Select"** to box the playfield.
3. Save settings to `config.yaml`.

---

## CLI & Research Engine (`main.py`)

The unified CLI entry point provides full control over data synthesis, training, and benchmarking:

```bash
# Launch Dev GUI
python main.py gui

# Launch Reinforcement Learning Lab
python main.py rl --episodes 2500

# Train UNet Policy (Two-Stage Freeze method)
python main.py train --version v8_5 --method two_stage_freeze

# Benchmark inference latency and mechanics
python main.py bench --weights data/models/unet_policy_v8_5.pt

# Evaluate accuracy against validation dataset
python main.py evaluate --weights data/models/unet_policy_v8_5.pt

# Build dataset from session recordings
python main.py build --source data/recordings --out data/dataset_v8
```

---

## Repository Structure

```text
├── main.py                     # Unified CLI entry point for R&D
├── normis_gui.py               # Lightweight desktop Player GUI
├── run_live.py                 # Live inference runner and settings console
├── install_helper.py           # Hardware-aware environment installer
├── config.yaml                 # Runtime and model configuration
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
│
├── run_gui.bat                 # Launcher: Dev GUI
├── run_player_gui.bat          # Launcher: Player GUI (Normis Edition)
├── run.bat                     # Launcher: Direct Live Bot
├── run_rl.bat                  # Launcher: RL Lab
├── settings.bat                # Launcher: Setup Console
├── install_dependencies.bat    # Launcher: Automated dependency installation
│
├── osu2/                       # Modern v8 UNet engine & pipelines
│   ├── danser_engine.py        # Synthetic Danser rendering engine
│   ├── data.py                 # Dataset loader and sequence generator
│   ├── fast_train.py           # Two-stage freeze training loop
│   ├── live.py                 # Core live inference loop
│   ├── model.py                # UNet policy architectures & action heads
│   ├── recording_gui.py        # Comprehensive R&D recording interface
│   └── rl_lab.py               # Reinforcement learning simulation
│
├── osu_bot/                    # Lightweight client library
│   ├── capture.py              # DXGI Desktop Duplication & screen capture
│   ├── controller.py           # Win32 / vMulti input controller
│   ├── live.py                 # Standalone relax runtime
│   └── model.py                # Inference-only policy wrapper
│
├── data/
│   ├── models/                 # Neural network checkpoints (.pt)
│   ├── skin_voidveil/          # Clean minimal skin elements for rendering
│   └── lazer_manifest.json     # osu! Lazer beatmap registry
│
├── tools/                      # Utilities & database extractors
│   ├── lazer_extractor/        # .NET osu! Lazer client.realm parser
│   ├── osu_stack_processor.cs  # Hitobject stacking logic reference
│   ├── bench.py                # Latency benchmarks
│   └── watch.py                # File monitoring
│
├── tests/                      # Automated unit test suite
│   └── test_core.py            # Model and controller test cases
│
└── .github/                    # CI workflows and issue templates
```

---

## Testing

Execute the test suite locally:
```bash
python -m unittest discover tests
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
