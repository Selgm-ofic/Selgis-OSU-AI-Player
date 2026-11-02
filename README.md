<p align="center">
  <img src="repo%20assets/logo.png" alt="Selgis osu! AI Player" width="420" />
</p>

# Selgis osu! AI Player

[![CI](https://github.com/Selgm-ofic/Selgis-OSU-AI-Player/actions/workflows/ci.yml/badge.svg)](https://github.com/Selgm-ofic/Selgis-OSU-AI-Player/actions/workflows/ci.yml)
[![Powered by Selgis-ML](https://img.shields.io/badge/Powered%20by-Selgis--ML-7B2CBF.svg)](https://github.com/Selgm-ofic/Selgis-ML)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Comprehensive deep-learning visual agent, training pipeline, and live inference assistant for osu! based on spatial-temporal UNet and recurrent action prediction architectures.

The policy training pipeline, numerical stability protection (NaN/Inf guardrails, loss spike rollback), convergence telemetry, and adaptive learning rate scheduling are built natively on top of the **[Selgis-ML Framework](https://github.com/Selgm-ofic/Selgis-ML)**.

> [!WARNING]
> **Experimental Research Codebase**: Many features, pipelines, and experimental modules in this repository were created and tested strictly in local development environments. Certain functionalities may not work, may be incomplete, or may operate unexpectedly under different hardware setups, screen resolutions, display scaling, or game client configurations.

---

## Key Features

- **Powered by Selgis-ML**: Native integration with the [Selgis-ML](https://github.com/Selgm-ofic/Selgis-ML) training engine, providing automated gradient unscaling safety, loss spike detection, state rollback, and mixed-precision (FP16 AMP) optimization.
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
- Python 3.10, 3.11, or 3.12 (ensure "Add python.exe to PATH" is checked)
- NVIDIA GPU with CUDA support recommended (GTX 1060+ or RTX series)

### 2. Dependency Installation

#### Option A: Automated One-Click Installer (Recommended)
Run the automated environment installer:
```bat
install_dependencies.bat
```
This script detects your GPU, installs PyTorch with CUDA hardware acceleration, configures all core libraries from `requirements.txt`, and installs the [Selgis-ML](https://github.com/Selgm-ofic/Selgis-ML) framework.

#### Option B: Manual Installation via pip
```bash
# 1. Install PyTorch with CUDA hardware acceleration (CUDA 12.1 / 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install core dependencies from requirements.txt
pip install -r requirements.txt

# 3. Alternatively, install Selgis-ML directly from GitHub:
pip install git+https://github.com/Selgm-ofic/Selgis-ML.git
```

### 3. Launching Interfaces

Choose your preferred interface:
- **Player GUI (Normis Edition)**: Double-click `run_player_gui.bat` (or `python normis_gui.py`)
- **Research & Recording GUI**: Double-click `run_gui.bat` (or `python main.py gui`)
- **Interactive CLI Setup**: Double-click `settings.bat` (or `python run_live.py --settings`)
- **Direct Live Bot**: Double-click `run.bat` (or `python run_live.py`)

---

## Training Policies with Selgis-ML

Training utilizes the unified `main.py` entry point with the `--engine selgis` backend (default):

```bash
# Train UNet v7.2 policy with Selgis-ML engine
python main.py train --engine selgis --version v7_2 --dataset data/dataset_v8_combined_subset --epochs 40 --batch 128

# Train UNet policy with Two-Stage Freeze transfer learning
python main.py train --engine selgis --version v8_5 --method two_stage_freeze --epochs 40 --batch 128

# Train on custom dataset with explicit learning rate and fp16
python main.py train --engine selgis --version v7_2 --dataset data/dataset_v8 --lr 0.001 --batch 128
```

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

The unified CLI entry point provides complete control over data synthesis, training, and benchmarking:

```bash
# Launch Dev GUI
python main.py gui

# Launch Reinforcement Learning Lab
python main.py rl --episodes 2500

# Benchmark inference latency and mechanics
python main.py bench --weights data/models/unet_policy_v7_2.pt

# Evaluate accuracy against validation dataset
python main.py evaluate --weights data/models/unet_policy_v7_2.pt --dataset data/dataset_v8_combined_subset

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
├── requirements.txt            # Python dependencies (including Selgis-ML)
├── AGENTS.md                   # AI Agent guidance and skill reference
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
│
├── skills/                     # Agent skills and operational specifications
│   └── selgis-osu-ai/          # AI Agent skill for Selgis osu! Player
│       └── SKILL.md            # Tensor contracts, schemas, and workflows
│
├── repo assets/                # Visual repository assets
│   └── logo.png                # Official project logo banner
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
│   ├── rl_lab.py               # Reinforcement learning simulation
│   └── selgis_train.py         # Selgis-ML Trainer integration
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
│   ├── test_core.py            # Model and controller test cases
│   └── test_selgis_train.py    # Selgis-ML integration test cases
│
└── .github/                    # CI workflows and issue templates
```

---

## Testing

Execute the automated test suite locally:
```bash
python -m unittest discover tests
```

---

## AI Agents & Automation

This repository includes a dedicated operational skill for AI coding assistants and autonomous agents. The skill provides explicit 5D tensor input/output specifications, checkpoint dictionary schemas, training protocols, and pre-commit verification checklists:

- **Agent Skill**: [`skills/selgis-osu-ai/SKILL.md`](skills/selgis-osu-ai/SKILL.md)
- **Agent Entrypoint**: [`AGENTS.md`](AGENTS.md)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
