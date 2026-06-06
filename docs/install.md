# Installation

## Prerequisites

l2l-lab uses Python 3.14. The [uv](https://docs.astral.sh/uv/) package manager makes it easy to create a virtual environment with that version, independent of your system Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

### 1. Create and activate the virtual environment

```bash
uv venv --seed --python 3.14
source .venv/bin/activate
```

### 2. Install l2l-lab

On an AMD GPU, install the ROCm PyTorch build first (see [GPU acceleration](#gpu-acceleration-amd)) so the commands below don't pull the default wheel.

To use the package:

```bash
pip install .
```

For development - editable install with test deps:

```bash
pip install -e . --group dev
```

## GPU acceleration (AMD)

PyTorch is l2l-lab's compute layer, shared by both the RLlib and AlphaZoo backends. It is pulled in automatically, and on NVIDIA or CPU the default wheel is correct.

AMD GPUs need the ROCm `amdgpu` driver and runtime ([ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)) plus the matching ROCm PyTorch build, installed before l2l-lab:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

Match `rocmX.Y` to your installed ROCm version (`cat /opt/rocm/.info/version`); the [PyTorch install selector](https://pytorch.org/get-started/locally/) lists the available ROCm wheels.

## Optional dependencies

l2l-lab integrates two optional projects, each installed from a local clone.

### AlphaZoo (backend)

The AlphaZero training backend and the MCTS agents (`AlphaZeroMCTSAgent`, `TraditionalMCTSAgent`).

```bash
pip install /path/to/alphazoo
```

### RL-SCS (environment)

The SCS game environment.

```bash
pip install /path/to/RL-SCS
```


## Weights & Biases (optional)

Wandb logging is opt-in. To enable it:

1. Sign up at https://wandb.ai.
2. Copy the template and fill it in:

   ```bash
   cp application.example.yml application.yml
   ```

   Set `enabled: true`, paste your `api_key`, and set `project` / `entity` / `tags` to taste.
