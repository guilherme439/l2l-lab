# l2l-lab — Project Overview

Reinforcement learning research framework for training agents in multi-agent game environments with different algorithms.

# Summary of the current codebase state:

## Repository layout

```
l2l-lab/
├── pyproject.toml          # flit build, deps, pytest config
├── src/l2l_lab/            # the package (all Python code lives here)
│   ├── cli.py              # CLI entry point; also exposed as the `l2l-lab` script
│   ├── testing/            # tester.py (YAML-driven manual testing flow)
│   ├── training/           # trainer.py, evaluator.py
│   ├── utils/              # graphs.py, checkpoint.py
│   ├── agents/  backends/  envs/  neural_networks/  rllib/
│   └── configs/            # Python dataclass definitions
├── configs/                # YAML assets (training/, testing/, search/)
├── test/                   # pytest suite + tiny fixtures
├── docs/
└── models/                 # runtime checkpoints (generated)
```

After `pip install -e .`, the package imports as `l2l_lab.*` (e.g. `from l2l_lab.agents import PolicyAgent`).

## Entry Points

- `l2l-lab {train|test} [--config PATH]` — installed console script (backed by `l2l_lab.cli:main`)
- Equivalent: `python -m l2l_lab.cli {train|test} [--config PATH]`
- `l2l_lab.training.trainer.Trainer` — orchestrates training loop, metrics, checkpoints, evaluation
- `l2l_lab.testing.tester.Tester` — evaluates saved policies against opponents

## Architecture

### Backend Abstraction (`l2l_lab.backends`)

`AlgorithmBackend` ABC (`backends/base.py`) with two implementations:
- `RLlibBackend` (`backends/rllib/backend.py`) — PPO, IMPALA via Ray RLlib
- `AlphaZooBackend` (`backends/alphazoo/backend.py`) — AlphaZero (optional dep)

Key interface: `setup()`, `start_training()`, `get_eval_model()`, `get_model_from_checkpoint()`, `save_checkpoint()`

Training runs in a daemon thread pushing `StepResult` objects to a metrics queue. The main thread consumes metrics, runs evaluations, saves checkpoints, and generates plots.

### Algorithm details

#### Rllib (`l2l_lab.rllib.algorithms`)

`BaseAlgorithmTrainer` ABC (`base.py`) with implementations:
- `PPOTrainer` (`ppo.py`) — supports multi-policy training and ICM (curiosity driven exploration)
- `IMPALATrainer` (`impala.py`)

These build RLlib `AlgorithmConfig` objects and extract per-iteration metrics.

##### Multi-Policy Training (`rllib/algorithms/multi_policy.py`)

`PolicySampler` manages a weighted set of opponent policies: the main (trainable) policy, frozen checkpoint policies, and a random policy. Opponents are sampled per episode for training diversity.

##### Intrinsic Curiosity Module (`rllib/algorithms/icm.py`)

Optional forward/inverse model that adds intrinsic reward. Enabled via `use_curiosity` in config.


### Neural Networks (`l2l_lab.neural_networks.architectures`)

Currently, all architectures follow a dual-head pattern: shared backbone → (policy_logits, value_estimate).
- `MLPNet.py`, `ConvNet.py`, `ResNet.py`, `RecurrentNet.py`
- Building blocks in `modules/`: `blocks.py`, `policy_heads.py`, `value_heads.py`

RLlib adapters in `rllib/modules/networks/` wrap these as `RLModule` instances.

### Agents (`l2l_lab.agents`)

`Agent` ABC with `choose_action(env) → action`. Each agent queries the live PettingZoo env (observation, action mask, whatever else it needs) internally.

- `RandomAgent` — random valid action
- `PolicyAgent` — wraps a `torch.nn.Module`, runs inference with action masking. Constructor takes `obs_space_format` so it can build the right obs-to-state conversion.
- `MCTSAgent` (optional — only available when `alphazoo` is installed) — runs alphazoo's MCTS on top of a trained model. Each `choose_action` call runs a fresh search tree via `alphazoo.utils.select_action_with_mcts_for`. Configured via `MCTSAgentConfig` with:
  - `model_name`, `checkpoint` — same shape as `PolicyAgentConfig`
  - `is_recurrent` — whether the backbone is an `AlphaZooRecurrentNet`
  - `search_config_path` — path to a YAML understood by `alphazoo.SearchConfig.from_yaml`

  See [`configs/testing/testing_mcts_config.example.yml`](../configs/testing/testing_mcts_config.example.yml) and [`configs/search/default.yml`](../configs/search/default.yml) for an example.

### Environments (`l2l_lab.envs`)

Factory registry (`envs/registry.py`) with factories in `envs/factories/`:
- `pettingzoo_classic.py` — Connect Four, Tic-Tac-Toe, Leduc Hold'em
- `scs.py` — custom SCS game (external dep: RL-SCS)

All environments support action masking.

### Evaluation (`l2l_lab.training.evaluator`, `l2l_lab.configs.training.EvaluationConfig`)

`EvaluationConfig` holds two lists of eval entries:
- `training_eval` — fires every `interval` iterations during training; opponent is always `random`.
- `checkpoint_eval` — fires at every checkpoint save; can pit the current model against `random`, the previous checkpoint (`opponent: policy`/`mcts`).

Each entry declares a `player` (`policy` or `mcts`), a `games_per_player` count (games played with the player as p0 *and* as p1 — total 2N), and optionally a `search_config_path` (required for `mcts` entries).

`Evaluator` builds the player/opponent agents from the backend's `get_eval_model()` / `get_model_from_checkpoint()`, drives `Tester.play_games` twice per entry (to alternate positions), and aggregates into a `GameResults`. Results are stored under `metrics["evaluations"][label]` keyed by the auto-derived label (`{player}_vs_{opponent}`). Duplicate labels across both lists raise a config validation error.

Metrics and graphs are label-agnostic — `graphs.plot_metrics` iterates `metrics["evaluations"]` and renders one chart per label.

### Configuration

Python dataclasses live in `src/l2l_lab/configs/`; YAML files live at the repo root in `configs/`.

- `TrainingConfig` → `AlgorithmConfig` + `NetworkConfig` + `EnvConfig` + `EvaluationConfig`
- Algorithm-specific: `PPOConfig`, `IMPALAConfig`
- Testing-specific: `TestingConfig` with agent configs

### Checkpoints

Saved to `models/{name}/checkpoints/{iteration}/` containing model weights, RLlib algorithm state, and metrics history. Checkpoints can be loaded as frozen opponent policies for multi-policy training or as agents for evaluation.

## Dependency groups

See [`pyproject.toml`](../pyproject.toml):

- Core `dependencies` — PyTorch, NumPy, PettingZoo, Gymnasium, Ray/RLlib, matplotlib, hexagdly, rlcard, pyyaml.
- `test` — pytest, yappi, snakeviz.
- `alphazoo` — declares the dep for the AlphaZoo backend / `MCTSAgent`; install the package editable first from your local clone.
- `scs` — declares the dep for the SCS env; install `RL-SCS` editable first from your local clone.
