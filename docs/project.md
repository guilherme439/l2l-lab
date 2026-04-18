# l2l-lab — Project Overview

Reinforcement learning research framework for training agents in multi-agent game environments with different algorithms.

# Summary of the current codebase state:

## Entry Points

- `main.py` — CLI: `python main.py {train|test} [--config PATH]`
- Training: `Trainer.py` — orchestrates training loop, metrics, checkpoints, evaluation
- Testing: `Tester.py` — evaluates saved policies against opponents

## Architecture

### Backend Abstraction (`backends/`)

`AlgorithmBackend` ABC (`backends/base.py`) with two implementations:
- `RLlibBackend` (`backends/rllib/backend.py`) — PPO, IMPALA via Ray RLlib
- `AlphaZooBackend` (`backends/alphazoo/backend.py`) — AlphaZero (optional dep)

Key interface: `setup()`, `start_training()`, `create_eval_agent()`, `save_checkpoint()`

Training runs in a daemon thread pushing `StepResult` objects to a metrics queue. The main thread consumes metrics, runs evaluations, saves checkpoints, and generates plots.

### Algorithm details

#### Rllib (`rllib/algorithms/`)

`BaseAlgorithmTrainer` ABC (`base.py`) with implementations:
- `PPOTrainer` (`ppo.py`) — supports multi-policy training and ICM (curiosity driven exploration)
- `IMPALATrainer` (`impala.py`)

These build RLlib `AlgorithmConfig` objects and extract per-iteration metrics.

##### Multi-Policy Training (`rllib/algorithms/multi_policy.py`)

`PolicySampler` manages a weighted set of opponent policies: the main (trainable) policy, frozen checkpoint policies, and a random policy. Opponents are sampled per episode for training diversity.

##### Intrinsic Curiosity Module (`rllib/algorithms/icm.py`)

Optional forward/inverse model that adds intrinsic reward. Enabled via `use_curiosity` in config.


### Neural Networks (`neural_networks/architectures/`)

Currently, all architectures follow a dual-head pattern: shared backbone → (policy_logits, value_estimate).
- `MLPNet.py`, `ConvNet.py`, `ResNet.py`, `RecurrentNet.py`
- Building blocks in `modules/`: `blocks.py`, `policy_heads.py`, `value_heads.py`

RLlib adapters in `rllib/modules/networks/` wrap these as `RLModule` instances.

### Agents (`agents/`)

`Agent` ABC with `choose_action(env) → action`. Each agent queries the live PettingZoo env (observation, action mask, whatever else it needs) internally.

- `RandomAgent` — random valid action
- `PolicyAgent` — wraps a `torch.nn.Module`, runs inference with action masking. Constructor takes `obs_space_format` so it can build the right obs-to-state conversion.
- `MCTSAgent` (optional — only available when `alphazoo` is installed) — runs alphazoo's MCTS on top of a trained model. Each `choose_action` call runs a fresh search tree via `alphazoo.Explorer.select_action_with_mcts_for`. Configured via `MCTSAgentConfig` with:
  - `model_name`, `checkpoint` — same shape as `PolicyAgentConfig`
  - `is_recurrent` — whether the backbone is an `AlphaZooRecurrentNet`
  - `search_config_path` — path to a YAML understood by `alphazoo.SearchConfig.from_yaml`

  See [`configs/files/testing/testing_mcts_config.example.yml`](../configs/files/testing/testing_mcts_config.example.yml) and [`configs/files/search/default.yml`](../configs/files/search/default.yml) for an example.

### Environments (`envs/`)

Factory registry (`envs/registry.py`) with factories in `envs/factories/`:
- `pettingzoo_classic.py` — Connect Four, Tic-Tac-Toe, Leduc Hold'em
- `scs.py` — custom SCS game (external dep: RL-SCS)

All environments support action masking.

### Configuration (`configs/`)

YAML files (`configs/files/`) deserialized into dataclass hierarchy (`configs/definition/`):
- `TrainingConfig` → `AlgorithmConfig` + `NetworkConfig` + `EnvConfig`
- Algorithm-specific: `PPOConfig`, `IMPALAConfig`
- Testing-specific: `TestingConfig` with agent configs

### Checkpoints

Saved to `models/{name}/checkpoints/{iteration}/` containing model weights, RLlib algorithm state, and metrics history. Checkpoints can be loaded as frozen opponent policies for multi-policy training or as agents for evaluation.

## Key Dependencies

- **Ray/RLlib** (2.50+): distributed training, algorithm implementations
- **PyTorch**: neural networks
- **PettingZoo**: multi-agent game environments
- **Optional**: `alphazoo` (AlphaZero), `RL-SCS` (SCS game), `hexagdly` (hex convolutions)
