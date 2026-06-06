[Reference document for LLM agents only. Human-facing docs live in README.md and the rest of docs/.]

# l2l-lab - Project Overview

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
│   ├── reporting/          # diagnostic CSV + Markdown snapshot writer
│   ├── utils/              # common.py, graphs.py, checkpoint.py, search.py
│   ├── agents/  backends/  envs/  neural_networks/  rllib/
│   └── configs/            # Python dataclass definitions
├── configs/                # YAML assets (training/, testing/, search/)
├── test/                   # pytest suite + tiny fixtures
├── docs/
└── models/                 # runtime checkpoints (generated)
```

After `pip install -e .`, the package imports as `l2l_lab.*` (e.g. `from l2l_lab.agents import PolicyAgent`).

## Entry Points

- `l2l-lab {train|test} [--config PATH]` - installed console script (backed by `l2l_lab.cli:main`)
- Equivalent: `python -m l2l_lab.cli {train|test} [--config PATH]`
- `l2l_lab.training.trainer.Trainer` - orchestrates training loop, metrics, checkpoints, evaluation
- `l2l_lab.testing.tester.Tester` - evaluates saved policies against opponents

## Architecture

### Backend Abstraction (`l2l_lab.backends`)

`AlgorithmBackend` ABC (`backends/base.py`) with two implementations:
- `RLlibBackend` (`backends/rllib/backend.py`) - PPO, IMPALA via Ray RLlib
- `AlphaZooBackend` (`backends/alphazoo/backend.py`) - AlphaZero (optional dep)

Key interface: `setup()`, `start_training()`, `get_eval_model()`, `get_model_from_checkpoint()`, `save_checkpoint()`

Training runs in a daemon thread pushing `StepResult` objects to a metrics queue. The main thread consumes metrics, runs evaluations, saves checkpoints, and generates plots.

### Algorithm details

#### Rllib (`l2l_lab.rllib.algorithms`)

`BaseAlgorithmTrainer` ABC (`base.py`) with implementations:
- `PPOTrainer` (`ppo.py`) - supports multi-policy training and ICM (curiosity driven exploration)
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

- `RandomAgent` - random valid action
- `PolicyAgent` - wraps a `torch.nn.Module`, runs inference with action masking. Constructor takes `obs_space_format` so it can build the right obs-to-state conversion.
- `AlphaZeroMCTSAgent` (optional - only available when `alphazoo` is installed) - runs alphazoo's network-guided MCTS on top of a trained model. Each `choose_action` call runs a fresh search tree via `alphazoo.utils.select_action_with_alphazero_mcts`. Configured via `AlphaZeroMCTSAgentConfig` with:
  - `model_name`, `checkpoint` - same shape as `PolicyAgentConfig`
  - `is_recurrent` - whether the backbone is an `AlphaZooRecurrentNet`
  - `search_config_path` - path to a YAML understood by `alphazoo.SearchConfig.from_yaml`
- `TraditionalMCTSAgent` (optional - only available when `alphazoo` is installed) - runs alphazoo's traditional MCTS (uniform priors, random rollouts to terminal - no neural network). Each `choose_action` call runs a fresh search tree via `alphazoo.utils.select_action_with_traditional_mcts`. Configured via `TraditionalMCTSAgentConfig` with only `search_config_path`.

  See [`configs/testing/testing_mcts_config.example.yml`](../configs/testing/testing_mcts_config.example.yml) and [`configs/search/default.yml`](../configs/search/default.yml) for an example.

### Environments (`l2l_lab.envs`)

Factory registry (`envs/registry.py`) with factories in `envs/factories/`:
- `pettingzoo_classic.py` - Connect Four, Tic-Tac-Toe, Leduc Hold'em
- `scs.py` - custom SCS game (external dep: RL-SCS)

All environments support action masking.

### Evaluation (`l2l_lab.training.evaluator`, `l2l_lab.configs.training.EvaluationConfig`)

`EvaluationConfig` holds two lists of eval entries:
- `training_eval` - fires every `interval` iterations during training. Allowed opponents are baselines that don't depend on a checkpoint: `random` or `traditional_mcts`.
- `checkpoint_eval` - fires at every checkpoint save. Opponent may be `random`, `traditional_mcts`, or the previous checkpoint (`opponent: policy`/`alphazero_mcts`).

Each entry declares a `player` (`policy`, `alphazero_mcts`, or `traditional_mcts`), an `opponent`, a `games_per_player` count (games played with the player as p0 *and* as p1 - total 2N), and optionally a `search_config_path` (required whenever player or opponent is an mcts kind).

`Evaluator` builds the player/opponent agents from the backend's `get_eval_model()` / `get_model_from_checkpoint()`, drives `Tester.play_games` twice per entry (to alternate positions), and aggregates into a `GameResults`. Results are stored under `metrics["evaluations"][label]` keyed by the auto-derived label (`{player}_vs_{opponent}`). Duplicate labels across both lists raise a config validation error.

Metrics and graphs are label-agnostic - `graphs.plot_metrics` iterates `metrics["evaluations"]` and renders one chart per label.

### Configuration

Python dataclasses live in `src/l2l_lab/configs/`; YAML files live at the repo root in `configs/`.

`TrainingConfig` is layered: `name` + `common: CommonConfig` + `env: EnvConfig` + `network: NetworkConfig` + `backend: {Rllib|Alphazoo}BackendConfig` + `evaluation: EvaluationConfig` + `reporting: ReportingConfig`. The backend wrapper owns `continue_training` / `continue_from_iteration` and, for alphazoo, `load_scheduler` / `load_optimizer`. The algorithm config is nested under `backend.algorithm` and holds the algorithm name plus an inner `config:` block (PPO/IMPALA: `AlgoPPOConfig` / `AlgoIMPALAConfig`; AlphaZero: the external `AlphaZooConfig`).

See [`configuration.md`](configuration.md) for the full field-by-field reference.

Testing uses a separate flat `TestingConfig` with agent configs.

### Checkpoints

Saved to `models/{name}/checkpoints/{iteration}/` containing model weights, RLlib algorithm state, and metrics history. Checkpoints can be loaded as frozen opponent policies for multi-policy training or as agents for evaluation.

### Reporting (`l2l_lab.reporting`)

Opt-in diagnostic layer, enabled by setting `reporting.enabled: true` in the training YAML. When enabled, the trainer writes LLM-friendly artifacts to `models/{name}/reports/`:

- `training.csv` - one row per iteration, appended live. Only flat scalars from `Trainer.metrics` are persisted; the header is locked at first write, so schema stays stable across resumes.
- `report_{iter:06d}.md` - full Markdown snapshot every `reporting.interval` iterations. Sections (omitted entirely when empty): header, scalar metrics with sparklines, evaluations with per-position win rates, env-registered probe states with policy distribution + value, and sample games.
- `config.yaml` - verbatim copy of the originating training YAML. On resume, if the current YAML differs structurally (canonical-YAML SHA-256 diff), an additional `config_{iter:06d}.yaml` is written - `config.yaml` is never overwritten.

**Probe states** are env-specific canonical observations registered via `l2l_lab.reporting.register_probe_states(env_name, provider)`. The provider returns `ProbeState` instances each carrying a pre-built observation dict (`{"observation", "action_mask"}`) so they're robust to non-deterministic envs. Only `connect_four` ships with probes in v1; the probes section is omitted from reports for other envs.

**Sample games** are returned directly by `Tester.play_games` via `GameResults.reports` when `reports_to_capture > 0`. `Evaluator._play_balanced` passes `reporting.sample_games_per_eval` as the capture count whenever reporting is enabled, stamps each returned `GameReport` with `(iteration, eval_label, as_position)`, and hands it to `Reporter.add_game_report`. Capture runs on both the `as_p0` and `as_p1` halves of each eval. The Reporter buffers the stamped reports and drains them into the next snapshot.

Reporting I/O runs synchronously on the trainer thread. See `src/l2l_lab/reporting/` for the implementation.

### Weights & Biases (`l2l_lab.utils.wandb`)

Opt-in cloud logging. When active, every per-iteration scalar handed to the trainer (`policy_loss`, `value_loss`, `combined_loss`, `learning_rate`, `replay_buffer_size`, `episode_len_mean`, `weight_max/min/avg`), the nested `memory/*` series, and the nested `evaluations/*` series (only on iterations they actually fire - `None` values are dropped) are streamed to a wandb run via `wandb.log(..., step=iteration)`. `wandb.init` also enables wandb's built-in system monitor, so CPU %, RAM, GPU util, GPU memory, disk, and network metrics are logged automatically every ~15s.

Configuration lives outside the training YAML, in `application.yml` at the repo root (gitignored - see `application.example.yml` for the schema):

```yaml
wandb:
  enabled: true
  api_key: "..."
  project: "l2l-lab"
  entity: null
  tags: []
```

The api key is read from this file and exported as `WANDB_API_KEY` before `wandb.init`. The `TrainingConfig` for the run is passed as the wandb run's `config` so hyperparameters are searchable in the dashboard.

Each invocation of the trainer creates a new wandb run; runs are never resumed. Every wandb run is tagged with `group=<run_name>` so all sessions of a given training name cluster together in the dashboard and can be overlaid as a single curve via wandb's "group by" controls. Scalars from earlier sessions are not re-uploaded - the trainer's own `metrics` history and the local CSV remain the canonical record of prior iterations.

Failure resilience: any failure path (missing `application.yml`, `enabled: false`, bad api key, network outage, missing dependencies) emits a single log line and disables wandb for the remainder of the run; the local CSV / Markdown / matplotlib graph sinks are unaffected.

## Dependencies

See [`pyproject.toml`](../pyproject.toml):

- Core `dependencies` - PyTorch, NumPy, PettingZoo, Gymnasium, Ray/RLlib, matplotlib, hexagdly, pyyaml, omegaconf, psutil, wandb.
- `test` / `dev` groups - pytest, py-spy.

Two optional integrations are separate local projects, not declared dependencies - install each from its local clone when needed:

- **alphazoo** - the AlphaZoo training backend and the MCTS agents.
- **RL-SCS** - the SCS environment.
