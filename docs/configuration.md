# Training Configuration Reference

All training runs are driven by a single YAML file parsed into `TrainingConfig`. Each top-level key maps to a dataclass in [`src/l2l_lab/configs/training/`](../src/l2l_lab/configs/training/). Most fields have defaults — you only specify what you want to override.

Full examples live under [`configs/training/*.example.yml`](../configs/training/).

## Table of Contents

- [Config Tree](#config-tree)
- [Top-Level](#top-level)
- [Common](#common)
- [Env](#env)
- [Network](#network)
- [Backend](#backend)
  - [Rllib Backend](#rllib-backend)
  - [Alphazoo Backend](#alphazoo-backend)
- [Algorithm](#algorithm)
  - [PPO](#ppo)
  - [IMPALA](#impala)
  - [AlphaZero](#alphazero)
- [Evaluation](#evaluation)
- [Reporting](#reporting)

---

## Config Tree

```
TrainingConfig
├── name
├── common: CommonConfig
│   ├── plot_interval
│   ├── info_interval
│   ├── eval_graph_split
│   ├── checkpoint_interval
│   └── plot_memory
├── env: EnvConfig
│   ├── name
│   ├── obs_space_format
│   └── kwargs
├── network: NetworkConfig
│   ├── architecture
│   └── <arch-specific kwargs>
├── backend: {Rllib|Alphazoo}BackendConfig
│   ├── name
│   ├── continue_training
│   ├── continue_from_iteration
│   ├── load_scheduler              # alphazoo only
│   ├── load_optimizer              # alphazoo only
│   └── algorithm: {Rllib|Alphazoo}AlgorithmConfig
│       ├── name
│       ├── iterations              # rllib only
│       └── config: <algo-specific>
├── evaluation: EvaluationConfig
│   ├── training_eval: list
│   └── checkpoint_eval: list
└── reporting: ReportingConfig
    ├── enabled
    ├── interval
    └── sample_games_per_eval
```

---

## Top-Level

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | *(required)* | Run name. Also the directory under `models/` where checkpoints and graphs land. |

All other top-level keys are nested sections described below.

---

## Common

Interval knobs. Setting any to `0` disables the corresponding action.

| Field | Type | Default | Description |
|---|---|---|---|
| `plot_interval` | int | `0` | Save progress plots every N iterations. `0` disables. |
| `info_interval` | int | `0` | Print the backend's periodic training-info block every N iterations. `0` disables. |
| `eval_graph_split` | int | `0` | When plotting evaluation results, split the time axis into chunks of this many iterations. `0` keeps a single chart. |
| `checkpoint_interval` | int | `0` | Save a checkpoint every N iterations. `0` disables and the trainer prints a warning at startup. |
| `plot_memory` | bool | `false` | When `true`, sample RAM usage once per iteration and emit a `memory.png` graph alongside the other plots. Tracks system-wide used memory plus the trainer's whole process tree (main + Ray/AlphaZoo workers via PSS, falling back to RSS off Linux). Memory data is excluded from the reporting CSV/Markdown snapshots. |

The `common:` section itself is required — parsing raises if it's missing, even though every field inside has a default.

---

## Env

Environment selection and construction kwargs. `EnvConfig` is defined in [`common/EnvConfig.py`](../src/l2l_lab/configs/common/EnvConfig.py).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | *(required)* | Environment name (e.g. `"tictactoe"`, `"connect_four"`, `"scs"`). Resolved by [`envs/registry.py`](../src/l2l_lab/envs/registry.py). |
| `obs_space_format` | `"channels_first" \| "channels_last" \| "flat"` | `"channels_first"` | Layout of the observation tensor. |
| `kwargs` | dict | `{}` | Passed straight to the env constructor. |

---

## Network

Neural network architecture + its construction kwargs. `NetworkConfig` is defined in [`training/NetworkConfig.py`](../src/l2l_lab/configs/training/NetworkConfig.py).

| Field | Type | Default | Description |
|---|---|---|---|
| `architecture` | `"ResNet" \| "ConvNet" \| "MLPNet" \| "RecurrentNet"` | *(required)* | Which architecture class to instantiate. |
| *(remaining keys)* | varies | — | Architecture-specific kwargs (e.g. `num_filters`, `num_blocks`, `hidden_layers`, `neurons_per_layer`). Any unknown key under `network:` is treated as a constructor kwarg. |

---

## Backend

Picks which training backend runs the loop and holds backend-specific knobs around resumption and state loading.

Shared fields on every backend:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `"rllib" \| "alphazoo"` | *(required)* | Dispatch key — selects which backend class is used. |
| `continue_training` | bool | `false` | When `true`, the run resumes from an existing checkpoint at `models/<name>/` instead of starting fresh. |
| `continue_from_iteration` | int \| null | `null` | Specific iteration to resume from. `null` = latest checkpoint. |
| `algorithm` | object | *(required)* | See [Algorithm](#algorithm) below. Its shape must match the backend. |

### Rllib Backend

No extra fields beyond the shared ones above. `continue_training: true` here restores the full Learner state (weights + optimizer + Learner step counter). Changing `lr` in the config between runs is silently ignored when continuing — use a piecewise `lr: [[step, value], ...]` schedule if you need mid-training LR changes.

### Alphazoo Backend

Adds fine-grained controls over what gets restored on resume:

| Field | Type | Default | Description |
|---|---|---|---|
| `load_scheduler` | bool \| null | `null` → `continue_training` | When `false`, the LR scheduler is rebuilt fresh from config (resetting the schedule). |
| `load_optimizer` | bool \| null | `null` → `continue_training` | When `false`, the optimizer is rebuilt fresh from config (losing Adam/SGD internal stats). |

Setting either to `true` when `continue_training` is `false` is an error — there's no state to load from.

When `load_optimizer: true` **and** `load_scheduler: false` (with `continue_training: true`), the optimizer's per-param-group `lr` is synced to the fresh scheduler's `starting_lr`. This is how you bump the LR back up on resume while keeping Adam m/v stats.

---

## Algorithm

### PPO

Used with `backend.name: "rllib"` and `algorithm.name: "ppo"`. The wrapper holds `name` + `iterations`; the `config:` block is parsed into [`AlgoPPOConfig`](../src/l2l_lab/configs/training/algorithms/AlgoPPOConfig.py).

| Field | Type | Default | Description |
|---|---|---|---|
| `iterations` | int | *(required)* | Total training iterations. |
| `config.use_curiosity` | bool | `false` | Enable ICM curiosity module. |
| `config.curiosity_coeff` | float | `0.05` | Curiosity reward weight (when `use_curiosity` is on). |
| `config.num_env_runners` | int | `0` | RLlib env-runner workers. `0` = run in the main process. |
| `config.rollout_fragment_length` | int | `64` | Per-worker rollout fragment length. |
| `config.train_batch_size_per_learner` | int | `512` | Training batch size per learner. |
| `config.minibatch_size` | int | `64` | SGD minibatch size within each epoch. |
| `config.num_epochs` | int | `3` | SGD epochs per training iteration. |
| `config.lr` | float \| list[[int, float]] | `0.0001` | Learning rate. Scalar = constant; list = `[[step, value], ...]` piecewise schedule. |
| `config.gamma` | float | `0.99` | Discount factor. |
| `config.entropy_coeff` | float | `0.05` | Entropy bonus coefficient. |
| `config.vf_loss_coeff` | float | `0.5` | Value-function loss coefficient. |
| `config.clip_param` | float | `0.2` | PPO clip parameter. |
| `config.use_kl_loss` | bool | `true` | Add a KL penalty to the loss. |
| `config.kl_coeff` | float | `0.2` | Initial KL coefficient. |
| `config.kl_target` | float | `0.01` | KL target used to adapt `kl_coeff`. |
| `config.policy` | [PolicyConfig](../src/l2l_lab/configs/training/PolicyConfig.py) \| null | `null` | Multi-policy self-play configuration. See the class for fields (`use_multiple_policies`, `number_previous_policies`, `main_policy_ratio`, `random_policy_ratio`). |

### IMPALA

Used with `backend.name: "rllib"` and `algorithm.name: "impala"`. Inner config: [`AlgoIMPALAConfig`](../src/l2l_lab/configs/training/algorithms/AlgoIMPALAConfig.py).

| Field | Type | Default | Description |
|---|---|---|---|
| `iterations` | int | *(required)* | Total training iterations. |
| `config.num_env_runners` | int | `2` | RLlib env-runner workers. |
| `config.rollout_fragment_length` | int | `50` | Per-worker rollout fragment length. |
| `config.num_learners` | int | `1` | IMPALA learner workers. |
| `config.lr` | float | `0.0005` | Learning rate. |
| `config.gamma` | float | `0.99` | Discount factor. |
| `config.entropy_coeff` | float | `0.01` | Entropy bonus coefficient. |
| `config.vf_loss_coeff` | float | `0.5` | Value-function loss coefficient. |
| `config.grad_clip` | float | `40.0` | Gradient clip norm. |
| `config.vtrace` | bool | `true` | Enable V-trace off-policy correction. |
| `config.vtrace_clip_rho_threshold` | float | `1.0` | V-trace ρ̄ clip. |
| `config.vtrace_clip_pg_rho_threshold` | float | `1.0` | V-trace policy-gradient ρ̄ clip. |

### AlphaZero

Used with `backend.name: "alphazoo"` and `algorithm.name: "alphazero"`. The inner `config:` block is the full external [`AlphaZooConfig`](https://github.com/guilherme439/alphazoo/blob/main/src/alphazoo/configs/alphazoo_config.py) from the alphazoo package — **refer to [alphazoo's configuration reference](https://github.com/guilherme439/alphazoo/blob/main/docs/configuration.md) for every field under `config:`**.

Differences from `AlphaZooConfig` standalone:

- There is no `iterations` field on `AlphazooAlgorithmConfig`. The total iteration count comes from `config.running.training_steps`.
- `config.data.observation_format` and `config.data.network_input_format` are overwritten by the l2l-lab env config at runtime.

---

## Evaluation

See [`EvaluationConfig.py`](../src/l2l_lab/configs/training/EvaluationConfig.py). Two lists of entries control how the trainer evaluates the current model during training.

### `evaluation.training_eval`

Fires on fixed iteration intervals. Current model plays vs `random`.

| Field | Type | Default | Description |
|---|---|---|---|
| `player` | `"policy" \| "mcts"` | *(required)* | Whether to evaluate the raw policy or a search-augmented agent. |
| `games_per_player` | int | *(required)* | Games as P0 *and* as P1 (total = 2× this). |
| `interval` | int | *(required)* | Fire every N iterations. |
| `search_config_path` | str \| null | `null` | Required when `player: mcts`. Path to a search YAML. |

### `evaluation.checkpoint_eval`

Fires on every checkpoint save. Current model plays vs `random`, the previous checkpoint's policy, or an MCTS-wrapped version of it.

| Field | Type | Default | Description |
|---|---|---|---|
| `player` | `"policy" \| "mcts"` | *(required)* | See above. |
| `opponent` | `"random" \| "policy" \| "mcts"` | *(required)* | Opponent type. `policy`/`mcts` use the most recent saved checkpoint. |
| `games_per_player` | int | *(required)* | Games as P0 *and* as P1. |
| `search_config_path` | str \| null | `null` | Required when `player: mcts` or `opponent: mcts`. |

Labels are auto-derived from `{player}_vs_{opponent}` (or `{player}_vs_random` for training entries). Duplicate labels across the two lists are a validation error.

---

## Reporting

Optional diagnostic layer. When enabled, writes structured artifacts to `models/<name>/reports/` for debugging training runs. Defined in [`training/ReportingConfig.py`](../src/l2l_lab/configs/training/ReportingConfig.py). Omitting the section is the same as `enabled: false`.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. When `false`, nothing is written. |
| `interval` | int | `100` | Snapshot every N iterations. The CSV row-per-iteration writes regardless. |
| `sample_games_per_eval` | int | `2` | Sample games captured per evaluation call, per position (`as_p0` and `as_p1`). |

### Output files

- `training.csv` — one row per iteration, flat scalars only. Header is locked at first write.
- `report_{iter:06d}.md` — full snapshot every `interval` iterations. Older snapshots are kept so you can diff progress.
- `config.yaml` — the originating training YAML, copied once at setup.
- `config_{iter:06d}.yaml` — only if the YAML structurally changed since the last run.

### Probe states

Snapshots include a **Probe states** section: fixed canonical observations fed through the current model to show its policy + value on known positions. Probe states are env-specific and registered in code, not YAML — only `connect_four` ships with built-in probes. For other envs the section is omitted. To add coverage, see [`reporting/probe_states.py`](../src/l2l_lab/reporting/probe_states.py).

Example block:

```yaml
reporting:
  enabled: true
  interval: 100
  sample_games_per_eval: 2
```
