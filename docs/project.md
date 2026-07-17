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
│   ├── training/           # trainer.py, evaluator.py, metrics_store.py
│   ├── reporting/          # diagnostic Markdown snapshot writer
│   ├── _utils/              # internal-only helpers: CommonUtils, GraphsUtils, CheckpointUtils, SearchUtils, WandbUtils, ...
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

Key interface: `setup()`, `start_training()`, `get_model_from_checkpoint()`, `save_checkpoint()`

Training runs in a daemon thread pushing `StepResult` objects to a metrics queue. The main thread only distributes work and collects results - it never runs evaluations or reporting inline. Evaluations run on a dedicated `EvalWorker` thread (`l2l_lab.training.eval_worker`); reporting snapshots run on the `Reporter`'s own worker thread. Both receive a CPU-resident model copy the training thread captured via the backend's private `_get_eval_model()` - the only place a model is ever copied off the training thread, keeping eval/report inference free of races with the live, in-training model. The main thread remains the single writer of the on-disk metrics store (`l2l_lab.training.metrics_store.MetricsStore`), wandb calls, and checkpoint bookkeeping; each worker reports back over its own queue, which the main thread drains without blocking.

### Iteration numbering

Iteration numbers are 0-indexed everywhere they surface: a run of `total_iterations = N` executes iterations `0` through `N - 1`. Two variables carry the count through the code:
- `current_iteration` - the 0-indexed loop index of the iteration being processed.
- `iterations_completed = current_iteration + 1` - how many iterations have finished.

Interval-gated actions (checkpoints, training/checkpoint evals, info logs, progress plots, report snapshots) fire when `iterations_completed` is a positive multiple of their interval, so with an interval of `K` the first firing is after exactly `K` iterations and every `K` thereafter.

Every persisted label - checkpoint directory names (`models/{name}/checkpoints/{current_iteration}/`), the metrics store's `iteration` column, Markdown report snapshots, and wandb steps - uses the 0-indexed `current_iteration`. A checkpoint written on the firing for interval `K` is therefore named `K - 1`, `2K - 1`, and so on; a checkpoint directory named `m` holds the model after `m + 1` completed iterations. For the alphazoo backend, the `iteration` in `algo/metadata.json` is stamped with the same `current_iteration` as the directory name (passed explicitly to `AlphaZoo.save`).

On resume, `restore` returns the directory name `m` it loaded from, and training continues at `starting_iteration = m + 1`. A run whose latest checkpoint equals `total_iterations - 1` is complete, and resuming it does nothing.

`AlgorithmBackend.restore` walks the checkpoint directories at or below `continue_from_iteration` from highest to lowest and loads the first that loads without error, returning its iteration (0 when none load). A save interrupted mid-write leaves a numbered directory missing the artifacts the backend restores from, so its load raises and the walk falls through to an earlier checkpoint. `is_rewind` then compares the loaded iteration against the highest checkpoint directory on disk: loading anything behind that highest directory is a rewind, which prompts and then deletes every checkpoint and report artifact past the loaded iteration - including a partial directory the load skipped.

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

`Evaluator` builds the player/opponent agents from a model snapshot handed to it (the `eval_model` on `StepResult`, captured by the backend once per snapshot-worthy step) and from `get_model_from_checkpoint()`, drives `Tester.play_games` twice per entry (to alternate positions), and aggregates into a `GameResults`. `Evaluator` runs on the `EvalWorker` thread rather than the main thread; its results travel back as an `EvalResult` and get recorded into the metrics store (`MetricsStore.record_eval`, keyed by the auto-derived label `{player}_vs_{opponent}`) once merged on the main thread. Duplicate labels across both lists raise a config validation error.

Metrics and graphs are label-agnostic - `GraphsUtils.plot_evaluations` iterates the metrics view's evaluation series and renders one chart per label.

### Configuration

Python dataclasses live in `src/l2l_lab/configs/`; YAML files live at the repo root in `configs/`. `TrainingConfig.from_yaml` / `TestingConfig.from_yaml` load the YAML to a plain dict with OmegaConf, then build the dataclass tree through a Pydantic `TypeAdapter` (`from_dict`), which validates fields and dispatches the polymorphic sections.

Polymorphic configs are Pydantic **discriminated unions** - a `BaseXConfig` plus `Literal`-tagged variant dataclasses, exposed as a union alias and resolved on a discriminator field:
- `NetworkConfig` on `architecture` (`ResNetConfig`, `ConvNetConfig`, `RecurrentNetConfig`, `MLPNetConfig`, `SNNetConfig`); conv-based nets carry a `PolicyHeadConfig` / `ValueHeadConfig`, each a union on `name`.
- `BackendConfig` on `name` (`RllibBackendConfig`, `AlphazooBackendConfig`).
- `AlgorithmConfig` on `name` (`PPOAlgorithmConfig`, `IMPALAAlgorithmConfig`, `AlphazooAlgorithmConfig`); the rllib variants share an `iterations` count and carry their inner hyperparameter config (`AlgoPPOConfig` / `AlgoIMPALAConfig`).
- `AgentConfig` on `agent_type` (`PolicyAgentConfig`, `RandomAgentConfig`, `AlphaZeroMCTSAgentConfig`, `TraditionalMCTSAgentConfig`).

`TrainingConfig` is layered: `name` + `common` + `env` + `network` + `backend` + `evaluation` + `reporting`. The backend owns `continue_training` / `continue_from_iteration` and, for alphazoo, `load_scheduler` / `load_optimizer`. `AlphazooAlgorithmConfig.config` holds the external `AlphaZooConfig`, built through `AlphaZooConfig.from_dict` so its own OmegaConf merge runs. `BaseNetworkConfig.from_dict` rebuilds a network config from the dict RLlib round-trips through `model_config`.

See [`configuration.md`](configuration.md) for the full field-by-field reference. Testing uses a flat `TestingConfig` (`p1`, `p2`, `env`, `num_games`) whose `p1` / `p2` are `AgentConfig`s.

### Checkpoints

Saved to `models/{name}/checkpoints/{iteration}/`. Checkpoints can be loaded as frozen opponent policies for multi-policy training or as agents for evaluation. Metrics history lives separately under `models/{name}/metrics/` (see Metrics persistence).

Checkpoints are written synchronously on the training thread, between steps (inside the alphazoo step callback, or inline in the rllib train loop), through each backend's `_write_checkpoint`. Every file in a checkpoint therefore captures the state of exactly the iteration the directory is named after; nothing mutates the model, optimizer, or replay buffer while the save runs.

Each checkpoint holds a `weights.pt` model state dict at its root plus an `algo/` subdirectory of backend resume state. Pairing `weights.pt` with the run-level architecture pickle `models/{name}/network_template.pkl` (written once at setup when checkpointing is configured) reconstructs a standalone model via `get_model_from_checkpoint` - for evaluation or for handing a trained net to someone else, independent of any backend's resume state. `CheckpointUtils.get_network_template_path` resolves that run-level pickle from a checkpoint directory. (The pickle-based architecture format is a stopgap; moving it to a portable format such as safetensors or TorchScript is a planned l2l-lab improvement.)

The `algo/` subdirectory holds each backend's own resume state: the RLlib algorithm state (`algo.save_to_path`) for the rllib backend, or the `metadata.json` / `optimizer.ckpt` / `scheduler.ckpt` / `replay_buffer.ckpt` that `AlphaZoo.save(save_model=False, iteration=...)` produces. Those resume files exist only to continue training in place.

Individual files are written atomically by `CheckpointUtils.atomic_write` (`l2l_lab._utils.checkpoint`): it serializes into l2l-lab's own temp directory (`$XDG_CACHE_HOME/l2l_lab/tmp`, else `~/.cache/l2l_lab/tmp`) and then `os.replace`s the result onto its destination. The rename is atomic when the temp directory and the destination share a filesystem; when they do not, `os.replace` raises `EXDEV` and the write falls back to a non-atomic move with a warning. The checkpoint directory as a whole is not atomic - it is filled file by file, so an interrupted save leaves a directory holding some complete files and missing others, which `restore` handles by loading an earlier checkpoint. The metrics store's `scalars.csv` / `evals.csv` are flushed per append; the `alphazoo` package writes its own resume files atomically through the same temp-then-rename scheme under `~/.cache/alphazoo/tmp`.

### Metrics persistence (`l2l_lab.training.metrics_store`)

`MetricsStore` streams metrics to `models/{name}/metrics/` so the full history never accumulates in memory. Dense per-iteration scalars append to `scalars.csv` (one row per iteration, nested groups such as `memory` flattened to `parent.child` columns); sparse evaluation results append to `evals.csv` (one row per iteration/label/position) and are mirrored in a compact in-memory structure. The main thread is the store's only writer.

Readers call `load_view` to reconstruct a `MetricsView` on demand: it streams `scalars.csv`, optionally downsampling to at most `_PLOT_MAX_POINTS` evenly-spaced points for plotting or keeping only a trailing window for a report snapshot, and pairs the scalars with the sparse evaluation points. Peak reader memory stays bounded regardless of run length. On resume the store truncates both files to the loaded iteration (dropping any rows a crash left past the last checkpoint) and reloads the evaluation points; a rewind truncates the same way. `GraphsUtils.plot_metrics` and the Markdown reporter both consume a `MetricsView` rather than a live dict, and tests read the history through `Trainer.load_metrics()`.

### Reporting (`l2l_lab.reporting`)

Opt-in diagnostic layer, enabled by setting `reporting.enabled: true` in the training YAML. When enabled, the trainer writes LLM-friendly artifacts to `models/{name}/reports/`:

- `report_{iter:06d}.md` - full Markdown snapshot every `reporting.interval` iterations. Sections (omitted entirely when empty): header, scalar metrics with sparklines, evaluations with per-position win rates, env-registered probe states with policy distribution + value, and sample games. Scalar and evaluation data are read from the metrics store (see Metrics persistence), the single record of training metrics.
- `configs/config.yaml` - verbatim copy of the originating training YAML. On resume, if the current YAML differs structurally (canonical-YAML SHA-256 diff), an additional `configs/config_{iter:06d}.yaml` is written - `config.yaml` is never overwritten.

**Probe states** are env-specific canonical observations registered via `l2l_lab.reporting.register_probe_states(env_name, provider)`. The provider returns `ProbeState` instances each carrying a pre-built observation dict (`{"observation", "action_mask"}`) so they're robust to non-deterministic envs. Only `connect_four` ships with probes in v1; the probes section is omitted from reports for other envs.

**Sample games** are returned directly by `Tester.play_games` via `GameResults.reports` when `reports_to_capture > 0`; `Evaluator` is constructed with `reporting.sample_games_per_eval` as that capture count whenever reporting is enabled. Capture runs on both the `as_p0` and `as_p1` halves of each eval, riding back on the `EvalResult` the `EvalWorker` returns. The trainer stamps each captured `GameReport` with `(iteration, eval_label, as_position)` and hands it to `Reporter.add_game_report`, which buffers it for the next snapshot.

The Reporter runs its own worker thread for all file I/O and probe-state inference, so `emit_snapshot` (a Markdown snapshot) only enqueues work from the main thread and never blocks it. The trainer calls `emit_snapshot` once every eval in flight as of that snapshot's iteration has completed, passing a `MetricsView` the store built for that iteration (trailing window of scalars plus evaluation points up to it) and the model snapshot it captured for that step - the same snapshot mechanism the `EvalWorker` consumes. See `src/l2l_lab/reporting/` for the implementation.

### Weights & Biases (`l2l_lab._utils.wandb`)

Opt-in cloud logging. When active, every per-iteration scalar handed to the trainer (`policy_loss`, `value_loss`, `combined_loss`, `learning_rate`, `replay_buffer_size`, `episode_len_mean`, `weight_max/min/avg`) and the nested `memory/*` series are streamed to a wandb run via `wandb.log(..., step=iteration)`. The nested `evaluations/*` series logs separately, through `log_evaluations(..., iteration)`, against a custom `eval_iteration` axis (`run.define_metric("evaluations/*", step_metric="eval_iteration")`, set up in `init`) rather than the training step: evaluations complete asynchronously and can land behind steps already logged, which `wandb.log(..., step=...)` would otherwise silently drop. `None` values are dropped before either call. `wandb.init` also enables wandb's built-in system monitor, so CPU %, RAM, GPU util, GPU memory, disk, and network metrics are logged automatically every ~15s.

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

Each invocation of the trainer creates a new wandb run; runs are never resumed. Every wandb run is tagged with `group=<run_name>` so all sessions of a given training name cluster together in the dashboard and can be overlaid as a single curve via wandb's "group by" controls. Scalars from earlier sessions are not re-uploaded - the metrics store under `models/{name}/metrics/` remains the canonical record of prior iterations.

Failure resilience: any failure path (missing `application.yml`, `enabled: false`, bad api key, network outage, missing dependencies) emits a single log line and disables wandb for the remainder of the run; the metrics store / Markdown / matplotlib graph sinks are unaffected.

## Dependencies

See [`pyproject.toml`](../pyproject.toml):

- Core `dependencies` - PyTorch, NumPy, PettingZoo, Gymnasium, Ray/RLlib, matplotlib, hexagdly, pyyaml, omegaconf, psutil, wandb.
- `test` / `dev` groups - pytest, py-spy.

Two optional integrations are separate local projects, not declared dependencies - install each from its local clone when needed:

- **alphazoo** - the AlphaZoo training backend and the MCTS agents.
- **RL-SCS** - the SCS environment.
