# Installation

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# (Optional, if you want the AlphaZoo backend) install alphazoo editable first
pip install -e /path/to/alphazoo

# (Optional, if you want the SCS env) install RL-SCS editable first
pip install -e /path/to/RL-SCS

# Install l2l-lab itself
pip install -e .

# To also pull in the optional groups:
pip install -e . --group test      # pytest, yappi, snakeviz
pip install -e . --group alphazoo  # marks alphazoo as a declared dep
pip install -e . --group scs       # marks RL-SCS as a declared dep
```

## Running tests

```bash
pytest -v
```

## Dependencies

l2l-lab hard-depends on:

- **PyTorch**, **NumPy**, **PettingZoo**, **Gymnasium** — core ML / env stack
- **Ray (rllib)** — one of the training backends
- **matplotlib**, **hexagdly**, **rlcard**, **pyyaml** — various utilities

Optional:

- **alphazoo** — required for the AlphaZoo training backend and the `MCTSAgent`
- **RL-SCS** — required if you want to use the SCS environment
