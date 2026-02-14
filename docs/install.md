## Installation

It is recommended to create a virtual environment before installing the dependencies:

**1. Create a virtual environment:**
```bash
python3 -m venv venv
```

**2. Activate the virtual environment:**

On Linux/macOS:
```bash
source venv/bin/activate
```

**3. Set up your requirements file:**

Copy the example requirements file:
```bash
cp requirements.example.txt requirements.txt
```

Then edit `requirements.txt` and update the local dependencies section.

**4. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

## Dependencies

This project depends on:
- **RL-SCS** - A reinforcement learning implementation for Standard Combat Series (SCS) game