# Ant Farm Reinforcement Learning

## Overview
Ant Farm is a reinforcement learning environment where an agent ("ant") navigates a 2D grid to reach a fixed goal while learning an optimal path. The agent improves through Q-learning and retains knowledge across multiple runs using Q-table persistence.

## Features
- **Grid-based environment** with a fixed goal position
- **Q-learning algorithm** for training the agent
- **Pygame visualization** to render the learning process
- **Q-table persistence** to retain learned data across runs
- **Configurable parameters** via `config.py`

## Project Structure
```
.
├── antFarm.py       # Legacy environment file (use env.py instead)
├── config.py        # Configuration settings (grid size, learning rates, etc.)
├── env.py           # Custom Gymnasium-based environment
├── main.py          # Main script to train and run the agent
├── q_learning.py    # Q-learning implementation
├── utils.py         # Utility functions (saving/loading Q-table, plotting results)
```

## Installation
### Requirements
Ensure you have Python 3.8+ and install the necessary dependencies:
```bash
pip install gymnasium pygame numpy matplotlib
```

## Running the Simulation
To start training the agent:
```bash
python main.py
```

## Configuration
Modify `config.py` to change settings such as:
```python
GRID_SIZE = 10
MAX_STEPS = 100
FIXED_GOAL = (5, 5)
NUM_EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
```

## Training Results
After training, results are plotted to visualize performance:
- **Total reward per episode**
- **Steps per episode**

The Q-table is saved automatically to `q_table.npy`.

## Future Enhancements
- Implement **Deep Q-Network (DQN)** for better generalization.
- Introduce **dynamic obstacles** in the environment.
- Optimize **reward shaping** for faster convergence.

---
Developed as a reinforcement learning project for AI experimentation. 🚀

