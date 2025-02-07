# Ant Farm Reinforcement Learning System

## Overview
Ant Farm is a **reinforcement learning (RL) environment** where an agent ("ant") learns to navigate a **2D grid** while avoiding obstacles and reaching a fixed goal. The agent improves over time using **Q-learning**, and the learned data is persisted across runs.

## Features
- **Fixed Goal Position**: The goal remains in the same location across episodes.
- **Q-learning Algorithm**: Reinforcement learning with epsilon-greedy exploration.
- **Obstacle Avoidance**: The environment includes obstacles that affect movement.
- **Pyglet-Based Rendering**: Real-time visualization with episode and step counters.
- **Q-table Persistence**: Saves and loads trained Q-values to avoid restarting learning.
- **Training Statistics**: Generates plots for training progress.

## Project Structure
```
.
├── config.py        # Configuration settings (grid size, obstacles, learning rates, etc.)
├── env.py           # Custom Gymnasium-based environment with obstacles
├── main.py          # Main script to train and run the agent
├── q_learning.py    # Q-learning implementation
├── utils.py         # Utility functions (saving/loading Q-table, plotting results)
├── q_table.npy      # Saved Q-table (persists learned policy)
```

## Installation
### Requirements
Ensure you have Python 3.8+ and install the necessary dependencies:
```bash
pip install gymnasium numpy matplotlib pyglet
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
MAX_STEPS = 25
FIXED_GOAL = (1, 1)
OBSTACLE_X = 3
OBSTACLE_Y = 3
OBSTACLE_WIDTH = 7
OBSTACLE_HEIGHT = 4
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
- **Histogram and box plot of steps**

The Q-table is saved automatically to `q_table.npy`.

## Future Enhancements
- Implement **Deep Q-Network (DQN)** for better generalization.
- Introduce **dynamic obstacles** in the environment.
- Optimize **reward shaping** for faster convergence.

---
Developed as a reinforcement learning project for AI experimentation. 🚀

