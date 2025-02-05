# config.py - Configuration Settings for Ant Farm RL System
# ---------------------------------------------------------
# This file defines all adjustable parameters for the reinforcement learning system.
# Modifying these values allows for easy experimentation without changing the core code.

# -----------------------------
# Environment Settings
# -----------------------------
GRID_SIZE = 40  # Defines the grid size (10x10 environment)
MAX_STEPS = 100  # Maximum number of steps per episode
FIXED_GOAL = (1, 1)  # Goal remains fixed at coordinate (5,5)

EPISODE_INTERVAL = 50000

OBSTACLE_X = 3  # Adjust the obstacle x
OBSTACLE_Y = 3  # Adjust the obstacle y
OBSTACLE_WIDTH = 7 # Adjust the obstacle width
OBSTACLE_HEIGHT = 4 # Adjust the obstacle height

# -----------------------------
# Q-learning Hyperparameters
# -----------------------------
NUM_EPISODES = 200000  # Total number of training episodes
ALPHA = 0.1  # Learning rate: How much new experiences overwrite old ones
GAMMA = 0.99  # Discount factor: Higher values prioritize future rewards
EPSILON = 1.0  # Initial exploration rate (100% random moves at the start)
EPSILON_DECAY = 0.999  # Decay rate: Slowly decreases exploration over time
MIN_EPSILON = 0.01  # Ensures a minimum exploration rate (1%)
REWARD_VALUE = 1000

# -----------------------------
# Q-table Persistence
# -----------------------------
Q_TABLE_FILE = "q_table.npy"  # File to store/load Q-table to retain learning across runs

