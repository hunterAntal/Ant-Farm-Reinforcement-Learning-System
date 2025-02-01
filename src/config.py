# config.py

# Environment settings
GRID_SIZE = 10
MAX_STEPS = 100
FIXED_GOAL = (5, 5)  # Set a fixed goal position (x, y)

# Q-learning hyperparameters
NUM_EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

# Q-table persistence
Q_TABLE_FILE = "q_table.npy"
