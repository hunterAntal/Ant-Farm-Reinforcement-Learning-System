# utils.py

import numpy as np
import matplotlib.pyplot as plt

def save_q_table(Q, file_path):
    """Save the Q-table to a file."""
    np.save(file_path, Q)
    print("Saved Q-table to", file_path)

def load_q_table(file_path):
    """Load a Q-table from a file if it exists; otherwise return an empty dictionary."""
    try:
        Q = np.load(file_path, allow_pickle=True).item()
        print("Loaded Q-table from", file_path)
    except FileNotFoundError:
        print("Q-table file not found. Starting with an empty Q-table.")
        Q = {}
    return Q

def plot_training_results(rewards, steps):
    """Plot training rewards and steps per episode."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward over Time")

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode")
    plt.tight_layout()
    plt.show()
