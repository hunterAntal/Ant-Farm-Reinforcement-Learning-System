import numpy as np
import matplotlib.pyplot as plt


def plot_histogram_and_boxplot(steps):
    """
    Creates a histogram and a box plot for the provided steps data.

    Parameters:
        steps (list or array-like): Values representing steps per episode.

    Returns:
        fig (Figure): The matplotlib figure object containing the two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(steps, bins=30, color='skyblue', edgecolor='black')
    ax1.set_title("Histogram of Steps per Episode")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Frequency")

    # Box Plot
    ax2.boxplot(steps, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                medianprops=dict(color='red'))
    ax2.set_title("Box Plot of Steps per Episode")
    ax2.set_xlabel("Steps")

    plt.tight_layout()
    return fig


def plot_training_results(rewards, steps):
    """
    Creates a plot for training rewards and steps per episode.

    Parameters:
        rewards (list or array-like): Total reward per episode.
        steps (list or array-like): Number of steps per episode.

    Returns:
        fig (Figure): The matplotlib figure object containing the two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rewards, color='blue')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Episode Reward over Time")

    ax2.plot(steps, color='orange')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Steps per Episode")

    plt.tight_layout()
    return fig


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
