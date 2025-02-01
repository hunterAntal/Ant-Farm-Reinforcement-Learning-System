# main.py

import os
import numpy as np
import pygame
from env import AntFarmEnv
from q_learning import q_learning
from utils import save_q_table, load_q_table, plot_training_results
import config

def main():
    # Create the environment with settings from config.py.
    env = AntFarmEnv(grid_size=config.GRID_SIZE, max_steps=config.MAX_STEPS, fixed_goal=config.FIXED_GOAL)

    # Load an existing Q-table if available.
    Q = load_q_table(config.Q_TABLE_FILE)

    # Train the agent using Q-learning.
    Q, rewards, steps = q_learning(
        env,
        Q=Q,
        num_episodes=config.NUM_EPISODES,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON
    )

    # Save the updated Q-table.
    save_q_table(Q, config.Q_TABLE_FILE)

    # Plot training results.
    plot_training_results(rewards, steps)

    # Run a demonstration episode with rendering.
    state, _ = env.reset()
    done = False
    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            env.render(episode=episode)  # Pass episode number to render function
            state_key = tuple(state)

            if state_key not in Q:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_key])

            state, reward, done, truncated, info = env.step(action)
            pygame.time.wait(2)

    env.close()

if __name__ == '__main__':
    main()
