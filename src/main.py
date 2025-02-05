import numpy as np
from env import AntFarmEnv
from q_learning import q_learning
from utils import save_q_table, load_q_table, plot_training_results
import config

def main():
    # Create the environment with settings from config.py
    env = AntFarmEnv(grid_size=config.GRID_SIZE, max_steps=config.MAX_STEPS, fixed_goal=config.FIXED_GOAL)

    # Load an existing Q-table if available
    Q = load_q_table(config.Q_TABLE_FILE)

    # Train the agent using Q-learning while showing a demo every 500 episodes
    Q, rewards, steps = q_learning(
        env,
        Q=Q,
        num_episodes=config.NUM_EPISODES,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON,
        demo_interval=config.EPISODE_INTERVAL  # Show demo every 500 episodes
    )

    # Save the updated Q-table
    save_q_table(Q, config.Q_TABLE_FILE)

    # Plot training results
    plot_training_results(rewards, steps)

    # Run a final demonstration episode with rendering after training completes
    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        state_key = tuple(state)
        if state_key not in Q:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state_key])
        state, reward, done, truncated, info = env.step(action)

    env.close()

if __name__ == '__main__':
    main()
