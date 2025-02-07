import numpy as np
from env import AntFarmEnv
from q_learning import q_learning
from utils import save_q_table, load_q_table, plot_training_results
import config_loader

def main():
    # Load configuration settings dynamically
    config = config_loader.load_config()

    # Create the environment (no extra parameters needed)
    env = AntFarmEnv()

    # Load an existing Q-table if available
    Q = load_q_table(config["Q_TABLE_FILE"])

    # Train the agent using Q-learning while showing a demo every 500 episodes
    Q, rewards, steps = q_learning(env, Q, demo_interval=500)

    # Save the updated Q-table
    save_q_table(Q, config["Q_TABLE_FILE"])

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
