import numpy as np
import time
import matplotlib.pyplot as plt
from utils import plot_histogram_and_boxplot, plot_training_results
import config_loader


def select_action(state, Q, env, epsilon):
    """
    Select an action using an epsilon-greedy policy.
    """
    state_key = tuple(state)
    if state_key not in Q or np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state_key])


def update_q_value(Q, state_key, action, reward, next_state_key, alpha, gamma, env):
    """
    Update the Q-value for a given state-action pair using the Q-learning update rule.
    """
    if state_key not in Q:
        Q[state_key] = np.zeros(env.action_space.n)
    if next_state_key not in Q:
        Q[next_state_key] = np.zeros(env.action_space.n)
    Q[state_key][action] += alpha * (reward + gamma * np.max(Q[next_state_key]) - Q[state_key][action])


def q_learning(env, Q, demo_interval=500):
    """
    Train an agent using Q-learning with configurations from config_loader.
    """
    config = config_loader.load_config()
    num_episodes = config["NUM_EPISODES"]
    alpha = config["ALPHA"]
    gamma = config["GAMMA"]
    epsilon = config["EPSILON"]
    epsilon_decay = config["EPSILON_DECAY"]
    min_epsilon = config["MIN_EPSILON"]

    rewards = []
    steps_list = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps_taken = 0

        while not done:
            steps_taken += 1
            state_key = tuple(state)
            action = select_action(state, Q, env, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            next_state_key = tuple(next_state)

            update_q_value(Q, state_key, action, reward, next_state_key, alpha, gamma, env)

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        steps_list.append(steps_taken)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % demo_interval == 0:
            print(f"--- Demonstration at Episode {episode} ---")
            demo(env, Q, episode)

    fig1 = plot_histogram_and_boxplot(steps_list)
    fig2 = plot_training_results(rewards, steps_list)
    plt.show()

    return Q, rewards, steps_list


def demo(env, Q, episode):
    """
    Run a demonstration episode with rendering.
    """
    if hasattr(env, 'update_episode_counter'):
        env.update_episode_counter(episode)

    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.1)
        state_key = tuple(state)
        action = np.argmax(Q[state_key]) if state_key in Q else env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
