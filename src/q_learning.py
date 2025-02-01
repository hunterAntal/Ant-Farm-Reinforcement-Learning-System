import numpy as np
import time
import matplotlib.pyplot as plt
from utils import plot_histogram_and_boxplot, plot_training_results


def select_action(state, Q, env, epsilon):
    """
    Select an action using an epsilon-greedy policy.

    Parameters:
        state: The current state.
        Q: The Q-table (a dictionary).
        env: The environment (used for random action sampling).
        epsilon: The exploration rate.

    Returns:
        action: The selected action.
    """
    state_key = tuple(state)
    if state_key not in Q or np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state_key])


def update_q_value(Q, state_key, action, reward, next_state_key, alpha, gamma, env):
    """
    Update the Q-value for a given state-action pair using the Q-learning update rule.

    Parameters:
        Q: The Q-table (a dictionary).
        state_key: The key for the current state.
        action: The action taken.
        reward: The reward received.
        next_state_key: The key for the next state.
        alpha: The learning rate.
        gamma: The discount factor.
        env: The environment (used to initialize Q entries).

    Returns:
        None (Q is updated in place).
    """
    # Initialize Q entries if needed
    if state_key not in Q:
        Q[state_key] = np.zeros(env.action_space.n)
    if next_state_key not in Q:
        Q[next_state_key] = np.zeros(env.action_space.n)
    Q[state_key][action] += alpha * (reward + gamma * np.max(Q[next_state_key]) - Q[state_key][action])


def q_learning(env, Q, num_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, demo_interval=500):
    """
    Train an agent using Q-learning.

    Parameters:
        env: The environment.
        Q: The initial Q-table (a dictionary).
        num_episodes: Total number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        epsilon_decay: Factor by which epsilon decays after each episode.
        min_epsilon: The minimum exploration rate.
        demo_interval: Run a demonstration every `demo_interval` episodes.

    Returns:
        Q: The updated Q-table.
        rewards: A list of total rewards per episode.
        steps_list: A list containing the number of steps taken in each episode.
    """
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

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Every demo_interval episodes, run a demonstration
        if episode % demo_interval == 0:
            print(f"--- Demonstration at Episode {episode} ---")
            demo(env, Q, episode)

    # After training, generate both plots and display them together.
    fig1 = plot_histogram_and_boxplot(steps_list)
    fig2 = plot_training_results(rewards, steps_list)
    plt.show()

    return Q, rewards, steps_list


def demo(env, Q, episode):
    """
    Run a demonstration episode with rendering.
    The current episode number is displayed in the top right of the window.

    Parameters:
        env: The environment.
        Q: The Q-table.
        episode: The current episode number.
    """
    # Update the episode counter (if the environment supports it)
    if hasattr(env, 'update_episode_counter'):
        env.update_episode_counter(episode)

    state, _ = env.reset()
    done = False
    while not done:
        env.render()  # Render the environment
        time.sleep(0.1)  # Slow down rendering for observation
        state_key = tuple(state)
        if state_key not in Q:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state_key])
        state, reward, done, truncated, info = env.step(action)
