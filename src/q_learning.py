# q_learning.py

import numpy as np

def q_learning(env, Q=None, num_episodes=5000, alpha=0.1, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
    """
    Basic Q-learning algorithm that can continue from an existing Q-table.

    Parameters:
      - env: The environment instance.
      - Q: (Optional) An existing Q-table dictionary.
      - num_episodes: Total training episodes.
      - alpha: Learning rate.
      - gamma: Discount factor.
      - epsilon: Initial exploration rate.
      - epsilon_decay: Decay rate for epsilon per episode.
      - min_epsilon: Minimum exploration rate.

    Returns:
      - Q: The updated Q-table.
      - episode_rewards: List with the total reward for each episode.
      - episode_steps: List with the number of steps taken for each episode.
    """
    if Q is None:
        Q = {}
    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_key = tuple(state)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Initialize unseen state in Q-table.
            if state_key not in Q:
                Q[state_key] = np.zeros(env.action_space.n)
            # Choose action using epsilon-greedy policy.
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_key])

            next_state, reward, done, truncated, info = env.step(action)
            next_state_key = tuple(next_state)
            if next_state_key not in Q:
                Q[next_state_key] = np.zeros(env.action_space.n)

            # Update Q-value.
            best_next = np.max(Q[next_state_key])
            Q[state_key][action] += alpha * (reward + gamma * best_next - Q[state_key][action])

            state_key = next_state_key
            total_reward += reward
            steps += 1

        # Decay epsilon after each episode.
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}, Epsilon = {epsilon:.3f}")

    return Q, episode_rewards, episode_steps
