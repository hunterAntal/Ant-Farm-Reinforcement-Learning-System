import numpy as np
import time


def q_learning(env, Q, num_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, demo_interval=500):
    rewards = []
    steps = []
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_key = tuple(state)
            # Epsilon-greedy action selection
            if state_key not in Q or np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_key])

            next_state, reward, done, truncated, info = env.step(action)
            next_state_key = tuple(next_state)

            # Initialize Q entries if needed
            if state_key not in Q:
                Q[state_key] = np.zeros(env.action_space.n)
            if next_state_key not in Q:
                Q[next_state_key] = np.zeros(env.action_space.n)

            # Update the Q-value for the state-action pair
            Q[state_key][action] = Q[state_key][action] + alpha * (
                        reward + gamma * np.max(Q[next_state_key]) - Q[state_key][action])

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        steps.append(episode)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Every demo_interval episodes, run a demonstration episode
        if episode % demo_interval == 0:
            print(f"--- Demonstration at Episode {episode} ---")
            demo(env, Q, episode)

    return Q, rewards, steps


def demo(env, Q, episode):
    """
    Run a demonstration episode with rendering.
    The current episode number is displayed in the top right of the window.
    """
    # Update the episode counter in the environment
    if hasattr(env, 'update_episode_counter'):
        env.update_episode_counter(episode)

    state, _ = env.reset()
    done = False
    while not done:
        env.render()  # Render the environment (with the episode counter)
        time.sleep(0.1)  # Slow down the rendering for observation
        state_key = tuple(state)
        if state_key not in Q:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state_key])
        state, reward, done, truncated, info = env.step(action)
