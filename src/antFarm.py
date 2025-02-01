import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
import sys


# ------------------------------------------------------------------------------
# Define the custom environment: AntFarmEnv
# ------------------------------------------------------------------------------

class AntFarmEnv(gym.Env):
    """
    A grid-world environment where an agent ("ant") tries to reach a goal.
    The grid is of size grid_size x grid_size.

    - **Observation**: An array of 4 integers: [agent_x, agent_y, goal_x, goal_y]
    - **Action Space**: Discrete(4)
        - 0: move up    (decrease y)
        - 1: move down  (increase y)
        - 2: move left  (decrease x)
        - 3: move right (increase x)
    - **Reward**: Shaped reward based on Manhattan distance.
         reward = 10 - new_distance, with a bonus if the agent reaches the goal.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=10, max_steps=100):
        super(AntFarmEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Define action space: 4 possible moves
        self.action_space = spaces.Discrete(4)

        # Observation: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)

        # Rendering parameters using pygame
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size
        self.screen = None
        self.clock = None

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        # Randomize the starting positions for the agent and goal.
        self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        self.goal_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        # Ensure the agent and goal don't start at the same cell.
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])

        self.steps = 0
        # Compute initial Manhattan distance.
        self.prev_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)

        obs = np.array([self.agent_pos[0], self.agent_pos[1], self.goal_pos[0], self.goal_pos[1]], dtype=np.int32)
        return obs, {}

    def step(self, action):
        self.steps += 1

        # Save the old distance before moving.
        old_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)

        # Update agent position based on action (with boundary checks).
        if action == 0:  # up: decrease y
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:  # down: increase y
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # left: decrease x
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # right: increase x
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)

        # Calculate new Manhattan distance.
        new_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)

        # Shaped reward: higher reward when closer to the goal.
        reward = 10 - new_distance

        # Check if the goal has been reached.
        done = False
        if new_distance == 0:
            reward += 50  # Bonus for reaching the goal.
            done = True

        # End episode if maximum steps have been taken.
        if self.steps >= self.max_steps:
            done = True

        obs = np.array([self.agent_pos[0], self.agent_pos[1], self.goal_pos[0], self.goal_pos[1]], dtype=np.int32)
        info = {}
        # Gymnasium expects (observation, reward, done, truncated, info)
        return obs, reward, done, False, info

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self, mode='human'):
        # Initialize pygame display if necessary.
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Ant Farm Environment")
            self.clock = pygame.time.Clock()

        # Fill the background.
        self.screen.fill((255, 255, 255))

        # Draw grid lines.
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.window_size, y))

        # Draw the goal as a red square.
        goal_rect = pygame.Rect(self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size, self.cell_size,
                                self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)

        # Draw the agent as a blue circle.
        agent_center = (self.agent_pos[0] * self.cell_size + self.cell_size // 2,
                        self.agent_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_center, self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ------------------------------------------------------------------------------
# Q-Learning Implementation
# ------------------------------------------------------------------------------

def q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
    """
    Basic Q-Learning algorithm.

    Parameters:
      - env: The environment instance.
      - num_episodes: How many episodes to train for.
      - alpha: Learning rate.
      - gamma: Discount factor.
      - epsilon: Initial exploration rate.
      - epsilon_decay: Decay factor for epsilon after each episode.
      - min_epsilon: Minimum exploration rate.

    Returns:
      - Q: The learned Q-table.
      - episode_rewards: List of total rewards per episode.
      - episode_steps: List of steps taken per episode.
    """
    Q = {}  # Q-table as a dictionary: keys are state tuples, values are Q-value arrays.
    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_key = tuple(state)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Initialize Q-values for unseen states.
            if state_key not in Q:
                Q[state_key] = np.zeros(env.action_space.n)

            # ε-greedy policy: choose a random action with probability ε.
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_key])

            next_state, reward, done, truncated, info = env.step(action)
            next_state_key = tuple(next_state)
            if next_state_key not in Q:
                Q[next_state_key] = np.zeros(env.action_space.n)

            # Q-Learning update rule.
            best_next = np.max(Q[next_state_key])
            Q[state_key][action] += alpha * (reward + gamma * best_next - Q[state_key][action])

            state_key = next_state_key
            total_reward += reward
            steps += 1

        # Decay ε after each episode.
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}, Epsilon = {epsilon:.3f}")

    return Q, episode_rewards, episode_steps


# ------------------------------------------------------------------------------
# Main: Training and Demonstration
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create the environment.
    env = AntFarmEnv(grid_size=50, max_steps=100)

    # Train the agent using Q-Learning.
    Q, rewards, steps = q_learning(
        env,
        num_episodes=5000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.01
    )

    # Plot training results.
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

    # Run a demonstration episode with rendering.
    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        state_key = tuple(state)
        # Use the learned policy; if unseen state, pick a random action.
        if state_key not in Q:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state_key])
        state, reward, done, truncated, info = env.step(action)
        pygame.time.wait(200)  # Pause 200ms between steps

    env.close()
