# env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class AntFarmEnv(gym.Env):
    """
    A grid-world environment where an agent ("ant") tries to reach a fixed goal.
    The grid is of size grid_size x grid_size.

    - **Observation**: An array of 4 integers: [agent_x, agent_y, goal_x, goal_y]
    - **Action Space**: Discrete(4)
        - 0: move up    (decrease y)
        - 1: move down  (increase y)
        - 2: move left  (decrease x)
        - 3: move right (increase x)
    - **Reward**: Shaped reward based on Manhattan distance:
         reward = 10 - new_distance, with a bonus if the agent reaches the goal.

    The goal position remains fixed throughout all episodes and simulation runs.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=10, max_steps=100, fixed_goal=None):
        super(AntFarmEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Define action and observation spaces.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)

        # Rendering parameters.
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size
        self.screen = None
        self.clock = None

        # Set the fixed goal.
        if fixed_goal is None:
            self.fixed_goal = np.array([np.random.randint(grid_size), np.random.randint(grid_size)])
        else:
            self.fixed_goal = np.array(fixed_goal)

        # Initialize the environment state.
        self.reset()

    def reset(self, seed=None, options=None):
        # Randomize the agent's starting position.
        self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        # Ensure the agent doesn't start on the fixed goal.
        while np.array_equal(self.agent_pos, self.fixed_goal):
            self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        # Use the fixed goal for every episode.
        self.goal_pos = self.fixed_goal.copy()

        self.steps = 0
        self.prev_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        obs = np.array([self.agent_pos[0], self.agent_pos[1],
                        self.goal_pos[0], self.goal_pos[1]], dtype=np.int32)
        return obs, {}

    def step(self, action):
        self.steps += 1

        # Update agent's position.
        if action == 0:  # Up
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)

        new_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        reward = 10 - new_distance

        done = False
        if new_distance == 0:
            reward += 50  # Bonus for reaching the goal.
            done = True
        if self.steps >= self.max_steps:
            done = True

        obs = np.array([self.agent_pos[0], self.agent_pos[1],
                        self.goal_pos[0], self.goal_pos[1]], dtype=np.int32)
        info = {}
        return obs, reward, done, False, info

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Ant Farm Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        # Draw grid lines.
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.window_size, y))

        # Draw the goal (red square).
        goal_rect = pygame.Rect(self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)

        # Draw the agent (blue circle).
        agent_center = (self.agent_pos[0] * self.cell_size + self.cell_size // 2,
                        self.agent_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_center, self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
