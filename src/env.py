import pyglet
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config_loader


class AntFarmEnv(gym.Env):
    def __init__(self):
        config = config_loader.load_config()
        self.grid_size = config["GRID_SIZE"]
        self.max_steps = config["MAX_STEPS"]
        self.fixed_goal = np.array(config["FIXED_GOAL"])

        super(AntFarmEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32)

        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size

        self.obstacle = {
            "x": config["OBSTACLE_X"],
            "y": config["OBSTACLE_Y"],
            "width": config["OBSTACLE_WIDTH"],
            "height": config["OBSTACLE_HEIGHT"]
        }

        self.reset()

        self.window = pyglet.window.Window(self.window_size, self.window_size, "Ant Farm")
        self.batch = pyglet.graphics.Batch()

    def reset(self, seed=None, options=None):
        self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        while np.array_equal(self.agent_pos, self.fixed_goal) or self._is_in_obstacle(self.agent_pos):
            self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])

        self.goal_pos = self.fixed_goal.copy()
        self.steps = 0
        self.episode_reward = 0
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), {}

    def _is_in_obstacle(self, pos):
        ox, oy, w, h = self.obstacle["x"], self.obstacle["y"], self.obstacle["width"], self.obstacle["height"]
        return (ox <= pos[0] < ox + w) and (oy <= pos[1] < oy + h)

    def step(self, action):
        self.steps += 1
        new_pos = self.agent_pos.copy()

        if action == 0:
            new_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:
            new_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:
            new_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:
            new_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)

        reward = -1

        if self._is_in_obstacle(new_pos):
            reward -= 5
            new_pos = self.agent_pos.copy()
        else:
            self.agent_pos = new_pos
            new_distance = np.sum(np.abs(self.agent_pos - self.goal_pos))
            reward += (10 - new_distance)

        self.episode_reward += reward
        done = self.steps >= self.max_steps or np.array_equal(self.agent_pos, self.goal_pos)
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), reward, done, False, {}

    def render(self):
        self.window.dispatch_events()
        self.window.clear()
        self.batch.draw()
        self.window.flip()

    def close(self):
        self.window.close()
