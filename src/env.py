import pyglet
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config


class AntFarmEnv(gym.Env):
    def __init__(self, grid_size=10, max_steps=100, fixed_goal=None):
        super(AntFarmEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # 4 actions (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)

        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size

        # Goal Position (you may adjust as needed)
        self.fixed_goal = np.array(fixed_goal if fixed_goal else [1, 1])

        # --- Define a simple obstacle ---
        # In this example, the obstacle is a rectangle starting at (8,8)
        # with a width and height of 4 grid cells.
        self.obstacle = {"x": config.OBSTACLE_X, "y": config.OBSTACLE_Y, "width": config.OBSTACLE_WIDTH, "height": config.OBSTACLE_HEIGHT}

        # Initialize the environment state
        self.reset()

        # Pyglet Window Setup
        self.window = pyglet.window.Window(self.window_size, self.window_size, "Ant Farm")
        self.batch = pyglet.graphics.Batch()

        # Draw Shapes
        # Goal shape (red)
        self.goal_shape = pyglet.shapes.Rectangle(
            self.goal_pos[0] * self.cell_size,
            self.goal_pos[1] * self.cell_size,
            self.cell_size, self.cell_size,
            color=(255, 0, 0), batch=self.batch
        )

        # Agent shape (blue)
        self.agent_shape = pyglet.shapes.Circle(
            self.agent_pos[0] * self.cell_size + self.cell_size // 2,
            self.agent_pos[1] * self.cell_size + self.cell_size // 2,
            self.cell_size // 3, color=(0, 0, 255), batch=self.batch
        )

        # Obstacle shape (green)
        self.obstacle_shape = pyglet.shapes.Rectangle(
            self.obstacle["x"] * self.cell_size,
            self.obstacle["y"] * self.cell_size,
            self.obstacle["width"] * self.cell_size,
            self.obstacle["height"] * self.cell_size,
            color=(0, 255, 0), batch=self.batch
        )

        # Grid Lines
        self.grid_lines = []
        for x in range(0, self.window_size, self.cell_size):
            self.grid_lines.append(
                pyglet.shapes.Line(x, 0, x, self.window_size, color=(200, 200, 200), batch=self.batch)
            )
        for y in range(0, self.window_size, self.cell_size):
            self.grid_lines.append(
                pyglet.shapes.Line(0, y, self.window_size, y, color=(200, 200, 200), batch=self.batch)
            )

        # Episode Counter Label (top right)
        self.episode_label = pyglet.text.Label(
            "Episode: 0",
            font_name='Arial',
            font_size=14,
            x=self.window_size - 10,
            y=self.window_size - 10,
            anchor_x='right',
            anchor_y='top',
            color=(255, 255, 255, 255)
        )

        # Step Counter Label (below the episode counter)
        self.step_label = pyglet.text.Label(
            "Step: 0",
            font_name='Arial',
            font_size=14,
            x=self.window_size - 10,
            y=self.window_size - 30,
            anchor_x='right',
            anchor_y='top',
            color=(255, 255, 255, 255)
        )

        # Rewards Counter Label (below the step counter)
        self.reward_label = pyglet.text.Label(
            "Reward: 0",
            font_name='Arial',
            font_size=14,
            x=self.window_size - 10,
            y=self.window_size - 50,
            anchor_x='right',
            anchor_y='top',
            color=(255, 255, 255, 255)
        )

    def reset(self, seed=None, options=None):
        # Randomly place the agent.
        # Ensure the agent is not placed on the goal or inside the obstacle.
        self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        while (np.array_equal(self.agent_pos, self.fixed_goal) or
               self._is_in_obstacle(self.agent_pos)):
            self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])

        self.goal_pos = self.fixed_goal.copy()
        self.steps = 0
        self.episode_reward = 0  # Reset cumulative reward
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), {}

    def _is_in_obstacle(self, pos):
        """Return True if the position (a 2D coordinate) is inside the obstacle."""
        ox, oy, w, h = self.obstacle["x"], self.obstacle["y"], self.obstacle["width"], self.obstacle["height"]
        return (ox <= pos[0] < ox + w) and (oy <= pos[1] < oy + h)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        self.steps += 1
        new_pos = self.agent_pos.copy()

        # Determine intended new position based on action.
        if action == 0:  # Up
            new_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:  # Down
            new_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # Left
            new_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # Right
            new_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)

        # If the intended new position is inside the obstacle, block the movement
        # and apply a penalty.
        reward = -1  # Penalize every step to encourage efficiency

        if self._is_in_obstacle(new_pos):
            reward -= 5  # Additional penalty for hitting an obstacle
            new_pos = self.agent_pos.copy()  # Stay in place
        else:
            self.agent_pos = new_pos
            new_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)

            if new_distance == 0:
                reward += config.REACHED_GOAL_BONUS  # Big reward for success
            else:
                reward += (10 - new_distance)  # Smaller rewards for getting closer

        self.episode_reward += reward
        done = self.steps >= self.max_steps

        # If goal is reached, reset the environment.
        if np.array_equal(self.agent_pos, self.goal_pos):
            return self.reset()[0], reward, True, False, {}

        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), reward, done, False, {}

        self.episode_reward += reward

        # If the goal is reached, reset the environment.
        if np.array_equal(self.agent_pos, self.goal_pos):
            return self.reset()[0], reward, True, False, {}

        done = self.steps >= self.max_steps
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), reward, done, False, {}

    def update_episode_counter(self, episode):
        self.episode_label.text = f"Episode: {episode}"

    def render(self):
        # Process window events and clear the window.
        self.window.dispatch_events()
        self.window.clear()

        # Update the positions of dynamic shapes.
        self.agent_shape.x = self.agent_pos[0] * self.cell_size + self.cell_size // 2
        self.agent_shape.y = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        self.goal_shape.x = self.goal_pos[0] * self.cell_size
        self.goal_shape.y = self.goal_pos[1] * self.cell_size

        # Draw static elements (grid, agent, goal, obstacle).
        self.batch.draw()

        # Draw on-screen counters.
        self.episode_label.draw()
        self.step_label.text = f"Step: {self.steps}"
        self.step_label.draw()
        self.reward_label.text = f"Reward: {self.episode_reward}"
        self.reward_label.draw()

        self.window.flip()

    def close(self):
        self.window.close()
