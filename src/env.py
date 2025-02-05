import pyglet
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config


class AntFarmEnv(gym.Env):
    def __init__(self, grid_size=config.GRID_SIZE, max_steps=config.MAX_STEPS, fixed_goal=None):
        super(AntFarmEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # 4 actions (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)

        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size

        # Goal Position (default is fixed_goal or [1,1])
        self.fixed_goal = np.array(fixed_goal if fixed_goal else [1, 1])

        # Initialize the environment state
        self.reset()

        # Pyglet Window Setup
        self.window = pyglet.window.Window(self.window_size, self.window_size, "Ant Farm")
        self.batch = pyglet.graphics.Batch()

        # Draw static shapes:
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

        # Create Maze Walls based on config.MAZE_LAYOUT (walls as gray rectangles)
        self.wall_shapes = []
        for y, row in enumerate(config.MAZE_LAYOUT):
            for x, cell in enumerate(row):
                if cell == 1:
                    wall = pyglet.shapes.Rectangle(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size, self.cell_size,
                        color=(128, 128, 128), batch=self.batch
                    )
                    self.wall_shapes.append(wall)

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

        # On-screen Labels
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
        # Ensure the agent is not placed on the goal or inside a wall.
        self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        while (np.array_equal(self.agent_pos, self.fixed_goal) or
               self._is_in_obstacle(self.agent_pos)):
            self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])

        self.goal_pos = self.fixed_goal.copy()
        self.steps = 0
        self.episode_reward = 0  # Reset cumulative reward
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), {}

    def _is_in_obstacle(self, pos):
        """Return True if the position (a 2D coordinate) is inside a wall in the maze."""
        x, y = pos
        return config.MAZE_LAYOUT[y][x] == 1  # Walls are represented as `1`

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        self.steps += 1
        new_pos = self.agent_pos.copy()

        # Update position based on action.
        # Note: With (0,0) at the bottom left, "Up" means increasing y.
        if action == 0:  # Up
            new_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 1:  # Down
            new_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 2:  # Left
            new_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # Right
            new_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)

        # If the intended new position is a wall, block the movement and apply a penalty.
        if self._is_in_obstacle(new_pos):
            reward = -5  # Penalty for hitting a wall
            new_pos = self.agent_pos.copy()  # Remain in place
        else:
            # Update the agent’s position and compute reward.
            self.agent_pos = new_pos
            new_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
            reward = 10 - new_distance
            if new_distance == 0:
                reward += config.REWARD_VALUE  # Bonus for reaching the goal

        self.episode_reward += reward

        # Check if goal is reached.
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

        # Update dynamic shapes (agent and goal).
        self.agent_shape.x = self.agent_pos[0] * self.cell_size + self.cell_size // 2
        self.agent_shape.y = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        self.goal_shape.x = self.goal_pos[0] * self.cell_size
        self.goal_shape.y = self.goal_pos[1] * self.cell_size

        # Draw all batch elements: grid lines, maze walls, agent, and goal.
        self.batch.draw()

        # Update and draw on-screen counters.
        self.episode_label.draw()
        self.step_label.text = f"Step: {self.steps}"
        self.step_label.draw()
        self.reward_label.text = f"Reward: {self.episode_reward}"
        self.reward_label.draw()

        self.window.flip()

    def close(self):
        self.window.close()
