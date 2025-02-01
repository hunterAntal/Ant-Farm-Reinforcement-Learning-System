import pyglet
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AntFarmEnv(gym.Env):
    def __init__(self, grid_size=10, max_steps=100, fixed_goal=None):
        super(AntFarmEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)  # 4 actions (Up, Down, Left, Right)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)

        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size

        # Goal Position
        self.fixed_goal = np.array(fixed_goal if fixed_goal else [1, 1])

        # Initialize the environment state
        self.reset()

        # Pyglet Window Setup
        self.window = pyglet.window.Window(self.window_size, self.window_size, "Ant Farm")
        self.batch = pyglet.graphics.Batch()

        # Load Shapes
        self.goal_shape = pyglet.shapes.Rectangle(
            self.goal_pos[0] * self.cell_size,
            self.goal_pos[1] * self.cell_size,
            self.cell_size, self.cell_size, color=(255, 0, 0), batch=self.batch)

        self.agent_shape = pyglet.shapes.Circle(
            self.agent_pos[0] * self.cell_size + self.cell_size // 2,
            self.agent_pos[1] * self.cell_size + self.cell_size // 2,
            self.cell_size // 3, color=(0, 0, 255), batch=self.batch)

        # Grid Lines
        self.grid_lines = []
        for x in range(0, self.window_size, self.cell_size):
            self.grid_lines.append(
                pyglet.shapes.Line(x, 0, x, self.window_size, color=(200, 200, 200), batch=self.batch))
        for y in range(0, self.window_size, self.cell_size):
            self.grid_lines.append(
                pyglet.shapes.Line(0, y, self.window_size, y, color=(200, 200, 200), batch=self.batch))

        # Episode Counter Label (top right of the window)
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

        # Step Counter Label (placed just below the episode counter)
        self.step_label = pyglet.text.Label(
            "Step: 0",
            font_name='Arial',
            font_size=14,
            x=self.window_size - 10,
            y=self.window_size - 30,  # 20 pixels below the episode label
            anchor_x='right',
            anchor_y='top',
            color=(255, 255, 255, 255)
        )

    def reset(self, seed=None, options=None):
        # Randomly place the agent; avoid starting on the goal
        self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        while np.array_equal(self.agent_pos, self.fixed_goal):
            self.agent_pos = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])

        self.goal_pos = self.fixed_goal.copy()
        self.steps = 0
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), {}

    def step(self, action):
        self.steps += 1

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

        if new_distance == 0:
            reward += 50  # Bonus for reaching goal
            return self.reset()[0], reward, True, False, {}

        done = self.steps >= self.max_steps
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32), reward, done, False, {}

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_episode_counter(self, episode):
        """Update the text of the episode counter."""
        self.episode_label.text = f"Episode: {episode}"

    def render(self):
        # Process window events, clear, update positions, and flip the buffer
        self.window.dispatch_events()
        self.window.clear()
        self.agent_shape.x = self.agent_pos[0] * self.cell_size + self.cell_size // 2
        self.agent_shape.y = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        self.batch.draw()
        self.episode_label.draw()  # Draw the episode counter label

        # Update and draw the step counter label using the current step count
        self.step_label.text = f"Step: {self.steps}"
        self.step_label.draw()

        self.window.flip()

    def close(self):
        self.window.close()
