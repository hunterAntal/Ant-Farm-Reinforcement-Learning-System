import yaml
import logging
import numpy as np
import time
import psutil
import h5py
import sqlite3
import pyglet

# Default configuration values
DEFAULT_CONFIG = {
    "GRID_SIZE": 10,
    "MAX_STEPS": 25,
    "FIXED_GOAL": (1, 1),
    "OBSTACLE_X": 3,
    "OBSTACLE_Y": 3,
    "OBSTACLE_WIDTH": 7,
    "OBSTACLE_HEIGHT": 4,
    "NUM_EPISODES": 40000,
    "ALPHA": 0.1,
    "GAMMA": 0.99,
    "EPSILON": 1.0,
    "EPSILON_DECAY": 0.999,
    "MIN_EPSILON": 0.01,
    "REACHED_GOAL_BONUS": 500,
    "Q_TABLE_FILE": "q_table.npy"
}

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(file_path="config.yaml"):
    """Loads and validates configuration from a YAML file."""
    try:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        logging.warning("Config file not found. Using default settings.")
        return DEFAULT_CONFIG
    except yaml.YAMLError as e:
        logging.error("Error parsing YAML: %s", e)
        raise SystemExit("Critical Error: Invalid YAML file.")

    validated_config = {}
    for key, default_value in DEFAULT_CONFIG.items():
        if key in config_data:
            value = config_data[key]
            # Special handling for FIXED_GOAL: accept both tuple and list
            if key == "FIXED_GOAL" and isinstance(value, list):
                validated_config[key] = tuple(value)
                continue
            if not isinstance(value, type(default_value)):
                logging.error("Invalid type for %s: Expected %s, got %s", key, type(default_value).__name__,
                              type(value).__name__)
                raise SystemExit("Critical Error: Misconfigured parameter.")
            validated_config[key] = value
        else:
            logging.warning("%s missing in config. Using default: %s", key, default_value)
            validated_config[key] = default_value

    # (Optional) A redundant conversion if you want to double-check
    if isinstance(validated_config["FIXED_GOAL"], list):
        validated_config["FIXED_GOAL"] = tuple(validated_config["FIXED_GOAL"])

    if not (0 < validated_config["EPSILON_DECAY"] <= 1):
        logging.error("EPSILON_DECAY must be in range (0, 1]. Found: %s", validated_config["EPSILON_DECAY"])
        raise SystemExit("Critical Error: EPSILON_DECAY out of range.")

    return validated_config



def benchmark_q_table(q_table, file_path="q_table.npy"):
    """Benchmarks .npy, HDF5, and SQLite storage methods."""
    logging.info("Starting Q-table benchmarking with %d states.", len(q_table))

    # Measure .npy performance
    start_time = time.time()
    np.save(file_path, q_table)
    write_time_npy = time.time() - start_time

    start_time = time.time()
    loaded_q_table = np.load(file_path, allow_pickle=True).item()
    read_time_npy = time.time() - start_time

    logging.info("Benchmark Results - .npy: (Write: %.4f sec, Read: %.4f sec)", write_time_npy, read_time_npy)

    return {
        "npy": (write_time_npy, read_time_npy)
    }


def run_view_mode(env, q_table, speed=1.0):
    """Runs the View Mode playback with real-time stats display."""
    state, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0

    while not done:
        env.render()
        time.sleep(0.1 / speed)  # Adjust speed dynamically
        state_key = tuple(state)
        action = np.argmax(q_table[state_key]) if state_key in q_table else env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

        logging.info("Step: %d, Action: %d, Reward: %.2f, Total Reward: %.2f", steps, action, reward, total_reward)

    logging.info("View Mode Complete - Total Steps: %d, Final Reward: %.2f", steps, total_reward)
    env.close()


# Load configuration
config = load_config()
