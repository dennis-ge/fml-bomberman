import os
from datetime import datetime, timezone

from agent_code.task1_double_q.rewards import *

#
# General settings
#


AGENT_NAME = "task1_double_q"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
TIMESTAMP = datetime.now(timezone.utc).strftime("%m-%dT%H:%M")
PROD_MODEL_NAME = f"./models/{AGENT_NAME}-eps-v8.pt"
PRODUCTION = False  # False for production
#
# ML/Hyperparameter
#
NUMBER_OF_FEATURES = 13

GREEDY_POLICY_NAME = 'greedy'
EPSILON_GREEDY_POLICY_NAME = 'epsilon_greedy'
DECAY_GREEDY_POLICY_NAME = 'decay_greedy'

TRANSITION_HISTORY_SIZE = 1000  # keep
ENEMY_TRANSITION_HISTORY_SIZE = 20  # record enemy transitions with probability.

EXPERIENCE_REPLAY_K = 300
EXPERIENCE_REPLAY_BATCH_SIZE = 10


class EnvSettings:
    PRINT_FIELD: bool
    MATCH_ID: str
    MODEL_NAME_1: str
    MODEL_NAME_2: str
    MODEL_NAME: str
    WEIGHTS_NAME: str
    REWARDS_NAME: str
    POLICY_NAME: str
    BIAS: float
    LEARNING_RATE: float
    DISCOUNT_FACTOR: float
    EPSILON_DECAY: float
    EPSILON_END: float
    EPSILON_START: float
    EPSILON: float
    NUMBER_OF_ROUNDS: int
    EXPERIENCE_REPLAY_ACTIVATED: bool
    REWARDS: dict

    def __init__(self):
        self.reload()

    def reload(self):
        self.PRINT_FIELD = os.environ.get("PRINT_FIELD", False)
        self.MATCH_ID = os.environ.get("MATCH_ID", f"{AGENT_NAME}-{TIMESTAMP}")
        self.MODEL_NAME = "../../dump/models/" + os.environ.get("MODEL_NAME", "")  # We store each model first within the dump directory
        if self.MODEL_NAME == "../../dump/models/":
            self.MODEL_NAME = PROD_MODEL_NAME
        self.REWARDS_NAME = f"../../dump/rewards/{self.MATCH_ID}.pt"
        self.WEIGHTS_NAME = f"../../dump/weights/{self.MATCH_ID}.pt"
        self.POLICY_NAME = os.environ.get("POLICY", GREEDY_POLICY_NAME)
        self.NUMBER_OF_ROUNDS = int(os.getenv("N_ROUNDS", 100))
        self.EPSILON = float(os.environ.get("EPS", 0.12))  # eps for epsilon greedy policy
        self.BIAS = float(os.environ.get("BIAS", 0.1))
        self.DISCOUNT_FACTOR = float(os.environ.get("GAMMA", 0.92))  # gamma discount factor
        self.LEARNING_RATE = float(os.environ.get("ALPHA", 0.05))  # alpha learning rate
        self.EPSILON_START = float(os.environ.get("EPS_START", 1))
        self.EPSILON_END = float(os.environ.get("EPS_MIN", 0.05))
        self.EPSILON_DECAY = float(os.environ.get("EPS_DECAY", 0.9994))
        self.EXPERIENCE_REPLAY_ACTIVATED = os.environ.get("EXPERIENCE_REPLAY_ACTIVATED", True)

        self.REWARDS = {}
        if not PRODUCTION:
            for name, item in REWARDS.items():
                self.REWARDS[name] = int(os.environ.get(name, item[0]))
        else:
            for name, item in REWARDS.items():
                self.REWARDS[name] = item[1]


env = EnvSettings()
