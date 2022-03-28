import os
from datetime import datetime, timezone

from agent_code.fml.rewards import *

AGENT_NAME = "fml_double"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
TIMESTAMP = datetime.now(timezone.utc).strftime("%m-%dT%H:%M")
PROD_MODEL_NAME = f"./models/fml-eps.pt"
PRODUCTION = False
DUMP_DIRECTORY = "../../dump/"

NUMBER_OF_FEATURES = 13

GREEDY_POLICY_NAME = 'greedy'
EPSILON_GREEDY_POLICY_NAME = 'epsilon_greedy'
DECAY_GREEDY_POLICY_NAME = 'decay_greedy'

TRANSITION_HISTORY_SIZE = 1000
OPPONENT_TRANSITION_HISTORY_SIZE = 20


class EnvSettings:
    PRINT_FIELD: bool
    MATCH_ID: str
    MODEL_NAME_1: str
    MODEL_NAME_2: str
    MODEL_NAME: str
    WEIGHTS_NAME: str
    REWARDS_NAME: str
    REWARDS_ACC_NAME: str
    ALL_COINS_COLLECTED: str
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
        self.MODEL_NAME = f'{DUMP_DIRECTORY}/models/{os.environ.get("MODEL_NAME", "")}'
        if PRODUCTION or self.MODEL_NAME == f'{DUMP_DIRECTORY}/models/':
            self.MODEL_NAME = PROD_MODEL_NAME
        self.REWARDS_NAME = f"{DUMP_DIRECTORY}/rewards/{self.MATCH_ID}.pt"
        self.REWARDS_ACC_NAME = f"{DUMP_DIRECTORY}/rewards/acc_{self.MATCH_ID}.pt"
        self.WEIGHTS_NAME = f"{DUMP_DIRECTORY}/weights/{self.MATCH_ID}.pt"
        self.ALL_COINS_COLLECTED = f"{DUMP_DIRECTORY}/all_coins/{self.MATCH_ID}.pt"
        self.POLICY_NAME = os.environ.get("POLICY", GREEDY_POLICY_NAME)
        self.NUMBER_OF_ROUNDS = int(os.getenv("N_ROUNDS", 100))
        self.BIAS = float(os.environ.get("BIAS", 1))
        self.DISCOUNT_FACTOR = float(os.environ.get("GAMMA", 0.99))
        self.EPSILON = float(os.environ.get("EPS", 0.12))
        self.LEARNING_RATE = float(os.environ.get("ALPHA", 0.01))
        self.EPSILON_START = float(os.environ.get("EPS_START", 1))
        self.EPSILON_END = float(os.environ.get("EPS_MIN", 0.05))
        self.EPSILON_DECAY = float(os.environ.get("EPS_DECAY", 0.9994))
        self.EXPERIENCE_REPLAY_ACTIVATED = (os.environ.get("EXPERIENCE_REPLAY_ACTIVATED", True))
        self.REWARDS = {}

        if not PRODUCTION:
            for name, item in REWARDS.items():
                self.REWARDS[name] = int(os.environ.get(name, item[0]))
        else:
            for name, item in REWARDS.items():
                self.REWARDS[name] = item[0]


env = EnvSettings()
