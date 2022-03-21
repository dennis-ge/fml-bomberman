import os
from datetime import datetime, timezone

import events as e

#
# General settings
#

AGENT_NAME = "task1"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
TIMESTAMP = datetime.now(timezone.utc).strftime("%m-%dT%H:%M")
PROD_MODEL_NAME = f"./models/{AGENT_NAME}-eps.pt"  # When a model is trained with enough rounds, we move it into the internal models directory

#
# ML/Hyperparameter
#
NUMBER_OF_FEATURES = 8

GREEDY_POLICY_NAME = 'greedy'
EPSILON_GREEDY_POLICY_NAME = 'epsilon_greedy'
DECAY_GREEDY_POLICY_NAME = 'decay_greedy'

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


class EnvSettings:
    MATCH_ID: str
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

    def __init__(self):
        self.reload()

    def reload(self):
        self.MATCH_ID = os.environ.get("MATCH_ID", f"{AGENT_NAME}-{TIMESTAMP}")
        self.MODEL_NAME = "../../dump/models/" + os.environ.get("MODEL_NAME", f"{self.MATCH_ID}.pt")  # We store each model first within the dump directory
        self.REWARDS_NAME = f"../../dump/rewards/{self.MATCH_ID}.pt"
        self.WEIGHTS_NAME = f"../../dump/weights/{self.MATCH_ID}.pt"
        self.POLICY_NAME = os.environ.get("POLICY", GREEDY_POLICY_NAME)
        self.NUMBER_OF_ROUNDS = int(os.getenv("N_ROUNDS", 100))
        self.EPSILON = float(os.environ.get("EPS", 0.4))  # eps for epsilon greedy policy
        self.EPSILON_START = float(os.environ.get("EPS_START", 1))
        self.EPSILON_END = float(os.environ.get("EPS_MIN", 0.05))
        self.EPSILON_DECAY = float(os.environ.get("EPS_DECAY", 0.9994))
        self.LEARNING_RATE = float(os.environ.get("ALPHA", 0.025))  # alpha learning rate
        self.DISCOUNT_FACTOR = float(os.environ.get("GAMMA", 0.99))  # gamma discount factor
        self.BIAS = float(os.environ.get("BIAS", 0.1))


env = EnvSettings()

#
# Custom Events
#
# Feature 1
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
# Feature 2
DID_NOT_COLLECT_COIN = "DID_NOT_COLLECT_COIN"
# Feature 3
BOMB_ACTION_WAS_INTELLIGENT = "BOMB_ACTION_WAS_INTELLIGENT"
BOMB_ACTION_WAS_NOT_INTELLIGENT = "BOMB_ACTION_WAS_NOT_INTELLIGENT"
WAIT_ACTION_IS_INTELLIGENT = "WAIT_ACTION_IS_INTELLIGENT"
# Feature 4
MOVED_OUT_OF_BLAST_RADIUS = "MOVED_OUT_OF_BLAST_RADIUS"
STAYED_IN_BLAST_RADIUS = "STAYED_IN_BLAST_RADIUS"
# Feature 5
MOVED_TOWARDS_BOMB_FIELDS = "MOVED_TOWARDS_BOMB_FIELDS"
MOVED_AWAY_FROM_BOMB_FIELDS = "MOVED_AWAY_FROM_BOMB_FIELDS"
# Feature 6
PLACED_BOMB_NEXT_TO_CRATE = "PLACED_BOMB_NEXT_TO_CRATE"
DID_NOT_PLACED_BOMB_NEXT_TO_CRATE = "DID_NOT_PLACED_BOMB_NEXT_TO_CRATE"
# Feature 7
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"
# Feature 9
STAYED_OUT_OF_BOMB_RADIUS = "STAYED_OUT_OF_BOMB_RADIUS"
MOVED_INTO_BOMB_RADIUS = "MOVED_INTO_BOMB_RADIUS"
REWARDS = {
    # Positive
    e.OPPONENT_ELIMINATED: 0,
    e.KILLED_OPPONENT: 0,
    MOVED_TOWARDS_CRATE: 2,
    e.BOMB_EXPLODED: 2,  # Own bomb dropped earlier on has exploded.
    e.CRATE_DESTROYED: 5,  # A crate was destroyed by own bomb.
    PLACED_BOMB_NEXT_TO_CRATE: 5,
    e.BOMB_DROPPED: 10,
    e.COIN_FOUND: 10,  # A coin has been revealed by own bomb.
    MOVED_AWAY_FROM_BOMB_FIELDS: 30,
    MOVED_TOWARDS_COIN: 15,
    e.COIN_COLLECTED: 20,
    WAIT_ACTION_IS_INTELLIGENT: 20,  # contrary to WAITED
    STAYED_OUT_OF_BOMB_RADIUS: 30,
    MOVED_OUT_OF_BLAST_RADIUS: 40,
    e.SURVIVED_ROUND: 50,
    # Negative
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_RIGHT: -1,
    e.MOVED_LEFT: -1,
    MOVED_AWAY_FROM_CRATE: -2,
    DID_NOT_PLACED_BOMB_NEXT_TO_CRATE: -40,
    DID_NOT_COLLECT_COIN: -25,
    MOVED_AWAY_FROM_COIN: -20,
    MOVED_TOWARDS_BOMB_FIELDS: -55,
    MOVED_INTO_BOMB_RADIUS: -60,
    e.WAITED: -20,
    e.GOT_KILLED: -50,
    e.INVALID_ACTION: -50,  # Picked a non-existent action or one that couldn’t be executed.
    STAYED_IN_BLAST_RADIUS: -80,
    e.KILLED_SELF: -100,
}
