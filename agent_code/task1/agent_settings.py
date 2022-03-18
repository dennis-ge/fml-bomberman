import os
from datetime import datetime, timezone
from typing import Tuple

import events as e

#
# Agent settings
#
AGENT_NAME = "task1"
TIMESTAMP = datetime.now(timezone.utc).strftime("%m-%dT%H:%M")

MODEL_NAME = f"../../dump/{AGENT_NAME}-{TIMESTAMP}.pt"  # We store each model first within the dump directory
PROD_MODEL_NAME = f"./models/{AGENT_NAME}-1000r.pt"  # When a model is trained with enough rounds, we move it into the internal models directory
REWARDS_NAME = f"../../dump/rewards-{AGENT_NAME}-{TIMESTAMP}.csv"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#
# ML/Hyperparameter
#
NUMBER_OF_FEATURES = 6

GREEDY_POLICY_NAME = 'greedy'
EPSILON_GREEDY_POLICY_NAME = 'epsilon_greedy'
DECAY_GREEDY_POLICY_NAME = 'decay_greedy'

policy_name = os.environ.get("POLICY", EPSILON_GREEDY_POLICY_NAME)

NUMBER_OF_ROUNDS = os.getenv("N_ROUNDS", 100)
EPSILON = os.environ.get("EPS", 0.15)  # eps for epsilon greedy policy
EPSILON_START = os.environ.get("EPS_START", 1)
EPSILON_END = os.environ.get("EPS_MIN", 0.05)
EPSILON_DECAY = os.environ.get("EPS_DECAY", 0.9994)

LEARNING_RATE = os.environ.get("ALPHA", 0.1)  # alpha learning rate
DISCOUNT_FACTOR = os.environ.get("GAMMA", 0.60)  # gamma discount factor
BIAS = os.environ.get("BIAS", 0.1)
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Custom Events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_OUT_OF_BLAST_RADIUS = "MOVED_OUT_OF_BLAST_RADIUS"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
STAYED_IN_BLAST_RADIUS = "STAYED_IN_BLAST_RADIUS"
# TODO MOVED_INTO_BLAST_RADIUS
STAYED_OUT_OF_EXPLOSION_RADIUS = "STAYED_OUT_OF_EXPLOSION_RADIUS"  # TODO verify if it is even possible to reach explosion radius
MOVED_INTO_EXPLOSION_RADIUS = "MOVED_INTO_EXPLOSION_RADIUS"

REWARDS = {
    # Positive
    e.CRATE_DESTROYED: 2,  # A crate was destroyed by own bomb.
    e.COIN_FOUND: 5,  # A coin has been revealed by own bomb.
    e.BOMB_DROPPED: 5,
    e.BOMB_EXPLODED: 2,  # Own bomb dropped earlier on has exploded.
    e.COIN_COLLECTED: 10,
    e.OPPONENT_ELIMINATED: 0,
    e.KILLED_OPPONENT: 20,
    e.SURVIVED_ROUND: 10,
    MOVED_TOWARDS_COIN: 5,
    MOVED_OUT_OF_BLAST_RADIUS: 15,
    STAYED_OUT_OF_EXPLOSION_RADIUS: 2,
    # Negative
    MOVED_AWAY_FROM_COIN: -10,
    STAYED_IN_BLAST_RADIUS: -15,
    MOVED_INTO_EXPLOSION_RADIUS: -50,
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_LEFT: -1,
    e.MOVED_RIGHT: -1,
    e.WAITED: -50,
    e.GOT_KILLED: -50,
    e.INVALID_ACTION: -50,  # Picked a non-existent action or one that couldn’t be executed.
    e.KILLED_SELF: -100,
}


def get_new_position(action: str, x: int, y: int) -> Tuple[int, int]:
    switch = {
        'UP': (x, y - 1),
        'DOWN': (x, y + 1),
        'RIGHT': (x + 1, y),
        'LEFT': (x - 1, y),
        'WAIT': (x, y),
        'BOMB': (x, y),
    }

    return switch[action]
