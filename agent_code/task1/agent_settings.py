import os
from datetime import datetime, timezone

import events as e

#
# Agent settings
#
AGENT_NAME = "task1"
TIMESTAMP = datetime.now(timezone.utc).strftime("%m-%dT%H:%M")

MODEL_NAME = "../../dump/" + os.environ.get("MODEL_NAME", f"{AGENT_NAME}-{TIMESTAMP}.pt")  # We store each model first within the dump directory
PROD_MODEL_NAME = f"./models/{AGENT_NAME}-trained.pt"  # When a model is trained with enough rounds, we move it into the internal models directory
REWARDS_NAME = f"../../dump/rewards-{AGENT_NAME}-{TIMESTAMP}.csv"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#
# ML/Hyperparameter
#
NUMBER_OF_FEATURES = 7

GREEDY_POLICY_NAME = 'greedy'
EPSILON_GREEDY_POLICY_NAME = 'epsilon_greedy'
DECAY_GREEDY_POLICY_NAME = 'decay_greedy'

policy_name = os.environ.get("POLICY", GREEDY_POLICY_NAME)


NUMBER_OF_ROUNDS = int(os.getenv("N_ROUNDS", 100))
EPSILON = float(os.environ.get("EPS", 0.15))  # eps for epsilon greedy policy
EPSILON_START = float(os.environ.get("EPS_START", 1))
EPSILON_END = float(os.environ.get("EPS_MIN", 0.05))
EPSILON_DECAY = float(os.environ.get("EPS_DECAY", 0.9994))

LEARNING_RATE = float(os.environ.get("ALPHA", 0.1))  # alpha learning rate
DISCOUNT_FACTOR = float(os.environ.get("GAMMA", 0.80))  # gamma discount factor
BIAS = float(os.environ.get("BIAS", 0.1))

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

#
# Custom Events
#
# Feature 1
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
# Feature 3
BOMB_ACTION_WAS_INTELLIGENT = "BOMB_ACTION_WAS_INTELLIGENT"
BOMB_ACTION_WAS_NOT_INTELLIGENT = "BOMB_ACTION_WAS_NOT_INTELLIGENT"
WAIT_ACTION_IS_INTELLIGENT = "WAIT_ACTION_IS_INTELLIGENT"
# Feature 4
MOVED_OUT_OF_BLAST_RADIUS = "MOVED_OUT_OF_BLAST_RADIUS"
STAYED_IN_BLAST_RADIUS = "STAYED_IN_BLAST_RADIUS"
# TODO MOVED_INTO_BLAST_RADIUS
# Feature 5
MOVED_TOWARDS_BOMB_FIELDS = "MOVED_TOWARDS_BOMB_FIELDS"
MOVED_AWAY_FROM_BOMB_FIELDS = "MOVED_AWAY_FROM_BOMB_FIELDS"
# Feature 6
PLACED_BOMB_NEXT_TO_CRATE = "PLACED_BOMB_NEXT_TO_CRATE"
DID_NOT_PLACED_BOMB_NEXT_TO_CRATE = "DID_NOT_PLACED_BOMB_NEXT_TO_CRATE"
# Feature 7
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"

REWARDS = {
    # Positive
    e.OPPONENT_ELIMINATED: 0,
    e.KILLED_OPPONENT: 0,
    MOVED_TOWARDS_CRATE: 2,
    e.BOMB_EXPLODED: 2,  # Own bomb dropped earlier on has exploded.
    e.CRATE_DESTROYED: 5,  # A crate was destroyed by own bomb.
    PLACED_BOMB_NEXT_TO_CRATE: 5,
    BOMB_ACTION_WAS_INTELLIGENT: 5,
    e.BOMB_DROPPED: 5,
    e.COIN_FOUND: 10,  # A coin has been revealed by own bomb.
    MOVED_AWAY_FROM_BOMB_FIELDS: 10,
    MOVED_TOWARDS_COIN: 15,
    e.COIN_COLLECTED: 20,
    WAIT_ACTION_IS_INTELLIGENT: 20,  # contrary to WAITED
    MOVED_OUT_OF_BLAST_RADIUS: 40,
    e.SURVIVED_ROUND: 50,
    # Negative
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_RIGHT: -1,
    e.MOVED_LEFT: -1,
    MOVED_AWAY_FROM_CRATE: -2,
    DID_NOT_PLACED_BOMB_NEXT_TO_CRATE: -10,
    BOMB_ACTION_WAS_NOT_INTELLIGENT: -10,
    MOVED_AWAY_FROM_COIN: -20,
    MOVED_TOWARDS_BOMB_FIELDS: -20,
    e.WAITED: -20,
    e.GOT_KILLED: -50,
    e.INVALID_ACTION: -50,  # Picked a non-existent action or one that couldn’t be executed.
    STAYED_IN_BLAST_RADIUS: -80,
    e.KILLED_SELF: -100,
}


