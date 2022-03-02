from typing import Tuple

import events as e

NUMBER_OF_FEATURES = 2
MODEL_NAME = "models/my_agent.pt"

# Possible Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']  # , 'BOMB']

# Names for policies
GREEDY_POLICY_NAME = 'greedy'
EPSILON_GREEDY_POLICY_NAME = 'epsilon_greedy'
DECAY_GREEDY_POLICY_NAME = 'decay_greedy'

EPSILON = 0.20  # eps for epsilon greedy policy
LEARNING_RATE = 0.1  # alpha learning rate
DISCOUNT_FACTOR = 0.50  # gamma discount factor
BIAS = 0.1

# Train Hyper parameters
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Custom Events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"

REWARDS = {
    # Positive
    e.CRATE_DESTROYED: 2,
    e.BOMB_EXPLODED: 2,
    e.COIN_FOUND: 5,
    e.BOMB_DROPPED: 5,
    e.COIN_COLLECTED: 10,
    e.OPPONENT_ELIMINATED: 0,
    e.KILLED_OPPONENT: 20,
    e.SURVIVED_ROUND: 10,
    MOVED_TOWARDS_COIN: 5,
    # Negative
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_LEFT: -1,
    e.MOVED_RIGHT: -1,
    e.WAITED: -50,
    e.GOT_KILLED: -50,
    e.INVALID_ACTION: -50,
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
