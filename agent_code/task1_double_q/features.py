from typing import List

import numpy as np
from scipy.spatial.distance import cityblock

from agent_code.coin_collector_agent.callbacks import look_for_targets
from agent_code.task1_double_q.agent_settings import *


def calc_min_distance(coins: List[Tuple[int, int]], x: int, y: int) -> int:
    min_d = 10000000000
    for (x_coin, y_coin) in coins:
        # d = np.linalg.norm(np.array((x, y)) - np.array((x_coin, y_coin)), ord=1)
        d = cityblock([x, y], [x_coin, y_coin])
        if d < min_d:
            min_d = d

    return min_d


def feat_1(free_fields: np.array, coins: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves towards coin
    """
    feature = np.zeros(len(ACTIONS))

    best_direction = look_for_targets(free_fields, (x, y), coins)

    if coins:
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1

    return feature


def feat_2(coins: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent collects coin
    """
    feature = np.zeros(len(ACTIONS))
    if coins:
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) in coins:
                feature[idx] = 1

    return feature


def feat_3(field: np.array, x: int, y: int) -> np.array:
    """
     Agent performs valid action
    """
    feature = np.zeros(len(ACTIONS))
    for idx, action in enumerate(ACTIONS):
        if action == "WAIT":
            continue

        new_x, new_y = get_new_position(action, x, y)
        if new_x < 0 or new_x >= field.shape[0] or new_y < 0 or new_y >= field.shape[1]:  # moving out of field
            continue

        if field[new_x, new_y] == -1:  # moving into wall
            continue

        feature[idx] = 1

    return feature


def feat_4(free_fields: np.array, crates: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves towards crate
    """
    feature = np.zeros(len(ACTIONS))

    if crates:
        for x, y in crates:
            free_fields[x, y] = True

        best_direction = look_for_targets(free_fields, (x, y), crates)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1

    return feature


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = game_state["field"]
    free_fields = field == 0
    crates = [(x, y) for x, y in np.ndindex(field.shape) if field[x, y] == 1]

    stacked_channels = np.vstack((
        [BIAS] * len(ACTIONS),
        feat_1(free_fields, game_state["coins"], *game_state["self"][3]),
        feat_2(game_state["coins"], *game_state["self"][3]),
        feat_3(game_state["field"], *game_state["self"][3]),
        feat_4(free_fields, crates, *game_state["self"][3]),
    ))

    return stacked_channels.T
