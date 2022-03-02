from typing import List

from agent_code.my_agent.agent_settings import *
from scipy.spatial.distance import cityblock
import numpy as np


def feat_1(coins, x: int, y: int) -> np.array:
    """
    Moved towards coin
    """
    feature = np.zeros(len(ACTIONS))

    if coins:
        old_min_d = calc_min_distance(coins, x, y)
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            new_min_d = calc_min_distance(coins, new_x, new_y)
            if new_min_d < old_min_d:
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

    stacked_channels = np.vstack((
        [BIAS] * len(ACTIONS),
        feat_1(game_state["coins"], *game_state["self"][3])
    ))

    return stacked_channels.T


def calc_min_distance(coins: List[Tuple[int, int]], x: int, y: int) -> int:
    min_d = 10000000000  # TODO set to inf
    for (x_coin, y_coin) in coins:
        # d = np.linalg.norm(np.array((x, y)) - np.array((x_coin, y_coin)), ord=1)
        d = cityblock([x,y], [x_coin, y_coin])
        if d < min_d:
            min_d = d

    return min_d
