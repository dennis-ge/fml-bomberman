from typing import List

import numpy as np

from agent_code.my_agent.agent_settings import *


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

    def feature1() -> np.array:
        # decide based on the current game_state which action is good
        feature = np.zeros(len(ACTIONS))

        try:
            if game_state["coins"]:
                old_min_d = calc_min_distance(game_state["coins"], *game_state["self"][3])
                for idx, action in enumerate(ACTIONS):
                    new_x, new_y = get_new_position(action, *game_state["self"][3])
                    new_min_d = calc_min_distance(game_state["coins"], new_x, new_y)
                    feature[idx] = 1 if new_min_d < old_min_d else 0
        except e:
            print(e)

        return feature

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    stacked_channels = np.stack((
        feature1()
    ))

    # feature vector: array of arrays for each action: [f1[a1, a2,...], f2[a1,...]]
    return stacked_channels.reshape(-1)


def calc_min_distance(coins: List[Tuple[int, int]], x: int, y: int) -> int:
    min_d = 10000000000  # TODO set to inf
    for (x_coin, y_coin) in coins:
        d = np.linalg.norm(np.array((x, y)) - np.array((x_coin, y_coin)), ord=1)
        if d < min_d:
            min_d = d

    return min_d
