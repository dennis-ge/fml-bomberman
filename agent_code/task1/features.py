from typing import List

import numpy as np
from scipy.spatial.distance import cityblock

from agent_code.rule_based_agent.callbacks import look_for_targets
from agent_code.task1.agent_settings import *
from settings import *


def calc_min_distance(coins: List[Tuple[int, int]], x: int, y: int) -> int:
    min_d = 10000000000  # TODO set to inf
    for (x_coin, y_coin) in coins:
        # d = np.linalg.norm(np.array((x, y)) - np.array((x_coin, y_coin)), ord=1)
        d = cityblock([x, y], [x_coin, y_coin])
        if d < min_d:
            min_d = d

    return min_d


def feat_1(field: np.array, coins: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves towards coin
    """
    feature = np.zeros(len(ACTIONS))

    free_space = field == 0
    best_direction = look_for_targets(free_space, (x, y), coins)

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
     Agent performs invalid action
    """
    feature = np.zeros(len(ACTIONS))

    for idx, action in enumerate(ACTIONS):
        if action == "WAIT":
            continue

        new_x, new_y = get_new_position(action, x, y)
        if new_x < 0 or new_x >= field.shape[0] or new_y < 0 or new_y >= field.shape[1]:  # moving out of field
            continue

        if field[new_x, new_y] == -1:  # moving into wall TODO add crate, enemy position
            continue

        feature[idx] = 1

    return feature


def get_blast_radius(field: np.array, bombs):
    radius = []
    for pos, countdown in bombs:
        x = pos[0]
        y = pos[1]
        radius.append((x, y))
        for i in range(1, BOMB_POWER + 1):
            if x + i < field.shape[0] and field[x + i, y] >= 0:
                radius.append((x + i, y))
            if x - i > 0 and field[x - i, y] >= 0:
                radius.append((x - i, y))
            if y + i < field.shape[1] and field[x, y + i] >= 0:
                radius.append((x, y + i))
            if y - i > 0 and field[x, y - i] >= 0:
                radius.append((x, y - i))

    return radius


def shortest_path_out_blast_radius(field: np.array, x, y, radius: List[Tuple[int, int]]) -> Tuple[int, int]:
    safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 0) and (x, y) not in radius]  # TODO include explosion map

    free_space = field == 0
    return look_for_targets(free_space, (x, y), safe_fields)


def feat_4(field: np.array, bombs: List[Tuple[Tuple[int, int], int]], x: int, y: int) -> np.array:
    """
    Agent stays/moves out of the blast radius
    TODO explosion map
    """
    feature = np.zeros(len(ACTIONS))

    if len(bombs) == 0:
        return feature

    radius = get_blast_radius(field, bombs)
    if (x, y) in radius:
        best_direction = shortest_path_out_blast_radius(field, x, y, radius)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1

    return feature


def feat_5(field: np.array, explosion_map: np.array, x: int, y: int) -> np.array:
    """
    Agent stays out of explosion radius
    """
    feature = np.zeros(len(ACTIONS))

    safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 0) and explosion_map[x, y] == 0]

    for idx, action in enumerate(ACTIONS):
        new_x, new_y = get_new_position(action, x, y)
        if (new_x, new_y) in safe_fields:
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

    explosion_map = game_state["explosion_map"]

    stacked_channels = np.vstack((
        [BIAS] * len(ACTIONS),
        feat_1(game_state["field"], game_state["coins"], *game_state["self"][3]),
        feat_2(game_state["coins"], *game_state["self"][3]),
        feat_3(game_state["field"], *game_state["self"][3]),
        feat_4(game_state["field"], game_state["bombs"], *game_state["self"][3]),
        feat_5(game_state["field"], game_state["explosion_map"], *game_state["self"][3]),
    ))

    return stacked_channels.T
