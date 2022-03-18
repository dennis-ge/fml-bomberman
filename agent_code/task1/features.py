from random import shuffle
from typing import List

import numpy as np
from scipy.spatial.distance import cityblock

from agent_code.task1.agent_settings import *
from settings import *


#  copied from rule_based_agent/callbacks.py
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


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

        if field[new_x, new_y] == -1 or field[new_x, new_y] == 1:  # moving into wall or crate
            continue

        # TODO move into enemy position

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


def feat_4(field: np.array, bombs: List[Tuple[Tuple[int, int], int]], explosion_map: np.array, x: int, y: int) -> np.array:
    """
    Agent stays/moves out of the blast radius (and does not move into explosion radius)
    """
    feature = np.zeros(len(ACTIONS))

    if len(bombs) == 0:
        return feature

    radius = get_blast_radius(field, bombs)

    if (x, y) in radius:
        safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 0) and (x, y) not in radius and explosion_map[x, y] == 0]  # TODO include explosion map
        free_space = field == 0
        best_direction = look_for_targets(free_space, (x, y), safe_fields)

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
    Converts the game state to the input of your model, i.e. a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    stacked_channels = np.vstack((
        [BIAS] * len(ACTIONS),
        feat_1(game_state["field"], game_state["coins"], *game_state["self"][3]),
        feat_2(game_state["coins"], *game_state["self"][3]),
        feat_3(game_state["field"], *game_state["self"][3]),
        feat_4(game_state["field"], game_state["bombs"], game_state["explosion_map"], *game_state["self"][3]),
        feat_5(game_state["field"], game_state["explosion_map"], *game_state["self"][3]),
    ))

    return stacked_channels.T
