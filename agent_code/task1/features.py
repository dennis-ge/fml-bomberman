from random import shuffle

import numpy as np

from agent_code.task1.agent_settings import *
from agent_code.task1.game_info import *


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    agent_x, agent_y = game_state["self"][3]
    bomb_action_possible = game_state["self"][2]

    coins = game_state["coins"]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    bomb_fields = get_bomb_fields(field, bombs, explosion_map)

    stacked_channels = np.vstack((
        [BIAS] * len(ACTIONS),
        feat_1(field, coins, agent_x, agent_y),
        feat_2(coins, agent_x, agent_y),
        feat_3(field, bomb_fields, bomb_action_possible, agent_x, agent_y),
        feat_4(field, bomb_fields, agent_x, agent_y),
        feat_5(field, bomb_fields, agent_x, agent_y),
        feat_6(field, bomb_fields, bomb_action_possible, agent_x, agent_y),
        # feat_7(field, bomb_action_possible, agent_x, agent_y),
    ))

    return stacked_channels.T


def feat_1(field: np.array, coins: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves towards coin
    - only one entry in the feature is equal to 1
    """
    feature = np.zeros(len(ACTIONS))

    free_space = field == 0
    best_direction, _ = look_for_targets(free_space, (x, y), coins)

    # take a look into look_for_targets_bug

    if coins:
        for idx, action in enumerate(ACTIONS):

            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1
            if action == "WAIT" or action == "BOMB":
                feature[idx] = 0

    return feature


def feat_2(coins: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent collects coin
    - multiple entries in the feature can be 1
    """
    feature = np.zeros(len(ACTIONS))

    if coins:
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) in coins:
                feature[idx] = 1

    return feature


def feat_3(field: np.array, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, x: int, y: int) -> np.array:
    """
     Agent performs intelligent action. An intelligent action is an action where the agent does
     not move out of the field, into walls or crates and also does not die.
    """
    feature = np.zeros(len(ACTIONS))

    for idx, action in enumerate(ACTIONS):
        new_x, new_y = get_new_position(action, x, y)

        if new_x < 0 or new_x >= field.shape[0] or new_y < 0 or new_y >= field.shape[1]:  # moving out of field
            continue

        if field[new_x, new_y] == -1 or field[new_x, new_y] == 1:  # moving into wall or crate
            continue

        # TODO move into enemy position
        # TODO check if bomb is in the current position

        if action == "WAIT" and not wait_is_intelligent(field, bomb_fields, x, y):
            continue

        if action == "BOMB" and not escape_possible(field, x, y):
            continue

        if action == "BOMB" and not bomb_action_possible:
            continue

        feature[idx] = 1

    return feature


def feat_4(field: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Positive: Agent stays/moves out of the blast radius (and does not move into explosion radius)
    """
    feature = np.zeros(len(ACTIONS))

    if len(bomb_fields) == 0:
        return feature

    if (x, y) in bomb_fields:
        safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 0) and (x, y) not in bomb_fields]
        free_space = field == 0
        best_direction, _ = look_for_targets(free_space, (x, y), safe_fields)

        for idx, action in enumerate(ACTIONS):
            if action == "BOMB":  # don't drop bomb when already in bomb radius
                continue

            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1

    return feature


def feat_5(fields: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Negative: Agent moves towards bomb/explosion
    """
    feature = np.zeros(len(ACTIONS))  # TODO: only bombs that are next to the agent

    if len(bomb_fields) > 0:
        free_space = fields == 0
        best_direction, _ = look_for_targets(free_space, (x, y), bomb_fields)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)

            if (new_x, new_y) == best_direction:
                feature[idx] = 1

    # set bomb and wait to zero, since only real moves should be evaluated
    feature[ACTIONS.index("BOMB")] = 0
    feature[ACTIONS.index("WAIT")] = 0
    return feature


def escape_possible(field: np.array, x: int, y: int) -> bool:
    radius = get_blast_radius(field, [((x, y), 0)])

    reachable_free_fields = give_reachable_free_fields(field, x, y, [])

    safe_fields = [(x, y) for (x, y) in reachable_free_fields if (x, y) not in radius]

    if len(safe_fields) > 0:
        min_dist = np.sum(np.abs(np.subtract(safe_fields, (x, y))), axis=1).min()
        return min_dist <= BOMB_TIMER
    else:
        return False


def feat_6(field: np.array, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, x: int, y: int) -> np.array:
    """
    Agent places bomb next to crate if he can escape
    """
    feature = np.zeros(len(ACTIONS))

    if not bomb_action_possible:
        return feature

    if is_crate_nearby(field, x, y):
        if escape_possible(field, bomb_fields, x, y):
            feature[ACTIONS.index("BOMB")] = 1

    return feature


def feat_7(field: np.array, bomb_action_possible: bool, x: int, y: int) -> np.array:
    """
    Agent moves towards crate, if he is able to place a bomb and there is no bomb in the same direction
    """
    feature = np.zeros(len(ACTIONS))

    crates = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 1)]

    free_space = field == 0
    best_direction, _ = look_for_targets(free_space, (x, y), crates)

    if not bomb_action_possible:
        return feature

    # TODO: check if there is a bomb in the same direction or if bomb action is possible

    if crates:
        for idx, action in enumerate(ACTIONS):
            if action == "WAIT":
                continue

            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1
            if action == "WAIT" or action == "BOMB":
                feature[idx] = 0
    return feature



def feat_8(field: np.array, bomb_action_possible: bool, x: int, y: int) -> np.array:
    """
    Agent places bomb that is useless

    1. check if agent can place bomb
    2. calculate bomb radius
    3. check if create gets destroyed by bomb
    4. TODO check if opponents can be destroyed
    """

    feature = np.zeros(len(ACTIONS))

    if not bomb_action_possible:
        return feature

    radius = get_blast_radius(field, [((x, y), 0)])

    crates = [(x, y) for x, y in np.ndindex(field.shape) if
              (field[x, y] == 1)]

    for crate in crates:
        if crate in radius and escape_possible(field, x, y):
            feature[ACTIONS.index("BOMB")] = 1
    return feature

# TODO: negative reward for setting bombs that cannot destroy any create or kill an oponnent
# TODO: add feature for entering/not entering blast radius
