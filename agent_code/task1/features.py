from random import shuffle
from pathfinding.core.grid import Grid

import numpy as np

from agent_code.task1.agent_settings import *
from settings import *


def give_reachable_free_fields(field: np.array, x: int, y: int, current_free_fields: List[Tuple[int, int]]):
    # list with reachable free fields
    all_free_fields = field == 0
    reachable_free_fields = current_free_fields

    if all_free_fields[x, y] and (x, y) not in current_free_fields:
        reachable_free_fields.append((x, y))
        upper_free_fields = give_reachable_free_fields(field, x, y - 1, reachable_free_fields)
        lower_free_fields = give_reachable_free_fields(field, x, y + 1, reachable_free_fields)
        rigth_free_fields = give_reachable_free_fields(field, x + 1, y, reachable_free_fields)
        left_free_fields = give_reachable_free_fields(field, x - 1, y, reachable_free_fields)

        reachable_free_fields = reachable_free_fields + left_free_fields + rigth_free_fields + lower_free_fields + upper_free_fields

    # remove duplicates
    return [t for t in (set(tuple(i) for i in reachable_free_fields))]


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


def feat_1(field: np.array, coins: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves towards coin
    """
    feature = np.zeros(len(ACTIONS))

    free_space = field == 0
    best_direction = look_for_targets(free_space, (x, y), coins)

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
    """
    feature = np.zeros(len(ACTIONS))

    if coins:
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) in coins:
                feature[idx] = 1

    return feature


def wait_is_intelligent(field: np.array, bombs: List[Tuple[Tuple[int, int], int]], explosion_map: np.array, x: int,
                        y: int) -> bool:
    radius = get_blast_radius(field, bombs)
    if len(radius) == 0:
        return False

    reachable_free_fields = give_reachable_free_fields(field, x, y, [])

    safe_fields_1 = [(x, y) for (x, y) in reachable_free_fields if (x, y) not in radius and explosion_map[x, y] == 0]

    safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if
                   (field[x, y] == 0) and (x, y) not in radius and explosion_map[x, y] == 0]
    neighbor_fields = get_neighbor_positions(x, y)
    for neighbor_field in neighbor_fields:
        if neighbor_field in safe_fields:
            return False

    #TODO: check for bug when staying on bomb and return wait as intelligent
    return True


def feat_3(field: np.array, bombs: List[Tuple[Tuple[int, int], int]], explosion_map: np.array,
           bomb_action_possible: bool, x: int, y: int) -> np.array:
    """
     Agent performs intelligent action
        Agent
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
        if action == "WAIT":
            if not wait_is_intelligent(field, bombs, explosion_map, x, y):
                continue

        if action == "BOMB" and not escape_possible(field, x, y):
            continue

        if action == "BOMB" and not bomb_action_possible:
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
            if field[x + i, y] == -1:
                break
            radius.append((x + i, y))
        for i in range(1, BOMB_POWER + 1):
            if field[x - i, y] == -1:
                break
            radius.append((x - i, y))
        for i in range(1, BOMB_POWER + 1):
            if field[x, y + i] == -1:
                break
            radius.append((x, y + i))
        for i in range(1, BOMB_POWER + 1):
            if field[x, y - i] == -1:
                break
            radius.append((x, y - i))

    return radius


def feat_4(field: np.array, bombs: List[Tuple[Tuple[int, int], int]], explosion_map: np.array, x: int,
           y: int) -> np.array:
    """
    Agent stays/moves out of the blast radius (and does not move into explosion radius)
    """
    feature = np.zeros(len(ACTIONS))

    if len(bombs) == 0:
        return feature

    radius = get_blast_radius(field, bombs)

    if (x, y) in radius:  # TODO multiple blast radius
        safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if
                       (field[x, y] == 0) and (x, y) not in radius and explosion_map[x, y] == 0]
        free_space = field == 0
        best_direction = look_for_targets(free_space, (x, y), safe_fields)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1

    return feature


def feat_5(fields: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves towards bomb -> Negative
    """

    # TODO: only bombs that are next to the agent

    feature = np.ones(len(ACTIONS))
    if len(bomb_fields) > 0:
        free_space = fields == 0
        best_direction = look_for_targets(free_space, (x, y), bomb_fields)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)

            if (new_x, new_y) == best_direction:
                feature[idx] = 0

    # set bomb and wait to zero, since only real moves should be evaluated
    feature[ACTIONS.index("BOMB")] = 0
    feature[ACTIONS.index("WAIT")] = 0
    return feature


def is_crate_nearby(field: np.array, x: int, y: int) -> bool:

    neighbor_fields = get_neighbor_positions(x, y)
    for neighbor_field_x, neighbor_field_y in neighbor_fields:

        if field[neighbor_field_x][neighbor_field_y] == 1:
            return True

    return False


def escape_possible(field: np.array, x: int, y: int) -> bool:
    radius = get_blast_radius(field, [((x, y), 0)])

    reachable_free_fields = give_reachable_free_fields(field, x, y, [])

    safe_fields = [(x, y) for (x, y) in reachable_free_fields if (x, y) not in radius]

    if len(safe_fields) > 0:
        min_dist = np.sum(np.abs(np.subtract(safe_fields, (x, y))), axis=1).min()
        return min_dist <= BOMB_TIMER
    else:
        return False

def feat_6(field: np.array, bomb_action_possible: bool, x: int, y: int) -> np.array:
    """
    Agent places bomb next to crate if he can escape

    1. check if there is a crate nearby
    2. calculate bomb radius
    2. check if it possible to escape
    """
    feature = np.zeros(len(ACTIONS))

    if not bomb_action_possible:
        return feature

    if is_crate_nearby(field, x, y):
        if escape_possible(field, x, y):
            feature[ACTIONS.index("BOMB")] = 1

    return feature


def feat_7(field: np.array, bomb_action_possible: bool, x: int, y: int) -> np.array:
    """
    Agent moves towards crate, if he is able to place a bomb and there is no bomb in the same direction
    """
    crates = [(x, y) for x, y in np.ndindex(field.shape) if
              (field[x, y] == 1)]

    feature = np.zeros(len(ACTIONS))

    free_space = field == 0
    best_direction = look_for_targets(free_space, (x, y), crates)

    if not bomb_action_possible:
        return feature

    #TODO: check if there is a bomb in the same direction or if bomb action is possible

    if crates:
        for idx, action in enumerate(ACTIONS):
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

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    blast_radius = get_blast_radius(field, bombs)
    # [()()()]
    #
    bomb_fields = [(x, y) for x, y in np.ndindex(explosion_map.shape) if
                   (explosion_map[x, y] != 0) or (x, y) in blast_radius]

    bomb_action_possible = game_state["self"][2]

    stacked_channels = np.vstack((
        [BIAS] * len(ACTIONS),
        feat_1(game_state["field"], game_state["coins"], *game_state["self"][3]),
        feat_2(game_state["coins"], *game_state["self"][3]),
        feat_3(game_state["field"], game_state["bombs"], game_state["explosion_map"], bomb_action_possible,
               *game_state["self"][3]),
        feat_4(game_state["field"], game_state["bombs"], game_state["explosion_map"], *game_state["self"][3]),
        feat_5(field, bomb_fields, *game_state["self"][3]),
        feat_6(game_state["field"], bomb_action_possible, *game_state["self"][3]),
        feat_7(game_state["field"], bomb_action_possible, *game_state["self"][3]),
        # feat_8(game_state["field"], bomb_action_possible, *game_state["self"][3])
    ))

    return stacked_channels.T
