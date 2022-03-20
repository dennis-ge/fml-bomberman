from typing import *

import numpy as np

from settings import *


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


def get_neighbor_positions(x: int, y: int) -> List[Tuple[int, int]]:
    return [
        (x, y + 1),
        (x + 1, y),
        (x, y - 1),
        (x - 1, y)
    ]


#  look_for_targets is copied from agent_code/rule_based_agent/callbacks.py
def look_for_targets(free_space, start, targets, logger=None) -> Tuple[Tuple[int,int] or None, int or None]:
    """Find direction of the closest target that can be reached via free tiles.

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
    if len(targets) == 0:
        return None, None

    frontier = [start]
    parent_dict = {start: start}  # stores the parent/previous tile for each child/current tile
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    # construct tree
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
        np.random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    path = []
    while True:
        path.insert(0, current)
        if parent_dict[current] == start:  # next direction starting from start
            return current, best
        current = parent_dict[current]


def get_blast_radius(field: np.array, bombs) -> List[Tuple[int, int]]:
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


def get_bomb_fields(field: np.array, bombs, explosion_map: np.array) -> List[Tuple[int, int]]:
    """
    fields that are in the radius of a bomb that is currently exploding or about to explode
    """
    blast_radius = get_blast_radius(field, bombs)
    return [(x, y) for x, y in np.ndindex(explosion_map.shape) if (explosion_map[x, y] != 0) or (x, y) in blast_radius]


def is_crate_nearby(field: np.array, x: int, y: int) -> bool:
    crates = field == 1

    neighbor_fields = get_neighbor_positions(x, y)
    for neighbor_field in neighbor_fields:
        if neighbor_field in crates:
            return True

    return False


def wait_is_intelligent(field: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> bool:
    """
    Checks if waiting is intelligent in the current position. It is intelligent when
    - any other action might end up in dead (moving into bomb radius or explosion map)

    TODO bug: if there is no escape possible, wait is an intelligent action
    """
    if len(bomb_fields) == 0:
        return False

    safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 0) and (x, y) not in bomb_fields]
    neighbor_fields = get_neighbor_positions(x, y)
    for neighbor_field in neighbor_fields:
        if neighbor_field in safe_fields:
            return False

    return True


def bomb_is_intelligent(field: np.array, bomb_action_available: bool, bomb_fields: List[Tuple[int, int]], x, y) -> bool:
    """
    Checks if a bomb is intelligent in the current position. It is intelligent when
    - bomb action is available
    - agent can escape from the blast radius
    - agent destroys anything of relevance (crate, TODO opponent)
    # TODO function necessary?
    """
    if not bomb_action_available:
        return False

    if escape_possible(field, bomb_fields, x, y):
        radius = get_blast_radius(field, [((x, y), 0)])
        crates = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 1)]

        for bomb_rad in radius:
            if bomb_rad in crates:
                return True

    return False


def escape_possible(field: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> bool:
    own_radius = get_blast_radius(field, [((x, y), 0)])
    bomb_fields = bomb_fields + own_radius

    safe_fields = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 0) and (x, y) not in bomb_fields]  # TODO improve by not calculating whole game board
    free_space = field == 0
    best_direction, found_target = look_for_targets(free_space, (x, y), safe_fields)

    if not best_direction or found_target not in safe_fields:
        return False

    min_dist = np.sum(np.abs(np.subtract(safe_fields, (x, y))), axis=1).min()

    return min_dist <= BOMB_TIMER
