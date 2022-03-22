from typing import *

import numpy as np

from settings import *


def get_new_position(action: str, pos: Tuple[int, int]) -> Tuple[int, int]:
    x, y = pos
    switch = {
        'UP': (x, y - 1),
        'DOWN': (x, y + 1),
        'RIGHT': (x + 1, y),
        'LEFT': (x - 1, y),
        'WAIT': (x, y),
        'BOMB': (x, y),
    }

    return switch[action]


def get_neighbor_positions(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = pos
    return [
        (x, y + 1),
        (x + 1, y),
        (x, y - 1),
        (x - 1, y)
    ]


def get_safe_fields(field: np.array, bomb_fields: List[Tuple[int, int]], enemies_pos: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    safe_fields = []
    for x, y in np.ndindex(field.shape):
        if (field[x, y] == 0) and (x, y) not in bomb_fields and (x, y) not in enemies_pos:
            safe_fields.append((x, y))

    return safe_fields


#  look_for_targets is copied from agent_code/rule_based_agent/callbacks.py
def look_for_targets(free_space, start, targets, logger=None) -> Tuple[Tuple[int, int] or None, int or None]:
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
    while True:
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


def is_crate_nearby(field: np.array, pos: Tuple[int, int]) -> bool:
    neighbor_fields = get_neighbor_positions(pos)
    for neighbor_x, neighbor_y in neighbor_fields:
        if field[neighbor_x, neighbor_y] == 1:
            return True

    return False


def is_opponent_nearby(pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]):
    neighbor_fields = get_neighbor_positions(pos)
    for neighbor_pos in neighbor_fields:
        if neighbor_pos in enemies_pos:
            return True

    return False


def wait_is_intelligent(field: np.array, bomb_fields: List[Tuple[int, int]], pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> bool:
    """
    Checks if waiting is intelligent in the current position. It is intelligent when any other action
    might end up in dead (moving into bomb radius or explosion map)
    When the agent is in the bomb radius and there is no neighbor field is safe, wait is not an intelligent action
    """
    if len(bomb_fields) == 0:
        return False

    safe_fields = get_safe_fields(field, bomb_fields, enemies_pos)
    neighbor_fields = get_neighbor_positions(pos)
    for neighbor_field in neighbor_fields:
        if neighbor_field in safe_fields:
            return False

    if pos in bomb_fields:
        return False

    return True


def wait_is_intelligent_alternative(field: np.array, bomb_fields: List[Tuple[int, int]], pos: Tuple[int, int]) -> bool:
    if len(bomb_fields) == 0:
        return False

    reachable_free_fields = give_reachable_free_fields(field, pos, [])

    safe_fields = [(x, y) for (x, y) in reachable_free_fields if (x, y) not in bomb_fields]

    neighbor_fields = get_neighbor_positions(pos)
    for neighbor_field in neighbor_fields:
        if neighbor_field in safe_fields:
            return False

    if pos in bomb_fields:
        return False

    return True


def escape_possible(field: np.array, bomb_fields: List[Tuple[int, int]], pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> bool:
    own_radius = get_blast_radius(field, [(pos, 0)])
    bomb_fields = bomb_fields + own_radius

    safe_fields = get_safe_fields(field, bomb_fields, enemies_pos)
    free_space = field == 0
    best_direction, found_target = look_for_targets(free_space, pos, safe_fields)

    if not best_direction or found_target not in safe_fields:
        return False

    min_dist = np.sum(np.abs(np.subtract(safe_fields, pos)), axis=1).min()

    return min_dist <= BOMB_TIMER


def escape_possible_alternative(field: np.array, pos: Tuple[int, int]) -> bool:
    radius = get_blast_radius(field, [(pos, 0)])

    reachable_free_fields = give_reachable_free_fields(field, pos, [])

    safe_fields = [(x, y) for (x, y) in reachable_free_fields if (x, y) not in radius]

    if len(safe_fields) > 0:
        min_dist = np.sum(np.abs(np.subtract(safe_fields, pos)), axis=1).min()
        return min_dist <= BOMB_TIMER
    else:
        return False


def give_reachable_free_fields(field: np.array, pos: Tuple[int, int], current_free_fields: List[Tuple[int, int]]):
    # list with reachable free fields
    all_free_fields = field == 0
    reachable_free_fields = current_free_fields

    x, y = pos
    if all_free_fields[x, y] and (x, y) not in current_free_fields:
        reachable_free_fields.append((x, y))
        upper_free_fields = give_reachable_free_fields(field, (x, y - 1), reachable_free_fields)
        lower_free_fields = give_reachable_free_fields(field, (x, y + 1), reachable_free_fields)
        rigth_free_fields = give_reachable_free_fields(field, (x + 1, y), reachable_free_fields)
        left_free_fields = give_reachable_free_fields(field, (x - 1, y), reachable_free_fields)

        reachable_free_fields = reachable_free_fields + left_free_fields + rigth_free_fields + lower_free_fields + upper_free_fields

    # remove duplicates
    return [t for t in (set(tuple(i) for i in reachable_free_fields))]
