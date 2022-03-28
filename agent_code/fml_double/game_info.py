from typing import *

import numpy as np

from agent_code.fml_double.agent_settings import *
from settings import *


def beautify_output(field: np.array, features: np.array, weights1: np.array, weights2: np.array, q_values: np.array):
    out = ""
    if env.PRINT_FIELD:
        out += f"Field {field}\n"
    out += f"Weights 1 {[f'{round(weight, 2)} ({idx})' for idx, weight in enumerate(weights1)]}\n"
    out += f"Weights 2 {[f'{round(weight, 2)} ({idx})' for idx, weight in enumerate(weights2)]}\n"
    out += "Feature   " + "\t ".join([f'{i}' for i in range(len(features[0]))]) + "\n"
    for i in range(len(features)):
        out += f"{ACTIONS[i]:6}: {features[i]}".replace("0.", " .") + f": {round(q_values[i], 3)}\n"
    return out[:-1]


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


def get_neighbor_positions_info(pos: Tuple[int, int], field: np.array, explosions: List[Tuple[int, int]], blast_radius: List[Tuple[int, int]]):
    neighbors = get_neighbor_positions(pos)
    free = []
    endangered = []
    for x, y in neighbors:
        if x < 0 or x >= field.shape[0]:
            continue
        if y < 0 or y >= field.shape[1]:
            continue

        if field[x, y] == 0 and (x, y) not in explosions:
            free.append((x, y))
            continue

        if field[x, y] == 0 and (x, y) in blast_radius:
            endangered.append((x, y))

    return free, endangered


def get_trapped_positions(field: np.array, positions: List[Tuple[int, int]], explosions: List[Tuple[int, int]], blast_radius: List[Tuple[int, int]]) -> Tuple[
    List[Tuple[int, int]], List[Tuple[int, int]]]:
    trapped_positions = []
    possible_trapped_positions = []

    for pos in positions:
        free, endangered = get_neighbor_positions_info(pos, field, explosions, blast_radius)

        if len(free) == 1:  # only when there is one free position left, the position is trapped
            trapped_positions.append(free[0])
        elif len(free) == 0:
            trapped_positions.append(pos)  # pos has no free position left. only right action would be wait
        if len(endangered) == 1:
            possible_trapped_positions.append(endangered[0])
    return trapped_positions, possible_trapped_positions


def trapped_positions_reachable(pos: Tuple[int, int], trapped_positions: List[Tuple[int, int]], possible_trapped_positions: List[Tuple[int, int]]) -> bool:
    """
    Checks whether the given trapped positions are reachable
    """
    neighbor_positions = get_neighbor_positions(pos)
    if len(trapped_positions) > 0:
        for neighbor_pos in neighbor_positions:
            if neighbor_pos in trapped_positions:
                return True

    if len(possible_trapped_positions) > 0:
        for neighbor_pos in neighbor_positions:
            if neighbor_pos in possible_trapped_positions:
                return True

    return False


def get_extended_neighbor_positions(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = pos
    return [
        (x, y + 1),
        (x, y + 2),
        (x + 1, y),
        (x + 2, y),
        (x, y - 1),
        (x, y - 2),
        (x - 1, y),
        (x - 2, y),
    ]


def get_safe_positions(field: np.array, bomb_positions: List[Tuple[int, int]], opponents_pos: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    safe_positions = []
    for x, y in np.ndindex(field.shape):
        if (field[x, y] == 0) and (x, y) not in bomb_positions and (x, y) not in opponents_pos:
            safe_positions.append((x, y))

    return safe_positions


#  look_for_targets is copied from agent_code/rule_based_agent/callbacks.py
def look_for_targets(free_space, start, targets, distance_satisfied: int = 0) -> Tuple[Tuple[int, int] or None, int or None]:
    """Find direction of the closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        distance_satisfied: defines the distance to a target when it can be seen as reached.
            e.g. - when looking for coins, we need to reach the exact target tile
                 - when looking for crates, the neighbor tiles are enough
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
    reachable = False
    # construct tree
    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == distance_satisfied:
            # Found path to a target's exact position, mission accomplished!
            best = current
            reachable = True
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
    # if we can't reach a target, return None
    if not reachable:
        return None, None

    # Determine the first step towards the best found target tile
    current = best

    while True:
        if parent_dict[current] == start:  # next direction starting from start
            return current, best
        current = parent_dict[current]


def get_blast_radius(field: np.array, bombs) -> Tuple[List[Tuple[int, int]],List[Tuple[int, int]]]:
    bombs_pos = []
    radius = []
    for pos, countdown in bombs:
        x, y = pos
        bombs_pos.append((x,y))
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

    return radius, bombs_pos


def is_crate_nearby(field: np.array, pos: Tuple[int, int]) -> bool:
    neighbor_positions = get_neighbor_positions(pos)
    for neighbor_x, neighbor_y in neighbor_positions:
        if field[neighbor_x, neighbor_y] == 1:
            return True

    return False


def is_opponent_nearby(pos: Tuple[int, int], opponents_pos: List[Tuple[int, int]]) -> bool:
    extended_neighbors = get_extended_neighbor_positions(pos)
    for neighbor_pos in extended_neighbors:
        if neighbor_pos in opponents_pos:
            return True

    return False


def get_nearby_field(field_max_x: int, field_max_y: int, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    minimized_field = []
    x, y = pos
    min_x = x - 5 if (x - 5) > 0 else 0
    max_x = x + 5 if (x + 5) < field_max_x else field_max_x
    min_y = y - 5 if (y - 5) > 0 else 0
    max_y = y + 5 if (y + 5) < field_max_y else field_max_y

    for curr_x in range(min_x, max_x):
        for curr_y in range(min_y, max_y):
            minimized_field.append((curr_x, curr_y))

    return minimized_field


def get_nearby_opponents(field_max_x: int, field_max_y: int, pos: Tuple[int, int], opponents_pos: List[Tuple[int, int]]):
    opponents_nearby = []
    nearby_field = get_nearby_field(field_max_x=field_max_x, field_max_y=field_max_y, pos=pos)
    for opponent_pos in opponents_pos:
        if opponent_pos in nearby_field:
            opponents_nearby.append(opponent_pos)
    return opponents_nearby


def wait_is_smart(pos: Tuple[int, int], bomb_positions: List[Tuple[int, int]], safe_positions: List[Tuple[int, int]]) -> bool:
    """
    Checks if waiting is intelligent in the current position. It is intelligent when any other action
    might end up in dead (moving into bomb radius or explosion map)
    When the agent is in the bomb radius and there is no neighbor position is safe, wait is not an intelligent action
    """
    if len(bomb_positions) == 0:
        return False

    neighbor_positions = get_neighbor_positions(pos)
    for neighbor_pos in neighbor_positions:
        if neighbor_pos in safe_positions:
            return False

    if pos in bomb_positions:
        return False

    return True


def is_escape_possible(field: np.ndarray, bomb_positions: List[Tuple[int, int]], pos: Tuple[int, int],
                       opponents_pos: List[Tuple[int, int]]) -> bool:
    own_radius, _ = get_blast_radius(field, [(pos, 0)])
    bomb_positions = bomb_positions + own_radius

    safe_positions = get_safe_positions(field, bomb_positions, opponents_pos)
    free_space = field == 0
    best_direction, found_target = look_for_targets(free_space, pos, safe_positions)

    if not best_direction or found_target not in safe_positions:
        return False

    min_dist = np.sum(np.abs(np.subtract(safe_positions, pos)), axis=1).min()

    return min_dist <= BOMB_TIMER
