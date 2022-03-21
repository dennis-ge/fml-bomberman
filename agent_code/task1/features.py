from scipy.spatial.distance import cityblock

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

    agent_pos: Tuple[int, int] = game_state["self"][3]
    bomb_action_possible = game_state["self"][2]

    coins = game_state["coins"]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    enemies_pos = [enemy[3] for enemy in game_state["others"]]

    bomb_fields = get_bomb_fields(field, bombs, explosion_map)

    stacked_channels = np.vstack((
        [env.BIAS] * len(ACTIONS),
        feat_1(field, coins, agent_pos),
        feat_2(coins, agent_pos),
        feat_3(field, bomb_fields, bomb_action_possible, agent_pos, enemies_pos),
        feat_4(field, bomb_fields, agent_pos, enemies_pos),
        feat_5(field, bomb_fields, agent_pos),
        # feat_6(field, bomb_fields, agent_pos),
        feat_7(field, bomb_fields, bomb_action_possible, agent_pos, enemies_pos),
        feat_8(field, bomb_action_possible, agent_pos),
        feat_9(field, bomb_fields, bomb_action_possible, agent_pos, enemies_pos),
    ))

    return stacked_channels.T


def feat_1(field: np.array, coins: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.array:
    """
    Agent moves towards coin
    - only one entry in the feature is equal to 1
    """
    feature = np.zeros(len(ACTIONS))

    if coins:
        free_space = field == 0
        best_direction, _ = look_for_targets(free_space, agent_pos, coins)
        for idx, action in enumerate(ACTIONS):
            if action == "WAIT" or action == "BOMB":
                feature[idx] = 0

            new_x, new_y = get_new_position(action, agent_pos)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1

    return feature


def feat_2(coins: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.array:
    """
    Agent collects coin
    - multiple entries in the feature can be 1
    """
    feature = np.zeros(len(ACTIONS))

    if coins:
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, agent_pos)
            if (new_x, new_y) in coins:
                feature[idx] = 1

    return feature


def feat_3(field: np.array, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, agent_pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.array:
    """
     Agent performs intelligent action. An intelligent action is an action where the agent does
     not move out of the field, into walls or crates and also does not die.
    """
    feature = np.zeros(len(ACTIONS))

    for idx, action in enumerate(ACTIONS):
        new_x, new_y = get_new_position(action, agent_pos)

        if new_x < 0 or new_x >= field.shape[0] or new_y < 0 or new_y >= field.shape[1]:  # moving out of field
            continue

        if field[new_x, new_y] == -1 or field[new_x, new_y] == 1:  # moving into wall or crate
            continue

        if (new_x, new_y) in enemies_pos:
            continue

        if action == "WAIT" and not wait_is_intelligent(field, bomb_fields, agent_pos, enemies_pos):
            continue

        # TODO check if bomb is in the current position
        if action == "BOMB" and not bomb_action_possible:
            continue

        if action == "BOMB" and not escape_possible(field, bomb_fields, agent_pos, enemies_pos):
            continue

        feature[idx] = 1

    return feature


def feat_4(field: np.array, bomb_fields: List[Tuple[int, int]], agent_pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.array:
    """
    Agent moves out of the blast radius (and does not move into other)
    TODO there are maybe multiple correct directions
    """
    feature = np.zeros(len(ACTIONS))

    if len(bomb_fields) == 0:
        return feature

    if agent_pos in bomb_fields:
        safe_fields = get_safe_fields(field, bomb_fields, enemies_pos)
        free_space = field == 0
        best_direction, _ = look_for_targets(free_space, agent_pos, safe_fields)

        for idx, action in enumerate(ACTIONS):
            if action == "BOMB":  # don't drop bomb when already in bomb radius
                continue

            new_x, new_y = get_new_position(action, agent_pos)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1
                break

    return feature


def feat_5(fields: np.array, bomb_fields: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.array:
    """
    Agent moves/stays out of bomb radius/explosion map
    """
    feature = np.ones(len(ACTIONS))

    bomb_fields_nearby = []
    for bomb_x, bomb_y in bomb_fields:
        distance = cityblock([agent_pos[0], agent_pos[1]], [bomb_x, bomb_y])
        if distance < 6:
            bomb_fields_nearby.append((bomb_x, bomb_y))

    if len(bomb_fields_nearby) > 0:
        free_space = fields == 0
        best_direction, _ = look_for_targets(free_space, agent_pos, bomb_fields_nearby)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, agent_pos)

            if (new_x, new_y) == best_direction:
                feature[idx] = 0

    # set bomb to zero, since only real moves should be evaluated
    feature[ACTIONS.index("BOMB")] = 0
    return feature


def feat_6(field: np.array, bomb_fields: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.array:
    """
    Agent does not move into bomb radius/explosion map.
    Requirement: Agent is currently not in bomb radius/explosion map
    - can contain multiple 1, each one indicating that the agent stays out of the bomb fields with that action
    """
    feature = np.zeros(len(ACTIONS))

    if len(bomb_fields) == 0:
        return feature

    if agent_pos not in bomb_fields:
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, agent_pos)

            if field[new_x, new_y] == -1 or field[new_x, new_y] == 1:  # moving into wall or crate
                continue

            if action == "BOMB":
                continue

            if (new_x, new_y) not in bomb_fields:
                feature[idx] = 1

    return feature


def feat_7(field: np.array, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, agent_pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.array:
    """
    Agent places bomb next to crate if he can escape
    """
    feature = np.zeros(len(ACTIONS))

    if not bomb_action_possible:
        return feature

    if is_crate_nearby(field, agent_pos):
        if escape_possible(field, bomb_fields, agent_pos, enemies_pos):
            feature[ACTIONS.index("BOMB")] = 1

    return feature


def feat_8(field: np.array, bomb_action_possible: bool, agent_pos: Tuple[int, int]) -> np.array:
    """
    Agent moves towards crate, if he is able to place a bomb
    """
    feature = np.zeros(len(ACTIONS))

    if not bomb_action_possible:
        return feature

    crates = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 1)]

    if len(crates) == 0:
        return feature

    free_space = field == 0
    best_direction, _ = look_for_targets(free_space, agent_pos, crates)

    for idx, action in enumerate(ACTIONS):
        if action == "BOMB" or action == "WAIT":
            continue

        new_x, new_y = get_new_position(action, agent_pos)
        if (new_x, new_y) == best_direction:
            feature[idx] = 1

    return feature


def feat_9(field: np.array, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.array:
    """
    Agent places bomb next to opponent if he can escape
    """
    feature = np.zeros(len(ACTIONS))

    if not bomb_action_possible:
        return feature

    if is_opponent_nearby(pos, enemies_pos):
        if escape_possible(field, bomb_fields, pos, enemies_pos):
            feature[ACTIONS.index("BOMB")] = 1

    return feature

# Reward for moving towards the nearest opponent: under certain conditions
# Use transitions of the other agents: think about weights
# Reward for setting bombs that can kill an opponent
# Reward bombs that are leading for a "safe" dead of an agent
# dead end feature
# Move away if other agent is nearby that has a bomb action available
# consider other agent pos
