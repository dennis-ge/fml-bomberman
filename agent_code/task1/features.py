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

    agent_x, agent_y = game_state["self"][3]
    bomb_action_possible = game_state["self"][2]

    coins = game_state["coins"]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    bomb_fields = get_bomb_fields(field, bombs, explosion_map)

    stacked_channels = np.vstack((
        [env.BIAS] * len(ACTIONS),
        feat_1(field, coins, agent_x, agent_y),
        feat_2(coins, agent_x, agent_y),
        feat_3(field, bomb_fields, bomb_action_possible, agent_x, agent_y),
        feat_4(field, bomb_fields, agent_x, agent_y),
        feat_5(field, bomb_fields, agent_x, agent_y),
        feat_6(field, bomb_fields, bomb_action_possible, agent_x, agent_y),
        # feat_7(field, bomb_action_possible, agent_x, agent_y),
        feat_9(field, bomb_fields, agent_x, agent_y)
    ))

    return stacked_channels.T


def feat_1(field: np.array, coins: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves towards coin
    - only one entry in the feature is equal to 1
    """
    feature = np.zeros(len(ACTIONS))

    if coins:
        free_space = field == 0
        best_direction, _ = look_for_targets(free_space, (x, y), coins)
        for idx, action in enumerate(ACTIONS):
            if action == "WAIT" or action == "BOMB":
                feature[idx] = 0

            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1

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

        if action == "BOMB" and not bomb_action_possible:
            continue

        if action == "BOMB" and not escape_possible(field, bomb_fields, x, y):
            continue

        feature[idx] = 1

    return feature


def feat_4(field: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent moves out of the blast radius (and does not move into other)
    TODO there are maybe multiple correct directions
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
                break

    return feature


def feat_5(fields: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Agent stays out of bomb radius/explosion map
    """
    # TODO: only bombs that are next to the agent
    feature = np.ones(len(ACTIONS))

    bomb_fields_nearby = []
    for bomb_x, bomb_y in bomb_fields:
        distance = cityblock([x, y], [bomb_x, bomb_y])
        if distance < 6:
            bomb_fields_nearby.append((bomb_x, bomb_y))

    if len(bomb_fields_nearby) > 0:
        free_space = fields == 0
        best_direction, _ = look_for_targets(free_space, (x, y), bomb_fields_nearby)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)

            if (new_x, new_y) == best_direction:
                feature[idx] = 0

    # set bomb to zero, since only real moves should be evaluated
    feature[ACTIONS.index("BOMB")] = 0
    return feature


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
            if action == "BOMB":
                continue

            new_x, new_y = get_new_position(action, x, y)
            if (new_x, new_y) == best_direction:
                feature[idx] = 1
            if action == "WAIT" or action == "BOMB":
                feature[idx] = 0
    return feature


def feat_9(field: np.array, bomb_fields: List[Tuple[int, int]], x: int, y: int) -> np.array:
    """
    Reward the agent for not moving into bomb radius/explosion map.
    Requirement: Agent is currently not in bomb radius/explosion map
    """
    feature = np.zeros(len(ACTIONS))

    if len(bomb_fields) == 0:
        # TODO: try out return equals 1
        return feature

    if (x, y) not in bomb_fields:
        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, x, y)

            if field[new_x, new_y] == -1 or field[new_x, new_y] == 1:  # moving into wall or crate
                continue

            if action == "BOMB":
                continue

            if (new_x, new_y) not in bomb_fields:
                feature[idx] = 1

    return feature


# TODO: negative reward for setting bombs that cannot destroy any create or kill an oponnent

def feat_10(field: np.array, x: int, y: int) -> np.array:
    """
    Agent places bomb next to opponent if he can escape
    """
    feature = np.zeros(len(ACTIONS))
    return feature

# Reward for moving towards the nearest opponent: under certain conditions
# Use transitions of the other agents: think about weights
# Reward for setting bombs that can kill an opponent
# Reward bombs that are leading for a "safe" dead of an agent
# dead end feature
