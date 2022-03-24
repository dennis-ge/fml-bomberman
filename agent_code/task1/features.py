from agent_code.task1.game_info import *


def state_to_features(game_state: dict) -> Union[Tuple[None, None], Tuple[np.ndarray, str]]:
    """
    Converts the game state to a feature vector.
    :param game_state:  A dictionary describing the current game board.
    :return: np.ndarray
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None, None

    agent_pos: Tuple[int, int] = game_state["self"][3]
    bomb_action_possible = game_state["self"][2]

    coins = game_state["coins"]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    enemies_pos = [enemy[3] for enemy in game_state["others"]]
    crates = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 1) or (x, y)]
    enemies_nearby = get_nearby_enemies(agent_pos, game_state["others"])

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
        feat_10(field, agent_pos, bomb_action_possible, enemies_nearby),
        feat_11(field, agent_pos, bomb_action_possible, enemies_pos, crates, coins),
        # feat_12(field, agent_pos, bomb_action_possible, enemies_pos)
    ))

    printable_field = ""
    if env.PRINT_FIELD:
        printable_field = field.copy()
        for pos, _ in bombs:
            printable_field[pos[0], pos[1]] = 7
        for x, y in enemies_pos:
            printable_field[x, y] = 6
        for x, y in np.ndindex(explosion_map.shape):
            if explosion_map[x, y] != 0:
                printable_field[x, y] = 9
        for x, y in coins:
            printable_field[x, y] = 8
        printable_field[agent_pos[0], agent_pos[1]] = 5
        printable_field = np.transpose(printable_field)
        printable_field = str(printable_field).replace("0", ".").replace("5", "x").replace("6", "o").replace("7", "b").replace("8", "c").replace("9", "e")

    return stacked_channels.T, printable_field


def feat_1(field: np.ndarray, coins: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.ndarray:
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


def feat_2(coins: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.ndarray:
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


def feat_3(field: np.ndarray, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, agent_pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.ndarray:
    """
     Agent performs valid action. A valid action is an action where the agent does
     not move out of the field, into walls, other enemies or crates
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

        if action == "BOMB" and not bomb_action_possible:
            continue

        if action == "BOMB" and not escape_possible(field, bomb_fields, agent_pos, enemies_pos):
            continue

        feature[idx] = 1

    return feature


def feat_4(field: np.ndarray, bomb_fields: List[Tuple[int, int]], agent_pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.ndarray:
    """
    Agent moves out of the blast radius (and does not move into other)
    TODO: there are maybe multiple correct directions
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

            new_pos = get_new_position(action, agent_pos)
            if new_pos == best_direction or new_pos not in bomb_fields:
                feature[idx] = 1

    return feature


def feat_5(field: np.ndarray, bomb_fields: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.ndarray:
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
        free_space = field == 0
        best_direction, _ = look_for_targets(free_space, agent_pos, bomb_fields_nearby)

        for idx, action in enumerate(ACTIONS):
            new_x, new_y = get_new_position(action, agent_pos)

            if (new_x, new_y) == best_direction:
                feature[idx] = 0

    # set bomb to zero, since only real moves should be evaluated
    feature[ACTIONS.index("BOMB")] = 0
    return feature


def feat_6(field: np.ndarray, bomb_fields: List[Tuple[int, int]], agent_pos: Tuple[int, int]) -> np.ndarray:
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


def feat_7(field: np.ndarray, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, agent_pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.ndarray:
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


def feat_8(field: np.ndarray, bomb_action_possible: bool, agent_pos: Tuple[int, int]) -> np.ndarray:
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
        if action == "BOMB": # or action == "WAIT":
            continue

        new_x, new_y = get_new_position(action, agent_pos)
        if (new_x, new_y) == best_direction:
            feature[idx] = 1

    return feature


def feat_9(field: np.ndarray, bomb_fields: List[Tuple[int, int]], bomb_action_possible: bool, pos: Tuple[int, int], enemies_pos: List[Tuple[int, int]]) -> np.ndarray:
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


def feat_10(field: np.ndarray, agent_pos: Tuple[int, int], bomb_action_possible: bool, enemies_nearby: List[Tuple[int, int]]) -> np.ndarray:
    """
    Agent moves away if he has no bom action available and enemies are around having a bomb action available
    - can contain multiple 1s
    """
    feature = np.zeros(len(ACTIONS))

    if len(enemies_nearby) > 0 and not bomb_action_possible:
        free_space = field == 0
        best_direction, _ = look_for_targets(free_space, agent_pos, enemies_nearby)

        for idx, action in enumerate(ACTIONS):
            new_pos = get_new_position(action, agent_pos)

            if action == "WAIT":
                continue

            if new_pos != best_direction:
                feature[idx] = 1


    return feature


def feat_11(field: np.ndarray, agent_pos: Tuple[int, int], bomb_action_available: bool, enemies: List[Tuple[int, int]], crates: List[Tuple[int, int]], coins: List[Tuple[int, int]]) -> np.ndarray:
    """
    Agent moves to enemy
    - can contain multiple 1s
    """
    feature = np.zeros(len(ACTIONS))

    if len(crates) > 5 or len(coins) > 2:  # TODO check if there are any coins left
        return feature

    # don't move to enemy when bomb action available
    # if bomb_action_available:
    #     return feature

    if len(enemies) > 0:
        free_space = field == 0
        best_direction, _ = look_for_targets(free_space, agent_pos, enemies)

        for idx, action in enumerate(ACTIONS):
            new_pos = get_new_position(action, agent_pos)

            if new_pos == best_direction:
                feature[idx] = 1

    return feature


def feat_12(field: np.ndarray, agent_pos: Tuple[int, int], bomb_action_possible: bool, enemies_pos: List[Tuple[int, int]]) -> np.ndarray:
    """
    Agent sets bomb if it leads to a safe dead of an opponent.
    """
    feature = np.zeros(len(ACTIONS))
    if bomb_action_possible:
        # check for fields that only have one direction for escaping
        safe_deads_exits = get_safe_dead_for_enemies(field, enemies_pos)

        if safe_dead_exits_reachable(safe_deads_exits, agent_pos):
            feature[ACTIONS.index("BOMB")] = 1

    return feature

# Use transitions of the other agents: think about weights
# consider other agent pos
