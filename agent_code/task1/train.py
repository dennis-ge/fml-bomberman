import pickle
from collections import deque

import agent_code.task1.rl as q
from agent_code.task1.features import *
from agent_code.task1.rl import Transition, EnemyTransition


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = []
    self.enemy_transitions = dict(deque(maxlen=ENEMY_TRANSITION_HISTORY_SIZE))
    self.episode = 0
    self.rewards = np.zeros(env.NUMBER_OF_ROUNDS)
    self.models = np.zeros((env.NUMBER_OF_ROUNDS, NUMBER_OF_FEATURES))


def get_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict or None) -> List[str]:
    custom_events = []

    bomb_fields = get_bomb_fields(old_game_state["field"], old_game_state["bombs"], old_game_state["explosion_map"])
    explosions = [(x, y) for x, y in np.ndindex(old_game_state["explosion_map"].shape) if (old_game_state["explosion_map"][x, y] != 0)]
    blast_radius = get_blast_radius(old_game_state["field"], old_game_state["bombs"])
    enemies_pos = [enemy[3] for enemy in old_game_state["others"]]
    enemies_nearby = get_nearby_enemies(old_game_state["field"], old_game_state["self"][3], old_game_state["others"])
    crates = [(x, y) for x, y in np.ndindex(old_game_state["field"].shape) if (old_game_state["field"][x, y] == 1)]

    for other_agent in old_game_state["others"]:
        if old_game_state["step"] == 1 and self.episode == 0:
            self.enemy_transitions[other_agent[0]] = deque(maxlen=ENEMY_TRANSITION_HISTORY_SIZE)

        self.enemy_transitions[other_agent[0]].append(EnemyTransition(other_agent[2], *other_agent[3]))

    # Feature 1
    if 1 in feat_1(old_game_state["field"], old_game_state["coins"], old_game_state["self"][3]):  # when it is possible to move towards a coin
        if moved_towards_coin(old_game_state, self_action):
            custom_events.append(MOVED_TOWARDS_COIN_1)
        else:
            custom_events.append(MOVED_AWAY_FROM_COIN_1)

    # Feature 2
    if 1 in feat_2(old_game_state["coins"], old_game_state["self"][3]):
        if old_game_state["self"][1] == new_game_state["self"][1]:  # score stayed the same
            custom_events.append(DID_NOT_COLLECT_COIN_2)

    # Feature 3
    if performed_valid_action(old_game_state, self_action, bomb_fields, enemies_pos):
        custom_events.append(VALID_ACTION_3)
    else:
        custom_events.append(INVALID_ACTION_3)

    # Feature 4
    if old_game_state["self"][3] in get_blast_radius(old_game_state["field"], old_game_state["bombs"]):
        if moved_out_of_blast_radius(old_game_state, bomb_fields, enemies_pos, self_action):
            custom_events.append(MOVED_OUT_OF_BLAST_RADIUS_4)
        else:
            custom_events.append(STAYED_IN_BLAST_RADIUS_4)

    # Feature 5
    if len(old_game_state["bombs"]) > 0:
        if stayed_out_of_bomb_fields(old_game_state, new_game_state, bomb_fields):
            custom_events.append(MOVED_AWAY_FROM_BOMB_FIELDS_5)
        else:
            if self_action != "BOMB":
                custom_events.append(MOVED_TOWARDS_BOMB_FIELDS_5)

    # Feature 6
    if len(old_game_state["bombs"]) > 0 and old_game_state["self"][3] not in get_blast_radius(old_game_state["field"], old_game_state["bombs"]):
        if did_not_move_into_bomb_fields(old_game_state, new_game_state, bomb_fields):
            custom_events.append(STAYED_OUT_OF_BOMB_RADIUS_6)
        else:
            custom_events.append(MOVED_INTO_BOMB_RADIUS_6)

    # Feature 7
    if 1 in feat_7(old_game_state["field"], bomb_fields, old_game_state["self"][2], old_game_state["self"][3], enemies_pos):
        if self_action == "BOMB":
            custom_events.append(PLACED_BOMB_NEXT_TO_CRATE_7)

    # Feature 8
    if 1 in old_game_state["field"]:
        if moved_towards_crate(old_game_state, new_game_state, crates):
            custom_events.append(MOVED_TOWARDS_CRATE_8)

    # Feature 9
    if 1 in feat_9(old_game_state["field"], bomb_fields, old_game_state["self"][2], old_game_state["self"][3], enemies_pos):
        if self_action == "BOMB":
            custom_events.append(PLACED_BOMB_NEXT_TO_OPPONENT_9)

    if placed_useless_bomb(old_game_state, self_action, bomb_fields, enemies_pos):
        custom_events.append(PLACED_USELESS_BOMB_7_9)

    # Feature 10
    enemies_nearby = get_nearby_enemies(old_game_state["field"], old_game_state["self"][3], old_game_state["others"])
    if len(enemies_nearby) > 0:
        if moved_away_from_dangerous_enemies(old_game_state, new_game_state, enemies_pos, bomb_fields):
            custom_events.append(MOVED_AWAY_FROM_DANGEROUS_ENEMY_10)
        else:
            custom_events.append(MOVED_AWAY_FROM_DANGEROUS_ENEMY_10)
    
    # # Feature 11
    feat_11_old = feat_11(old_game_state["field"], old_game_state["self"][3], old_game_state["self"][2], enemies_pos)
    if 1 in feat_11_old:
        if moved_towards_enemy(feat_11_old, self_action):
            custom_events.append(MOVED_TOWARDS_ENEMY_11)
        else:
            custom_events.append(MOVED_AWAY_FROM_ENEMY_11)

    # Feature 12
    trapped_tiles, possible_trapped_tiles = get_trapped_tiles(old_game_state["field"], enemies_pos, explosions, blast_radius)
    if trapped_tiles_pos_reachable(old_game_state["self"][3], trapped_tiles, possible_trapped_tiles):
        if self_action == "BOMB":
            custom_events.append(KILLED_ENEMY_IN_TRAP_12)
        else:
            custom_events.append(DID_NOT_KILL_ENEMY_IN_TRAP_12)

    feat_13_old = feat_13(old_game_state["field"], old_game_state["self"][3], explosions, blast_radius, enemies_nearby)
    if 1 in feat_13_old:
        if moved_into_dead_end(feat_13_old, self_action):
            custom_events.append(MOVED_INTO_DANGEROUS_POSITION_13)
        else:
            custom_events.append(DID_NOT_MOVE_INTO_DANGEROUS_POSITION_13)

    return custom_events


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    if old_game_state is None:
        return None

    events.extend(get_custom_events(self, old_game_state, self_action, new_game_state))
    current_rewards = reward_from_events(self, events)
    self.rewards[self.episode] += current_rewards
    transition = Transition(old_game_state, self_action, new_game_state, current_rewards)
    self.transitions.append(transition)

    if EXPERIENCE_REPLAY_ACTIVATED:
        if len(self.transitions) == EXPERIENCE_REPLAY_K:
            current_transitions = self.transitions.copy()
            np.random.shuffle(current_transitions)

            for _ in range(0, int(EXPERIENCE_REPLAY_K / EXPERIENCE_REPLAY_BATCH_SIZE)):
                current_batch = current_transitions[:EXPERIENCE_REPLAY_BATCH_SIZE]
                current_transitions = current_transitions[EXPERIENCE_REPLAY_BATCH_SIZE:]

                for batch_transition in current_batch:
                    self.model = q.td_update(self.model, batch_transition)
    else:
        self.model = q.td_update(self.model, transition)

    # self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state:
    :param last_action:
    :param events:
    """

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step (Episode {self.episode})')
    current_rewards = reward_from_events(self, events)
    self.rewards[self.episode] += current_rewards
    self.models[self.episode] = self.model

    last_transition = Transition(last_game_state, last_action, None, current_rewards)
    # self.transitions.append(last_transition)
    fake_q_values = np.zeros(self.model.shape)
    self.logger.debug(beautify_output(last_transition.printable_field, last_transition.state_features, self.model, fake_q_values))

    with open(env.MODEL_NAME, "wb") as file:
        pickle.dump(self.model, file)

    self.episode += 1
    # TODO: print value of the rewards and results of the game in order to see if we have created a good agent
    # use average of the last 10? games and play whole games without breaks after dead from us

    if self.episode == env.NUMBER_OF_ROUNDS:
        with open(env.REWARDS_NAME, 'wb') as file:
            pickle.dump(self.rewards, file)

        with open(env.WEIGHTS_NAME, 'wb') as file:
            pickle.dump(self.models, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in env.REWARDS:
            reward_sum += env.REWARDS[event]
        else:
            self.logger.info(f"ERROR: REWARD not in LIST {event}")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def moved_towards_coin(old_state, self_action: str) -> bool:
    """
    Feature 1: Checks whether the agent moved towards a coin.
    """
    feature_old = feat_1(old_state["field"], old_state["coins"], old_state["self"][3])

    idx = np.where(feature_old == 1)[0][0]

    return ACTIONS[idx] == self_action


def performed_valid_action(old_state, self_action, bomb_fields, enemies_pos) -> bool:
    """
    Feature 3: Checks whether the agent chose one of the valid/intelligent actions
    """
    feature_old = feat_3(old_state["field"], bomb_fields, old_state["self"][2], old_state["self"][3], enemies_pos)

    # indexes of action that are seen as valid/intelligent
    idxs = np.where(feature_old == 1)[0]
    for idx in idxs:
        if ACTIONS[idx] == self_action:
            return True

    return False


def moved_out_of_blast_radius(old_state, bomb_fields, enemies_pos, self_action: str, ) -> bool:
    """
    Feature 4: Checks whether the agent moved out of the blast radius
    Note: The agent is in the blast radius in the old state.
    """
    feature_old = feat_4(old_state["field"], bomb_fields, old_state["self"][3], enemies_pos)

    idxs = np.where(feature_old == 1)[0]
    for idx in idxs:
        if ACTIONS[idx] == self_action:
            return True

    return False


def stayed_out_of_bomb_fields(old_state, new_state, bomb_fields) -> bool:
    """
    Feature 5: Checks whether the agent moved towards an explosion or bomb that is about to explode.
    """
    feature_old = feat_5(old_state["field"], bomb_fields, old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]
    idxs = np.where(feature_old == 1)[0]
    # get action indexes where agent moves into bomb fields
    for idx in idxs:
        expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])
        if (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y):
            return True

    return False


def did_not_move_into_bomb_fields(old_state, new_state, bomb_fields) -> bool:
    """
    Feature 6: Checks if the agent did not move into a bomb field when he was in a safe place before
    """

    feature_old = feat_6(old_state["field"], bomb_fields, old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    idxs = np.where(feature_old == 1)[0]  # get action indexes where agent moves stays out of bomb fields
    for idx in idxs:
        expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])
        if (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y):
            return True

    return False


def moved_towards_crate(old_state, self_action: str, crates: List[Tuple[int, int]]) -> bool:
    """
    Feature 8: Checks whether the agent moved towards a crate.
    """
    feature_old = feat_8(old_state["field"], old_state["self"][2], old_state["self"][3], crates)

    # check if move to crate is possible
    if feature_old.max() == 0:
        return False

    idx = np.where(feature_old == 1)[0][0]

    return ACTIONS[idx] == self_action


def placed_useless_bomb(old_state, last_action, bomb_fields, enemies_pos):
    """
    Check if the agent has set a bomb that cannot destroy a crate or any another agent

    1. Check if the last action of the agent was "BOMBED"
    2. Calculate blast radius of current bomb and get the affected fields
    3. Check if crate was in blast radius
    4. Check if other agent was in blast radius
    5. If agent cannot escapse return also True
    """
    if last_action != "BOMB":
        return False

    field = old_state["field"]
    agent_pos = old_state["self"][3]

    blast_radius = get_blast_radius(old_state["field"], [(agent_pos, 0)])

    if not escape_possible(field, bomb_fields, agent_pos, enemies_pos):
        return True

    # Check if at least one crate is destroyed in bomb radius
    crates = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 1)]
    for crate in crates:
        if crate in blast_radius:
            return False

    for enemy_pos in enemies_pos:
        if enemy_pos in blast_radius:
            return False

    return True


def moved_away_from_dangerous_enemies(old_state, new_state, enemies_nearby, bomb_fields) -> bool:
    """
    Feature 10:
    """
    feature_old = feat_10(old_state["field"], old_state["self"][3], old_state["self"][2], enemies_nearby, bomb_fields)

    actual_new_x, actual_new_y = new_state["self"][3]

    idxs = np.where(feature_old == 1)[0]
    for idx in idxs:
        expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])
        if (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y):
            return True

    return False


def moved_towards_enemy(feature_old, self_action) -> bool:
    """
    Feature 11:
    """
    idx = np.where(feature_old == 1)[0][0]

    return ACTIONS[idx] == self_action


def moved_into_dead_end(feature_old, self_action) -> bool:
    """
    Feature 13:
    """
    idxs = np.where(feature_old == 1)[0]
    for idx in idxs:
        if ACTIONS[idx] == self_action:
            return True

    return False
