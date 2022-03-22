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
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.enemy_transitions = dict(deque(maxlen=ENEMY_TRANSITION_HISTORY_SIZE))
    self.episode = 0
    self.rewards = np.zeros(env.NUMBER_OF_ROUNDS)
    self.models = np.zeros((env.NUMBER_OF_ROUNDS, NUMBER_OF_FEATURES))


def get_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict or None) -> List[str]:
    custom_events = []
    bomb_fields = get_bomb_fields(old_game_state["field"], old_game_state["bombs"], old_game_state["explosion_map"])
    enemies_pos = [enemy[3] for enemy in old_game_state["others"]]

    for other_agent in old_game_state["others"]:
        if old_game_state["step"] == 1 and self.episode == 0:
            self.enemy_transitions[other_agent[0]] = deque(maxlen=ENEMY_TRANSITION_HISTORY_SIZE)

        self.enemy_transitions[other_agent[0]].append(EnemyTransition(other_agent[2], *other_agent[3]))

    # Feature 1
    if len(old_game_state["coins"]) > 0:
        if moved_towards_coin(old_game_state, new_game_state):
            custom_events.append(MOVED_TOWARDS_COIN)
        else:
            custom_events.append(MOVED_AWAY_FROM_COIN)

    # Feature 2
    if len(old_game_state["coins"]) > 0 and 1 in feat_2(old_game_state["coins"], old_game_state["self"][3]):
        if old_game_state["self"][1] == new_game_state["self"][1]:
            custom_events.append(DID_NOT_COLLECT_COIN)

    # Feature 3
    #if valid_action(old_game_state, self_action, bomb_fields, enemies_pos):
    #    custom_events.append(VALID_ACTION)
    #else:
    #    custom_events.append(INVALID_ACTION)

    # Feature 3
    # if self_action == "WAIT":
    #     if wait_is_intelligent(old_game_state["field"], bomb_fields, old_game_state["self"][3]):
    #         custom_events.append(WAIT_ACTION_IS_INTELLIGENT)


    # Feature 4
    if old_game_state["self"][3] in get_blast_radius(old_game_state["field"], old_game_state["bombs"]):
        if moved_out_of_blast_radius(old_game_state, new_game_state, bomb_fields, enemies_pos):
            custom_events.append(MOVED_OUT_OF_BLAST_RADIUS)
        else:
            custom_events.append(STAYED_IN_BLAST_RADIUS)

    # Feature 5
    if len(old_game_state["bombs"]) > 0:
        if stayed_out_of_bomb_fields(old_game_state, new_game_state, bomb_fields):
            custom_events.append(MOVED_AWAY_FROM_BOMB_FIELDS)
        else:
            if self_action != "BOMB":
                custom_events.append(MOVED_TOWARDS_BOMB_FIELDS)

    # # Feature 6
    # if len(old_game_state["bombs"]) > 0:
    #     if did_not_move_into_bomb_fields(old_game_state, new_game_state, bomb_fields):
    #         custom_events.append(STAYED_OUT_OF_BOMB_RADIUS)
    #     else:
    #         custom_events.append(MOVED_INTO_BOMB_RADIUS)

    # Feature 7
    if 1 in feat_7(old_game_state["field"], bomb_fields, old_game_state["self"][2], old_game_state["self"][3], enemies_pos):
        if self_action == "BOMB":
            custom_events.append(PLACED_BOMB_NEXT_TO_CRATE)

    # Feature 8
    if 1 in old_game_state["field"]:
        if moved_towards_crate(old_game_state, new_game_state):
            custom_events.append(MOVED_TOWARDS_CRATE)

    # Feature 9
    if 1 in feat_9(old_game_state["field"], bomb_fields, old_game_state["self"][2], old_game_state["self"][3], enemies_pos):
        if self_action == "BOMB":
            custom_events.append(PLACED_BOMB_NEXT_TO_OPPONENT)

    if useless_bomb_dropped(old_game_state, self_action):
        custom_events.append(SET_USELESS_BOMB)

    # Feature 10
    enemies_nearby = get_nearby_enemies(old_game_state["self"][3], old_game_state["others"])
    if len(enemies_nearby) > 0:
        if moved_away_from_dangerous_enemies(old_game_state, new_game_state, enemies_pos):
            custom_events.append(MOVED_AWAY_FROM_DANGEROUS_ENEMY)
        else:
            custom_events.append(MOVED_AWAY_FROM_DANGEROUS_ENEMY)

    # Feature 11
    crates = [(x, y) for x, y in np.ndindex(old_game_state["field"].shape) if (old_game_state["field"][x, y] == 1) or (x, y)]
    if len(crates) == 0 and len(old_game_state["coins"]) == 0 and len(enemies_pos) > 0:
        if moved_towards_enemy(old_game_state, new_game_state, enemies_pos, crates):
            custom_events.append(MOVED_TOWARDS_ENEMY)
        else:
            custom_events.append(MOVED_AWAY_FROM_ENEMY)
    return custom_events

    # Feature 12
    dead_end_exits = get_dead_end_exits(old_game_state["field"], enemies_pos)
    if dead_end_exit_reachable(dead_end_exits, old_game_state["self"][3]):
        if kill_enemy_in_dead_end(self_action):
            custom_events.append(KILLED_ENEMY_IN_DEAD_END)
        else:
            custom_events.append(MISSED_KILL_ENEMY_IN_DEAD_END)

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
    self.model = q.td_update(self.model, transition)
    self.transitions.append(transition)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


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
    self.transitions.append(last_transition)
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
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in REWARDS:
            reward_sum += REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def moved_towards_coin(old_state, new_state) -> bool:
    """
    Feature 1: Checks whether the agent moved towards a coin.
    """
    feature_old = feat_1(old_state["field"], old_state["coins"], old_state["self"][3])

    idx = np.where(feature_old == 1)[0][0]
    expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    return (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y)


def valid_action(old_state, self_action, bomb_fields, enemies_pos) -> bool:
    """
    Feature 3: Checks wheter the agent chose of the valid/intelligent actions
    """
    feature_old = feat_3(old_state["field"], bomb_fields, old_state["self"][2], old_state["self"][3], enemies_pos)

    # indexes of action that are seen as valid/intelligent
    idxs = np.where(feature_old == 1)[0]
    for idx in idxs:
        if ACTIONS[idx] == self_action:
            return True

    return False

def moved_out_of_blast_radius(old_state, new_state, bomb_fields, enemies_pos) -> bool:
    """
    Feature 4: Checks whether the agent moved out of the blast radius
    Note: The agent is in the blast radius in the old state.
    """
    feature_old = feat_4(old_state["field"], bomb_fields, old_state["self"][3], enemies_pos)

    idx = np.where(feature_old == 1)[0][0]
    expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    return (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y)


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


def moved_towards_crate(old_state, new_state) -> bool:
    """
    Feature 8: Checks whether the agent moved towards a crate.
    """
    feature_old = feat_8(old_state["field"], old_state["self"][2], old_state["self"][3])

    # check if move to crate is possible
    if feature_old.max() == 0:
        return False

    idx = np.where(feature_old == 1)[0][0]
    expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    return (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y)


def useless_bomb_dropped(old_state, last_action):
    """
    Check if the agent has set a bomb that cannot destroy a crate or any another agent

    1. Check if the last action of the agent was "BOMBED"
    2. Calculate blast radius of current bomb and get the affected fields
    3. Check if crate was in blast radius
    4. Check if other agent was in blast radius
    5. If agent cannot escapse return also True
    """
    if last_action == "BOMB":
        field = old_state["field"]
        bomb_x, bomb_y = old_state["self"][3]

        bomb = (bomb_x, bomb_y)
        blast_radius_of_bomb = get_blast_radius(old_state["field"], [(bomb, 0)])

        if not escape_possible_alternative(field, bomb):
            return True

        # Check if at least one crate is destroyed in bomb radius
        crates = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 1)]
        for crate in crates:
            if crate in blast_radius_of_bomb:
                return False

        # Check if at least one opponent is in the bomb radius
        opponents = old_state["others"]
        for opponent in opponents:
            opponent_x, opponent_y = opponent[3]
            if (opponent_x, opponent_y) in blast_radius_of_bomb:
                return False

        return True
    else:
        return False


def moved_away_from_dangerous_enemies(old_state, new_state, enemies_nearby) -> bool:
    """
    Feature 10:
    """
    feature_old = feat_10(old_state["field"], old_state["self"][3], old_state["self"][2], enemies_nearby)

    actual_new_x, actual_new_y = new_state["self"][3]

    idxs = np.where(feature_old == 1)[0]
    for idx in idxs:
        expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])
        if (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y):
            return True

    return False


def moved_towards_enemy(old_state, new_state, enemies, crates) -> bool:
    """
    Feature 11:
    """
    feature_old = feat_11(old_state["field"], old_state["self"][3], enemies, crates, old_state["coins"])

    idx = np.where(feature_old == 1)[0][0]
    expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    return (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y)

def kill_enemy_in_dead_end(self_action) -> bool:
    """
    Feature 12:
    """
    if self_action == "BOMB":
        return True
    else:
        return False
