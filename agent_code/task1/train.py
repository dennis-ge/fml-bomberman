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
    self.batch_size = 50
    self.batch_increment_size = 20

def get_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict or None) -> List[str]:
    custom_events = []

    field = old_game_state["field"]
    free_space = field == 0

    agent_pos: Tuple[int, int] = old_game_state["self"][3]
    bomb_action_possible = old_game_state["self"][2]

    coins = old_game_state["coins"]
    crates = [(x, y) for x, y in np.ndindex(field.shape) if (field[x, y] == 1)]

    bombs = old_game_state["bombs"]
    explosion_map = old_game_state["explosion_map"]
    explosions = [(x, y) for x, y in np.ndindex(explosion_map.shape) if (explosion_map[x, y] != 0)]
    blast_radius = get_blast_radius(field, bombs)
    bomb_fields = explosions + blast_radius

    enemies_pos = [enemy[3] for enemy in old_game_state["others"]]
    enemies_nearby = get_nearby_enemies(field.shape[0], field.shape[1], agent_pos, enemies_pos)

    safe_fields = get_safe_fields(field, bomb_fields, enemies_pos)

    escape_possible = is_escape_possible(field, bomb_fields, agent_pos, enemies_pos)

    for other_agent in old_game_state["others"]:
        if old_game_state["step"] == 1 and self.episode == 0:
            self.enemy_transitions[other_agent[0]] = deque(maxlen=ENEMY_TRANSITION_HISTORY_SIZE)

        self.enemy_transitions[other_agent[0]].append(EnemyTransition(other_agent[2], *other_agent[3]))

    # Feature 1
    feat_1_old = feat_1(free_space=free_space, coins=coins, agent_pos=agent_pos)
    if 1 in feat_1_old:
        if check_for_single_one(feat_1_old, self_action):
            custom_events.append(MOVED_TOWARDS_COIN_1)
        else:
            custom_events.append(MOVED_AWAY_FROM_COIN_1)

    # Feature 2
    if 1 in feat_2(coins=coins, agent_pos=agent_pos):
        if old_game_state["self"][1] == new_game_state["self"][1]:  # score stayed the same
            custom_events.append(DID_NOT_COLLECT_COIN_2)

    # Feature 3
    feat_3_old = feat_3(field=field, bomb_fields=bomb_fields, bomb_action_possible=bomb_action_possible,
                        agent_pos=agent_pos, enemies_pos=enemies_pos, escape_possible=escape_possible, safe_fields=safe_fields)
    if 1 in feat_3_old:
        if check_for_multiple_ones(feat_3_old, self_action):
            custom_events.append(VALID_ACTION_3)
        else:
            custom_events.append(INVALID_ACTION_3)

    # Feature 4
    feat_4_old = feat_4(field=field, free_space=free_space, bomb_fields=bomb_fields, agent_pos=agent_pos, safe_fields=safe_fields)
    if 1 in feat_4_old:
        if check_for_multiple_ones(feat_4_old, self_action):
            custom_events.append(MOVED_OUT_OF_BLAST_RADIUS_4)
        else:
            custom_events.append(STAYED_IN_BLAST_RADIUS_4)

    # Feature 5
    feat_5_old = feat_5(field_max_x=field.shape[0], field_max_y=field.shape[1], free_space=free_space,
                        bomb_fields=bomb_fields, agent_pos=agent_pos)
    if 1 in feat_5_old:
        if check_for_multiple_ones(feat_5_old, self_action):
            custom_events.append(MOVED_AWAY_FROM_BOMB_FIELDS_5)
        elif self_action != "BOMB":
            custom_events.append(MOVED_TOWARDS_BOMB_FIELDS_5)

    # Feature 6
    feat_6_old = feat_6(field=field, bomb_fields=bomb_fields, agent_pos=agent_pos)
    if 1 in feat_6_old:
        if check_for_multiple_ones(feat_6_old, self_action):
            custom_events.append(STAYED_OUT_OF_BOMB_RADIUS_6)
        else:
            custom_events.append(MOVED_INTO_BOMB_RADIUS_6)

    # Feature 7
    feat_7_old = feat_7(field=field, bomb_action_possible=bomb_action_possible, agent_pos=agent_pos, escape_possible=escape_possible)
    if 1 in feat_7_old:
        if self_action == "BOMB":
            custom_events.append(PLACED_BOMB_NEXT_TO_CRATE_7)

    # Feature 8
    feat_8_old = feat_8(free_space=free_space, bomb_action_possible=bomb_action_possible, agent_pos=agent_pos, crates=crates)
    if 1 in feat_8_old:
        if check_for_single_one(feat_8_old, self_action):
            custom_events.append(MOVED_TOWARDS_CRATE_8)
        else:
            custom_events.append(MOVED_AWAY_FROM_CRATE_8)

    # Feature 9
    feat_9_old = feat_9(bomb_action_possible=bomb_action_possible, pos=agent_pos, enemies_pos=enemies_pos, escape_possible=escape_possible)
    if 1 in feat_9_old:
        if self_action == "BOMB":
            custom_events.append(PLACED_BOMB_NEXT_TO_OPPONENT_9)

    if placed_useless_bomb(field=field, agent_pos=agent_pos, self_action=self_action, bomb_fields=bomb_fields,
                           enemies_pos=enemies_pos, crates=crates):
        custom_events.append(PLACED_USELESS_BOMB_7_9)

    # Feature 10
    feat_10_old = feat_10(free_space=free_space, agent_pos=agent_pos, bomb_action_possible=bomb_action_possible, enemies_nearby=enemies_nearby,
                          bomb_fields=bomb_fields)
    if 1 in feat_10_old:
        if check_for_multiple_ones(feat_10_old, self_action):
            custom_events.append(MOVED_AWAY_FROM_DANGEROUS_ENEMY_10)
        else:
            custom_events.append(MOVED_TOWARDS_DANGEROUS_ENEMY_10)

    # # Feature 11
    feat_11_old = feat_11(free_space=free_space, agent_pos=agent_pos, bomb_action_possible=bomb_action_possible, enemies_pos=enemies_pos)
    if 1 in feat_11_old:
        if check_for_single_one(feat_11_old, self_action):
            custom_events.append(MOVED_TOWARDS_ENEMY_11)
        else:
            custom_events.append(MOVED_AWAY_FROM_ENEMY_11)

    # Feature 12
    feat_12_old = feat_12(field, agent_pos, bomb_action_possible, explosions=explosions, blast_radius=blast_radius, enemies_pos=enemies_pos)
    if 1 in feat_12_old:
        if check_for_multiple_ones(feat_12_old, self_action):
            custom_events.append(KILLED_ENEMY_IN_TRAP_12)
        else:
            custom_events.append(DID_NOT_KILL_ENEMY_IN_TRAP_12)

    feat_13_old = feat_13(field=field, free_space=free_space, agent_pos=agent_pos, explosions=explosions, blast_radius=blast_radius,
                          enemies_nearby=enemies_nearby)
    if 1 in feat_13_old:
        if check_for_multiple_ones(feat_13_old, self_action):
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

    if not env.EXPERIENCE_REPLAY_ACTIVATED:
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
    fake_q_values = np.zeros(len(ACTIONS))

    self.logger.debug(beautify_output(last_transition.printable_field, last_transition.state_features, self.model,fake_q_values))
    self.episode += 1

    rewards = [t.reward for t in self.transitions]
    sum_rewards = np.sum(np.abs(rewards))
    reward_p = [np.abs(r) / sum_rewards for r in rewards]

    if env.EXPERIENCE_REPLAY_ACTIVATED:
        if (self.episode % 5) == 0:  # every 5 rounds
            sample_size = self.batch_size if self.batch_size <= len(self.transitions) else len(self.transitions)
            # current_batch = np.random.choice(self.transitions, sample_size, p=reward_p)
            # current_batch = np.random.choice(self.transitions, sample_size)
            current_batch = self.transitions.copy()
            self.logger.info(f"Transition Count: {len(self.transitions)}, Batch Count {sample_size}")
            self.batch_size += self.batch_increment_size
            np.random.shuffle(current_batch)
            self.transitions = []

            model_batch = np.zeros(len(self.model))
            for batch_transition in current_batch:
                model_batch = q.td_update(model_batch, batch_transition, sample_size)
            self.model = self.model + model_batch

    with open(env.MODEL_NAME, "wb") as file:
        pickle.dump(self.model, file)

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


def check_for_single_one(feature_vector: np.ndarray, self_action: str) -> bool:
    """
    helper function checking for a given feature vector whether the chosen action is desired
    """
    idx = np.where(feature_vector == 1)[0][0]

    return ACTIONS[idx] == self_action


def check_for_multiple_ones(feature_vector: np.ndarray, self_action: str) -> bool:
    """
    helper function checking for a given feature vector whether the chosen action is desired
    """
    idxs = np.where(feature_vector == 1)[0]
    for idx in idxs:
        if ACTIONS[idx] == self_action:
            return True

    return False


def placed_useless_bomb(field: np.ndarray, agent_pos: Tuple[int, int], self_action: str, bomb_fields: List[Tuple[int, int]],
                        enemies_pos: List[Tuple[int, int]], crates: List[Tuple[int, int]]) -> bool:
    """
    Check if the agent has set a bomb that cannot destroy a crate or any another agent

    1. Check if the last action of the agent was "BOMBED"
    2. Calculate blast radius of current bomb and get the affected fields
    3. Check if crate was in blast radius
    4. Check if other agent was in blast radius
    5. If agent cannot escapse return also True
    """
    if self_action != "BOMB":
        return False

    blast_radius = get_blast_radius(field, [(agent_pos, 0)])

    if not is_escape_possible(field, bomb_fields, agent_pos, enemies_pos):
        return True

    # Check if at least one crate is destroyed in bomb radius
    for crate in crates:
        if crate in blast_radius:
            return False

    for enemy_pos in enemies_pos:
        if enemy_pos in blast_radius:
            return False

    return True
