import pickle
from collections import deque
from csv import writer

import agent_code.task1.rl as q
from agent_code.task1.features import *
from agent_code.task1.rl import Transition


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.rewards = 0


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

    if old_game_state is not None:
        custom_events = []

        # Feature 1
        if len(old_game_state["coins"]) > 0:
            if moved_towards_coin(old_game_state, new_game_state):
                custom_events.append(MOVED_TOWARDS_COIN)
            else:
                custom_events.append(MOVED_AWAY_FROM_COIN)

        # Feature 2
        # if len(old_game_state["coins"]) > 0 and 1 in feat_2(old_game_state["coins"], *old_game_state["self"][3]):
        #     if old_game_state["self"][1] == new_game_state["self"][1]:
        #         custom_events.append(DID_NOT_COLLECT_COIN)

        bomb_fields = get_bomb_fields(old_game_state["field"], old_game_state["bombs"], old_game_state["explosion_map"])

        # if old_game_state["self"][3] == new_game_state["self"][3]:  # Action was wait
        #     # second if statement ensures that we don't say wait is intelligent when there is no escape way of a bomb
        #     if wait_is_intelligent(old_game_state["field"], bomb_fields, *old_game_state["self"][3]):
        #         custom_events.append(WAIT_ACTION_IS_INTELLIGENT)

        # Feature 4
        if old_game_state["self"][3] in get_blast_radius(old_game_state["field"], old_game_state["bombs"]):
            if moved_out_of_blast_radius(old_game_state, new_game_state):
                custom_events.append(MOVED_OUT_OF_BLAST_RADIUS)
            else:
                custom_events.append(STAYED_IN_BLAST_RADIUS)

        # Feature 5
        if len(old_game_state["bombs"]) > 0:
            if stayed_out_of_bomb_fields(old_game_state, new_game_state):
                custom_events.append(MOVED_AWAY_FROM_BOMB_FIELDS)
            else:
                custom_events.append(MOVED_TOWARDS_BOMB_FIELDS)

        # Feature 6 Placed bomb next to crate
        if 1 in feat_6(old_game_state["field"], bomb_fields, old_game_state["self"][2], *old_game_state["self"][3]):
            if self_action == "BOMB":
                custom_events.append(DID_NOT_PLACED_BOMB_NEXT_TO_CRATE)
            else:
                custom_events.append(PLACED_BOMB_NEXT_TO_CRATE)

        # Feature 7
        # if 1 in old_game_state["field"]:
        #     if moved_towards_crate(old_game_state, new_game_state):
        #         custom_events.append(MOVED_TOWARDS_CRATE)

        events.extend(custom_events)
        self.logger.debug(f'Custom event occurred: {custom_events}')

        current_rewards = reward_from_events(self, events)
        self.rewards += current_rewards
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
    self.episode += 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step (Episode {self.episode})')
    current_rewards = reward_from_events(self, events)
    self.rewards += current_rewards
    self.transitions.append(Transition(last_game_state, last_action, None, current_rewards))
    # Store the model
    with open(MODEL_NAME, "wb") as file:
        pickle.dump(self.model, file)
    with open(REWARDS_NAME, 'a') as file:
        w = writer(file)
        w.writerow([self.rewards])


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


def moved_towards_coin(old_state, new_state):
    """
    Feature 1: Checks whether the agent moved towards a coin.
    """
    feature_old = feat_1(old_state["field"], old_state["coins"], *old_state["self"][3])

    idx = np.where(feature_old == 1)[0][0]
    expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], *old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    return (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y)


def moved_out_of_blast_radius(old_state, new_state):
    """
    Feature 4: Checks whether the agent moved out of the blast radius
    Note: The agent is in the blast radius in the old state.
    """
    bomb_fields = get_bomb_fields(old_state["field"], old_state["bombs"], old_state["explosion_map"])
    feature_old = feat_4(old_state["field"], bomb_fields, *old_state["self"][3])

    idx = np.where(feature_old == 1)[0][0]
    expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], *old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    return (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y)


def stayed_out_of_bomb_fields(old_state, new_state):
    """
    Feature 5: Checks whether the agent moved towards an explosion or bomb that is about to explode.
    """
    bomb_fields = get_bomb_fields(old_state["field"], old_state["bombs"], old_state["explosion_map"])
    feature_old = feat_5(old_state["field"], bomb_fields, *old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]
    idxs = np.where(feature_old == 1)[0]  # get action indexes where agent moves into bomb fields
    for idx in idxs:
        expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], *old_state["self"][3])
        if (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y):
            return True

    return False


def moved_towards_crate(old_state, new_state):
    """
    Feature 7: Checks whether the agent moved towards a crate.
    """

    feature_old = feat_7(old_state["field"], old_state["self"][2], *old_state["self"][3])

    # check if move to crate is possible
    if feature_old.max() == 0:
        return False

    idx = np.where(feature_old == 1)[0][0]
    expected_new_x, expected_new_y = get_new_position(ACTIONS[idx], *old_state["self"][3])

    actual_new_x, actual_new_y = new_state["self"][3]

    return (expected_new_x, expected_new_y) == (actual_new_x, actual_new_y)
