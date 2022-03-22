import pickle
from timeit import default_timer as timer

import numpy as np

from agent_code.task1.agent_settings import *
from agent_code.task1.features import state_to_features
from agent_code.task1.game_info import beautify_output
from agent_code.task1.rl import create_policy, max_q


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.episode = 0
    self.prev_eps = env.EPSILON_START
    env.reload()

    self.policy = create_policy(env.POLICY_NAME, self.logger)
    if self.train or not os.path.isfile(PROD_MODEL_NAME):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(NUMBER_OF_FEATURES)
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open(PROD_MODEL_NAME, "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    start = timer()

    self.logger.debug(f"--- Choosing an action for step {game_state['step']} at position {game_state['self'][3]}")

    if self.train and np.random.random() < env.EPSILON:
        rand_action = np.random.choice(ACTIONS, p=[.167, .167, .167, .167, .166, .166])
        self.logger.debug(f"Chosen the following action purely at random: {rand_action}")
        return rand_action

    # get best action based on q_values
    features, printable_field = state_to_features(game_state)
    _, best_actions, q_values = max_q(features, self.model)

    self.logger.debug(beautify_output(printable_field, features, self.model, q_values))

    if env.POLICY_NAME == DECAY_GREEDY_POLICY_NAME:
        action, self.prev_eps = self.policy(ACTIONS[np.random.choice(best_actions)], self.episode, self.prev_eps)
        return action

    action = self.policy(ACTIONS[np.random.choice(best_actions)])
    end = timer()
    self.logger.debug(f"Elapsed time for act: {round(end - start, 5)}s")
    return action
