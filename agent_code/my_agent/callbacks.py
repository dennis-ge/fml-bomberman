import os
import pickle

import numpy as np

from agent_code.my_agent.agent_settings import *
from agent_code.my_agent.features import state_to_features
from agent_code.my_agent.rl import create_policy, max_q

policy_name = os.environ.get("POLICY", EPSILON_GREEDY_POLICY_NAME)


def print_features_and_model(self, features: np.array, model: np.array):
    self.logger.debug(f"Features: {[list(item) for item in features]}, Model: {model}")


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
    self.policy = create_policy(policy_name, self.logger)
    if self.train or not os.path.isfile(MODEL_NAME):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(NUMBER_OF_FEATURES)
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_NAME, "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # if self.train and random.random() < EPSILON:
    #     rand_action = np.random.choice(ACTIONS, p=[.22, .22, .22, .22, .12])
    #     self.logger.debug(f"Chosen the following action purely at random: {rand_action}")
    #     return rand_action

    self.logger.debug(f"--- Choosing an action for step {game_state['step']}")

    # get best action based on q_values
    features = state_to_features(game_state)
    _, best_actions = max_q(features, self.model)
    print_features_and_model(self, features, self.model)
    return self.policy(ACTIONS[np.random.choice(best_actions)])
