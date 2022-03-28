import pickle
from timeit import default_timer as timer

import numpy as np

from agent_code.fml_double.agent_settings import *
from agent_code.fml_double.features import state_to_features
from agent_code.fml_double.game_info import beautify_output
from agent_code.fml_double.rl import create_policy, max_q


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
    if self.train or not os.path.isfile(env.MODEL_NAME):
        self.logger.info(f"Setting up model from scratch.")
        weights = np.random.rand(NUMBER_OF_FEATURES)
        guess = [1, 9, 20, 30, 32, 40, 35, 7, 40, 10, 3, 42, -5]
        self.weights1 = guess
        self.weights2 = guess
        # self.weights1 = weights / weights.sum()
        # self.weights2 = weights / weights.sum()
    else:
        self.logger.info(f"Loading model from saved state: {env.MODEL_NAME}")
        with open(env.MODEL_NAME, "rb") as file:
            weights = pickle.load(file)
            self.weights1 = weights[:NUMBER_OF_FEATURES]
            self.weights2 = weights[NUMBER_OF_FEATURES:]


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

    # get best action based on q_values
    features, printable_field = state_to_features(game_state)
    _, best_actions, q_values = max_q(features, self.weights1, self.weights2)

    self.logger.debug(beautify_output(printable_field, features, self.weights1, self.weights2, q_values))

    if env.POLICY_NAME == DECAY_GREEDY_POLICY_NAME:
        action, self.prev_eps = self.policy(ACTIONS[np.random.choice(best_actions)], self.episode, self.prev_eps)
        return action

    action = self.policy(ACTIONS[np.random.choice(best_actions)])
    end = timer()
    self.logger.debug(f"Elapsed time for act: {round(end - start, 5)}s")
    return action
