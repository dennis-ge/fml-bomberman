import logging
from typing import Any

from agent_code.my_agent.feature_extraction import *

"""
create_policy: returns function
policy: gets best action and returns selected action (sometimes the same, sometimes sth different) 

"""


def create_policy(policy_name: str, logger: logging.Logger):
    """
    Creates a policy based on the parameters.
    :param policy_name: Name of the policy.
    :param logger:
    :return: A policy function.
    """

    def greedy_policy(action: str):
        logger.debug(f"Greedy policy: Chosen action is '{action}'")
        return action

    def epsilon_greedy_policy(action: str):
        rand_action = np.random.choice(ACTIONS)
        chosen_action = np.random.choice([action, rand_action], p=[1 - EPSILON, EPSILON])
        logger.debug(f"Epsilon greedy policy: Given action is '{action}', Chosen action is '{chosen_action}'")
        return chosen_action

    if policy_name == GREEDY_POLICY_NAME:
        return greedy_policy
    elif policy_name == EPSILON_GREEDY_POLICY_NAME:
        return epsilon_greedy_policy
    elif policy_name == DECAY_GREEDY_POLICY_NAME:
        raise NotImplementedError("Decay greedy policy not implemented.")

    raise ValueError(f'Unknown policy {policy_name}')


def q_function(features: np.array, model: np.array):
    # no action in q_function since this is represented in model/weights
    return np.dot(features, model)

    # q_value = 0
    # for index, feature in enumerate(features):
    #     # iterate over all features and compute the action
    #     q_value += feature * model[index]
    # return q_value


def max_q(features: np.array, model: np.array) -> Tuple[float, List[int]]:
    q_values = q_function(features, model)
    q_max = np.max(q_values)
    a_max = np.where(q_values == q_max)[0]  # best actions

    return q_max, a_max


def td_update(model: np.array, t: Transition) -> np.array:
    """
    Update the model based on the TD error.
    :return:
    """
    # q_values = np.dot(state_to_features(game_state), self.model)
    # best_actions = np.where(q_values == np.max(q_values))[0]

    updated_model = np.zeros(len(ACTIONS))
    q_max, _ = max_q(t.next_state, model)

    # q_values = np.zeros(len(ACTIONS))
    # for idx, next_action in enumerate(ACTIONS):
    #     q_values[idx] = q_function(t.next_state, self.model)

    for idx, weight in enumerate(model):
        td_error = (t.reward + DISCOUNT_FACTOR * q_max - q_function(t.state, model))
        updated_model[idx] = weight + LEARNING_RATE * td_error

    # self.logger.info(f'Model after TD Update: {updated_model}')
    return updated_model
