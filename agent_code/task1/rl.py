import logging

from agent_code.task1.features import *


class Transition:
    def __init__(self, state: dict, action: str, next_state: dict or None, reward: int):
        self.state: dict = state
        self.action: str = action
        self.next_state: dict = next_state
        self.reward: float = reward
        self.state_features: np.array = state_to_features(self.state)
        self.next_state_features: np.array = state_to_features(self.next_state)


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


def max_q(features: np.array, model: np.array) -> Tuple[float, List[int]]:
    q_values = np.dot(features, model)
    q_max = np.max(q_values)
    a_max = np.where(q_values == q_max)[0]  # best actions

    return q_max, a_max


def td_update(model: np.array, t: Transition) -> np.array:
    """
    Update the model based on the TD error.
    :return:
    """
    updated_weights = np.zeros(len(model))
    q_max, _ = max_q(t.next_state_features, model)

    state_action = t.state_features[ACTIONS.index(t.action), :]
    for _ in range(len(model)):
        td_error = t.reward + DISCOUNT_FACTOR * q_max - np.dot(state_action, model)
        updated_weights = updated_weights + LEARNING_RATE * td_error * state_action

    updated_model = model + updated_weights
    return updated_model

