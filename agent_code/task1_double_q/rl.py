import logging

from agent_code.task1_double_q.features import *


class Transition:
    def __init__(self, state: dict, action: str, next_state: dict or None, reward: int):
        self.state: dict = state
        self.action: str = action
        self.next_state: dict = next_state
        self.reward: float = reward
        self.state_features: np.array = state_to_features(self.state)
        self.next_state_features: np.array = state_to_features(self.next_state)


def create_policy(name: str, logger: logging.Logger):
    """
    Creates a policy based on the parameters.
    :param name: Name of the policy.
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

    def decay_greedy_policy(action: str, curr_episode: int, prev_eps: float):
        eps = EPSILON_START
        if curr_episode > 0:
            new_eps = prev_eps * EPSILON_DECAY
            eps = new_eps if new_eps > EPSILON_END else EPSILON_END
        rand_action = np.random.choice(ACTIONS)
        chosen_action = np.random.choice([action, rand_action], p=[1 - eps, eps])
        logger.debug(f"Decay epsilon greedy policy: Given action is '{action}', Chosen action is '{chosen_action}' with eps={eps}")
        return chosen_action, eps

    if name == GREEDY_POLICY_NAME:
        return greedy_policy
    elif name == EPSILON_GREEDY_POLICY_NAME:
        return epsilon_greedy_policy
    elif name == DECAY_GREEDY_POLICY_NAME:
        return decay_greedy_policy

    raise ValueError(f'Unknown policy {name}')


def max_q(features: np.array, weights1: np.array, weights2: np.array) -> Tuple[float, List[int]]:
    q_values = np.dot(features, weights1) + np.dot(features, weights2)
    q_max = np.max(q_values)
    a_max = np.where(q_values == q_max)[0]  # best actions

    return q_max, a_max


def max_q_single(features: np.array, weights: np.array) -> Tuple[float, List[int]]:
    q_values = np.dot(features, weights)
    q_max = np.max(q_values)
    a_max = np.where(q_values == q_max)[0]  # best actions

    return q_max, a_max


def td_update(weights1: np.array, weights2: np.array, t: Transition) -> Tuple[np.array, np.array]:
    """
    Update the model based on the TD error.
    :return:
    """
    batch_updated_weights1, batch_updated_weights2 = np.zeros(len(weights1)), np.zeros(len(weights2))

    state_action = t.state_features[ACTIONS.index(t.action), :]
    for _ in range(len(weights1)):
        if np.random.rand() < 0.5:
            _, best_actions = max_q_single(t.next_state_features, weights1)
            selected_features = t.next_state_features[np.random.choice(best_actions), :]
            max_q_next, _ = max_q_single(selected_features, weights2)
            td_error = t.reward + DISCOUNT_FACTOR * max_q_next - np.dot(state_action, weights1)
            batch_updated_weights1 = batch_updated_weights1 + LEARNING_RATE * td_error * state_action
        else:
            _, best_actions = max_q_single(t.next_state_features, weights2)
            selected_features = t.next_state_features[np.random.choice(best_actions), :]
            max_q_next, _ = max_q_single(selected_features, weights1)
            td_error = t.reward + DISCOUNT_FACTOR * max_q_next - np.dot(state_action, weights2)
            batch_updated_weights2 = batch_updated_weights2 + LEARNING_RATE * td_error * state_action

    updated_weights1 = weights1 + batch_updated_weights1
    updated_weights2 = weights2 + batch_updated_weights2
    return updated_weights1, updated_weights2
