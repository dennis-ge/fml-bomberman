from agent_code.task1.features import *


class Transition:
    def __init__(self, state: dict, action: str, next_state: dict or None, reward: int):
        self.state: dict = state
        self.action: str = action
        self.next_state: dict = next_state
        self.reward: float = reward
        self.state_features, self.printable_field = state_to_features(self.state)
        self.next_state_features, _ = state_to_features(self.next_state)


class EnemyTransition:
    def __init__(self, bomb_action_possible: bool, x: int, y: int):
        self.bomb_action_possible = bomb_action_possible
        self.x = x
        self.y = y


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
        chosen_action = np.random.choice([action, rand_action], p=[1 - env.EPSILON, env.EPSILON])
        logger.debug(f"Epsilon greedy policy: Given action is '{action}', Chosen action is '{chosen_action}'")
        return chosen_action

    def decay_greedy_policy(action: str, curr_episode: int, prev_eps: float):
        eps = env.EPSILON_START
        if curr_episode > 0:
            new_eps = prev_eps * env.EPSILON_DECAY
            eps = new_eps if new_eps > env.EPSILON_END else env.EPSILON_END
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


def max_q(features: np.array, model: np.array) -> Tuple[float, List[int], List[float]]:
    q_values = np.dot(features, model)
    q_max = np.max(q_values)
    a_max = np.where(q_values == q_max)[0]  # best actions

    return q_max, a_max, q_values


def td_update(model: np.array, t: Transition) -> np.array:
    """
    Update the model based on the TD error.
    :return:
    """
    updated_weights = np.zeros(len(model))
    q_max, _, _ = max_q(t.next_state_features, model)

    state_action = t.state_features[ACTIONS.index(t.action), :]
    for _ in range(len(model)):
        td_error = t.reward + env.DISCOUNT_FACTOR * q_max - np.dot(state_action, model)
        updated_weights = updated_weights + env.LEARNING_RATE * td_error * state_action

    updated_model = model + updated_weights
    return updated_model
