import logging
import os
import random
import shutil
import string
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
from pythonjsonlogger import jsonlogger

from main import main as play

SEED = 42
TIMESTAMP = datetime.now(timezone.utc).strftime("%dT%H:%M")


#
# Helper Methods
#
def setup_logger() -> logging.Logger:
    logger = logging.getLogger('Trainer')
    logger.setLevel("DEBUG")

    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.info('Initializing Logger')

    return logger


logger = setup_logger()


def unique_id(chars=string.ascii_uppercase + string.digits, N=10):
    return ''.join(random.choice(chars) for _ in range(N))


def create_match_name(match_id, agents: str):
    return f"{match_id}-{'-vs-'.join(agents.split(' '))}"


EnvVariables = namedtuple("EnvironmentalVariables", ["policy", "model_name", "n_rounds", "gamma", "match_id"])


def set_env(env: EnvVariables, rewards: List[Tuple[str, int]]):
    os.environ["POLICY"] = env.policy
    os.environ["MODEL_NAME"] = env.model_name
    os.environ["N_ROUNDS"] = f"{env.n_rounds}"
    os.environ["GAMMA"] = f"{env.gamma}"
    os.environ["MATCH_ID"] = env.match_id

    if len(rewards) > 0:
        for item in rewards:
            os.environ[item[0]] = f"{item[1]}"


# [ 2,2,]
# [1,3]
# [2,2,1]


def append_list_to_list(first: list, second: list) -> list:
    result = []
    for item in second:
        f_copy = first.copy()
        f_copy.append(item)
        result.append(f_copy)
    return result


def get_rewards(n_samples: int) -> List[List[Tuple[str, int]]]:
    from agent_code.task1.rewards import REWARDS_LIST

    all_rewards = []
    for i in range(n_samples):
        rewards = []
        for reward in REWARDS_LIST:
            choice = np.random.choice(reward[2])
            rewards.append((reward[0], choice))
        all_rewards.append(rewards)

    return all_rewards
    # for top_level in REWARDS_LIST:
    #
    #     curr = top_level[2]
    #     for i in range(len(REWARDS_LIST)):
    #         second_level = REWARDS_LIST[i]
    #         if top_level == second_level:
    #             continue

    # unique_combinations = []
    # permut = itertools.permutations(curr, len(second_level[2]))
    # for comb in permut:
    #     zipped = zip(comb, second_level[2])
    #     unique_combinations.append(list(zipped))
    # unique_combinations = []

    # Extract Combination Mapping in two lists
    # using zip() + product()
    # unique_combinations = list(list(zip(curr, element))
    #                            for element in itertools.product(second_level[2], repeat=len(curr)))
    #
    # curr = list(unique_combinations[0][0])

    # for top_level in REWARDS_LIST:
    #     top_level_perm = []
    #     idxs = np.zeros(len(REWARDS_LIST), dtype=int)
    #     for _ in range(0, len(top_level[2])):
    #         curr_perm = []
    #         i = 0
    #         curr = REWARDS_LIST[i]
    #         while curr != REWARDS_LIST[-1:][0]:
    #             if len(curr[2]) > idxs[i]:
    #                 curr_perm.append(curr[2][idxs[i]])
    #                 idxs[i] += 1
    #             else:
    #                 curr_perm.append(curr[2][len(curr[2]-1)])
    #
    #             i += 1
    #             curr = REWARDS_LIST[i]
    #         top_level_perm.append(curr_perm)
    #
    #     all_permutations.append(top_level_perm)


#
# Game related
#
def get_opponents(arg):
    switch = {
        "": "",
        "rule": " rule_based_agent rule_based_agent rule_based_agent",
        "random": " random_agent random_agent random_agent",
        "peaceful": " peaceful_agent peaceful_agent peaceful_agent",
    }
    return switch.get(arg)


GameIteration = namedtuple("GameIteration", ["agents", "match_name", "n_rounds", "log_dir", "save_stats",
                                             "scenario", "seed"])


def play_iteration(iteration: GameIteration, env: EnvVariables, rewards: List[Tuple[str, int]]):
    """
    Plays one iteration of the game with the given settings
    """
    game_args = ["play", "--no-gui"]
    game_args.extend(["--agents", *iteration.agents.split(" ")])
    game_args.extend(["--n-rounds", str(iteration.n_rounds)])
    game_args.extend(["--log-dir", iteration.log_dir])
    game_args.extend(["--scenario", iteration.scenario])
    game_args.extend(["--seed", str(42)]) if iteration.seed else []
    game_args.extend(["--match-name", iteration.match_name])
    game_args.extend(["--save-stats", iteration.save_stats])
    game_args.extend(["--train", str(1)])

    set_env(env, rewards)
    play(game_args)


def play_game(scenario: str, agents: str, env: EnvVariables, all_rewards: List[List[Tuple[str, int]]]):
    """
    play x iterations of the game with the given settings
    """

    def execute(e, r, **kwargs):
        logger.info("Executing iteration", extra=kwargs)
        logger.info("Game rewards for iteration", extra=dict(r))
        it = GameIteration(**kwargs)
        play_iteration(it, e, r)

    for rewards in all_rewards:
        env.match_id = f"task1-{unique_id()}"
        env.model_name = f"{env.match_id}.pt"
        mn = create_match_name(env.match_id, agents)
        execute(env, rewards, agents=agents, match_name=mn, n_rounds=env.n_rounds, scenario=scenario, save_stats=f"results/{mn}.json",
                log_dir=os.path.dirname(os.path.abspath(__file__)) + "/logs", seed=False)

        first_agent = agents.split(" ")[0]
        shutil.copy2(f'dump/models/{env.model_name}', f'agent_code/{first_agent}/models/{env.model_name}')


#
# Hyperparameter tuning
#
def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--agent", type=str, default="task1")
    parser.add_argument("--o", type=str, default="")
    parser.add_argument("--s", type=str, default="classic")

    args = parser.parse_args(argv)

    agents = args.agent + get_opponents(args.o)

    env = EnvVariables(policy="epsilon_greedy", gamma=0.8, n_rounds=args.rounds, match_id="", model_name="")

    all_rewards = get_rewards(20)

    play_game(agents=agents, scenario=args.s, env=env, all_rewards=all_rewards)


if __name__ == "__main__":
    main()
