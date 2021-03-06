import json
import logging
import os
import pickle
import random
import shutil
import string
from argparse import ArgumentParser
from datetime import datetime, timezone
from typing import List, Tuple, Union

import numpy as np
from pythonjsonlogger import jsonlogger

from main import main as play

SEED = 42
TIMESTAMP = datetime.now(timezone.utc).strftime("%dT%H:%M")


#
# Helper Methods
#
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")

    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.info('Initializing Logger')

    return logger


def append_list_to_list(first: list, second: list) -> list:
    result = []
    for item in second:
        f_copy = first.copy()
        f_copy.append(item)
        result.append(f_copy)
    return result


def get_rewards(agent_name: str, n_samples: int) -> List[List[Tuple[str, int]]]:
    if agent_name == "fml":
        from agent_code.fml.rewards import REWARDS
    else:
        from agent_code.fml_double.rewards import REWARDS
    all_rewards = []
    for i in range(n_samples):
        rewards = []
        for name, item in REWARDS.items():
            choice = np.random.choice(item[1])
            rewards.append((name, int(choice)))
        all_rewards.append(rewards)

    return all_rewards


def get_biased_rewards(agent_name: str, n_samples: int) -> List[List[Tuple[str, int]]]:
    if agent_name == "fml":
        from agent_code.fml.rewards import REWARDS
    else:
        from agent_code.fml_double.rewards import REWARDS

    all_rewards = []
    for i in range(n_samples):
        rewards = []
        for name, item in REWARDS.items():
            deviation = [i for i in range(item[0] - 10, item[0] + 10)]
            choice = np.random.choice(deviation)
            rewards.append((name, int(choice)))
        all_rewards.append(rewards)

    return all_rewards


def unique_id(chars=string.ascii_uppercase + string.digits, N=4):
    return ''.join(random.choice(chars) for _ in range(N))


def create_match_name(match_id, agents: str):
    return f"{match_id}-{'-vs-'.join(agents.split(' '))}"


def get_opponents(arg):
    switch = {
        "": "",
        "rule": " rule_based_agent rule_based_agent rule_based_agent",
        "random": " random_agent random_agent random_agent",
        "peaceful": " peaceful_agent peaceful_agent peaceful_agent",
        "coin": " coin_collector_agent coin_collector_agent coin_collector_agent",
        "mix": " rule_based_agent coin_collector_agent peaceful",
    }
    return switch.get(arg)


class EnvVariables:

    def __init__(self, policy: str = None, model_name: str = None, n_rounds: int = None, gamma: float = None,
                 match_id: str = None, stats_file: str = None, eps: float = None, alpha: float = None,
                 eps_max: float = None, eps_min: float = None, eps_decay: float = None):
        self.policy = policy
        self.model_name = model_name
        self.n_rounds = n_rounds
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.alpha = alpha
        self.match_id = match_id
        self.stats_file = stats_file

    def set(self, rewards: Union[List[Tuple[str, int]], None]):
        os.environ["POLICY"] = self.policy
        os.environ["MODEL_NAME"] = self.model_name
        os.environ["N_ROUNDS"] = str(self.n_rounds)
        os.environ["GAMMA"] = str(self.gamma)
        os.environ["EPS"] = str(self.eps)
        os.environ["ALPHA"] = str(self.alpha)
        os.environ["MATCH_ID"] = self.match_id
        os.environ["EPS_START"] = str(self.eps_max)
        os.environ["EPS_MIN"] = str(self.eps_min)
        os.environ["EPS_DECAY"] = str(self.eps_decay)

        if rewards:
            for item in rewards:
                os.environ[item[0]] = f"{item[1]}"


def play_iteration(agents: str, n_rounds: int, scenario: str, match_name: str, stats_file: str, train: bool = True, seed: bool = False, multithread: bool = True):
    game_args = ["play", "--no-gui"]
    game_args.extend(["--agents", *agents.split(" ")])
    game_args.extend(["--n-rounds", str(n_rounds)])
    game_args.extend(["--scenario", scenario])
    game_args.extend(["--save-stats", stats_file])
    if train:
        game_args.extend(["--match-name", match_name])
        game_args.extend(["--train", str(1)])
    if seed:
        game_args.extend(["--seed", str(42)])
    if multithread:
        game_args.extend(["--single-process"])

    play(game_args)


class Runner:

    def __init__(self, scenario: str, agent: str, opponents: str, n_rounds_train: int, n_rounds: int, runner_id: str = None, seed: bool = False):
        self.logger = get_logger(__name__)
        self.scenario = scenario
        self.primary_agent = agent
        self.agents = agent + get_opponents(opponents)
        self.n_rounds = n_rounds
        self.n_rounds_train = n_rounds_train
        self.seed = seed
        self.runner_id = runner_id if runner_id else unique_id()
        self.logger.info(f"Starting runner {self.runner_id} for {self.primary_agent}")

    def start(self, envs: List[EnvVariables], all_rewards: Union[List[List[Tuple[str, int]]], None]):
        for env in envs:
            self.runner_id = unique_id()
            all_matches = {
                "general": {
                    "eps": env.eps,
                    "gamma": env.gamma,
                    "alpha": env.alpha,
                }
            }

            for idx, rewards in enumerate(all_rewards):
                env.match_id = f"{self.runner_id}-{self.primary_agent}-{unique_id()}"
                env.model_name = f"{env.match_id}.pt"
                self._run(idx, env, rewards)

                all_matches[env.match_id] = dict(rewards)

            if all_rewards:
                with open(f"results/runner-{self.runner_id}_rounds-{env.n_rounds}_rewards-{len(all_rewards)}.json", "w") as file:
                    json.dump(all_matches, file)

    def start_simple(self, env: EnvVariables):
        if self.primary_agent == "fml":
            from agent_code.fml.rewards import REWARDS
        else:
            from agent_code.fml_double.rewards import REWARDS

        env.match_id = f"{self.primary_agent}-eps"
        env.model_name = f"{env.match_id}.pt"
        env.n_rounds = self.n_rounds
        self._run(0, env, None)

        with open(f"agent_code/{self.primary_agent}/models/{env.model_name}", "rb") as file:
            model = pickle.load(file)
            if self.primary_agent == "fml_double":
                from agent_code.fml_double.agent_settings import NUMBER_OF_FEATURES
                weights1 = model[:NUMBER_OF_FEATURES]
                weights2 = model[NUMBER_OF_FEATURES:]
                print(f"Weights 1: {[f'{round(weight, 2)} ({idx})' for idx, weight in enumerate(weights1)]}")
                print(f"Weights 2: {[f'{round(weight, 2)} ({idx})' for idx, weight in enumerate(weights2)]}")
            else:
                print(f"Model {[f'{round(weight, 2)} ({idx})' for idx, weight in enumerate(model)]}")

        with open(f'agent_code/{self.primary_agent}/logs/{self.primary_agent}_train.log', 'r') as file:
            data = file.read().replace('\n', '')

        reward_occ = {}
        for idx, reward in enumerate(REWARDS.keys()):
            reward_occ[idx] = [reward, int(data.count(reward) / 2)]

        for idx in range(0, len(reward_occ.keys()), 2):
            output = f"{reward_occ[idx][0]:40} - {reward_occ[idx][1]:5} |"
            if idx + 1 < len(reward_occ.keys()):
                output += f"{reward_occ[idx + 1][1]:5} - {reward_occ[idx + 1][0]:40}"
            print(output)

    def _run(self, idx: int, env: EnvVariables, rewards: Union[List[Tuple[str, int]], None]):
        self.logger.info(f"Rewards for iteration {idx}", extra={"rewards": rewards})
        env.n_rounds = self.n_rounds_train
        self.logger.info(f"Executing training iteration {idx}", extra={"match_name": env.match_id, "agents": self.agents,
                                                                       "n_rounds": env.n_rounds, "scenario": self.scenario,
                                                                       "alpha": env.alpha, "gamma": env.gamma, "eps": env.eps,
                                                                       "policy": env.policy})
        stats_file = f"results/{env.match_id}_train.json"
        env.set(rewards)
        play_iteration(agents=self.agents, scenario=self.scenario, n_rounds=env.n_rounds, match_name=env.match_id, stats_file=stats_file, seed=self.seed)

        with open(stats_file, "rb") as f:
            stats = json.load(f)
            self.logger.info(f"Statistics for training iteration {idx} ({env.match_id})", extra=stats["by_agent"][self.primary_agent])

        if rewards:
            stats_file = f"results/{env.match_id}.json"
            prev_policy = env.policy
            env.policy = "greedy"
            env.n_rounds = self.n_rounds
            env.set(rewards)
            self.logger.info(f"Executing greedy iteration {idx}", extra={"match_name": env.match_id, "agents": self.agents,
                                                                         "n_rounds": env.n_rounds, "scenario": self.scenario,
                                                                         "alpha": env.alpha, "gamma": env.gamma, "eps": env.eps,
                                                                         "policy": env.policy})
            play_iteration(agents=self.agents, scenario=self.scenario, n_rounds=env.n_rounds, match_name=env.match_id, stats_file=stats_file, train=False)
            env.policy = prev_policy
            with open(stats_file, "rb") as f:
                stats = json.load(f)
                self.logger.info(f"Statistics for greedy iteration {idx} ({env.match_id})", extra=stats["by_agent"][self.primary_agent])

        if not rewards and os.path.isfile(f'dump/models/{env.model_name}'):
            shutil.copy2(f'dump/models/{env.model_name}', f'agent_code/{self.primary_agent}/models/{env.model_name}')


def get_env_samples(n: int):
    envs = []
    for _ in range(n):
        alpha = np.random.choice([0.01, 0.05])
        gamma = np.random.choice([0.7, 0.8, 0.9, 0.95, 0.99])
        eps = np.random.choice([0.05, 0.10, 0.15, 0.2])
        eps_max = np.random.choice([0.9, 1])
        eps_min = np.random.choice([0.05])
        eps_decay = np.random.choice([0.95, 0.99])
        envs.append(EnvVariables(alpha=alpha, gamma=gamma, eps=eps, eps_min=eps_min, eps_max=eps_max, eps_decay=eps_decay))

    return envs
#
# Hyperparameter tuning
#
def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--rounds-train", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--agent", type=str, default="fml")
    parser.add_argument("--o", type=str, default="rule")
    parser.add_argument("--s", type=str, default="classic")
    parser.add_argument("--simple", type=bool, default=False)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", default=False, action="store_false")
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--policy", type=str, default="epsilon_greedy")
    parser.add_argument("--env-samples", type=int, default=1)

    args = parser.parse_args(argv)

    all_rewards = None
    if not args.simple:
        # all_rewards = get_rewards(args.agent, args.samples)
        all_rewards = get_biased_rewards(args.agent, args.samples)

    sr = Runner(scenario=args.s, n_rounds_train=args.rounds_train, n_rounds=args.rounds, agent=args.agent, opponents=args.o, seed=args.seed)

    if args.simple:
        env = EnvVariables(policy=args.policy, gamma=args.gamma, eps=args.eps, alpha=args.alpha, eps_max=1, eps_decay=0.99, eps_min=0.05)
        sr.start_simple(env)
    else:
        envs = get_env_samples(args.env_samples)
        sr.start(envs, all_rewards)


if __name__ == "__main__":
    main()
