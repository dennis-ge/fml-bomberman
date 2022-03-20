import logging
import os
import random
import shutil
import string
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime, timezone
from typing import List

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


def create_match_name(agents: str):
    return f"{unique_id()}-{'-vs-'.join(agents.split(' '))}"


EnvVariables = namedtuple("EnvironmentalVariables", ["policy", "model_name", "n_rounds", "gamma"])


def set_env(env: EnvVariables):
    os.environ["POLICY"] = env.policy
    os.environ["MODEL_NAME"] = env.model_name
    os.environ["N_ROUNDS"] = f"{env.n_rounds}"
    os.environ["GAMMA"] = f"{env.gamma}"


#
# Game related
#


GameIteration = namedtuple("GameIteration", ["agents", "match_name", "n_rounds", "log_dir", "save_stats",
                                             "scenario", "seed"])


def play_iteration(iteration: GameIteration, env: EnvVariables):
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

    set_env(env)
    play(game_args)


def play_game(env: EnvVariables, scenario: str, n_rounds: int, all_agents: List[str]):
    """
    play x iterations of the game with the given settings
    """

    def execute(**kwargs):
        # envs = *env
        logger.info("Executing iteration", extra=kwargs)
        it = GameIteration(**kwargs)
        play_iteration(it, env)

    for agent in all_agents:
        mn = create_match_name(agent)
        execute(agents=agent, match_name=mn, n_rounds=n_rounds, scenario=scenario, save_stats=f"results/{TIMESTAMP}-{mn}.json",
                log_dir=os.path.dirname(os.path.abspath(__file__)) + "/logs", seed=False)
        shutil.copy2(f'dump/{env.model_name}', f'agent_code/{agent.split(" ")[0]}/models/{env.model_name}')


#
# Hyperparameter tuning
#
def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--o", type=str, default="")

    args = parser.parse_args(argv)

    agents = "task1"
    if args.o == "rule":
        agents += " rule_based_agent rule_based_agent rule_based_agent"

    if args.o == "random":
        agents += " random_agent random_agent random_agent"

    if args.o == "peaceful":
        agents += " peaceful_agent peaceful_agent peaceful_agent"



    n_rounds = args.rounds
    envs = [
        EnvVariables(policy="epsilon_greedy", model_name="task1-trained.pt", gamma=0.8, n_rounds=n_rounds),
        #EnvVariables(policy="decay_greedy", model_name="task1-decay-trained.pt", gamma=0.8, n_rounds=n_rounds),
    ]

    # env = EnvVariables(policy="epsilon_greedy", model_name=model_name, gamma="0.8", n_rounds=f"{n_rounds}")
    # all_agents = ["task1 task1_double_q", "task1", "task1_double_q"]
    for env in envs:
        play_game(env=env, scenario="classic", n_rounds=n_rounds, all_agents=[agents])


if __name__ == "__main__":
    main()
