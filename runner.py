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


def create_match_name(match_id, agents: str):
    return f"{match_id}-{'-vs-'.join(agents.split(' '))}"


EnvVariables = namedtuple("EnvironmentalVariables", ["policy", "model_name", "n_rounds", "gamma", "match_id"])


def set_env(env: EnvVariables):
    os.environ["POLICY"] = env.policy
    os.environ["MODEL_NAME"] = env.model_name
    os.environ["N_ROUNDS"] = f"{env.n_rounds}"
    os.environ["GAMMA"] = f"{env.gamma}"
    os.environ["MATCH_ID"] = env.match_id


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


def play_game(scenario: str, agents: str, envs: List[EnvVariables]):
    """
    play x iterations of the game with the given settings
    """

    def execute(e, **kwargs):
        logger.info("Executing iteration", extra=kwargs)
        it = GameIteration(**kwargs)
        play_iteration(it, e)

    for env in envs:
        mn = create_match_name(env.match_id, agents)
        execute(env, agents=agents, match_name=mn, n_rounds=env.n_rounds, scenario=scenario, save_stats=f"results/{TIMESTAMP}-{mn}.json",
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

    args = parser.parse_args(argv)

    agents = args.agent + get_opponents(args.o)

    envs = [
        EnvVariables(policy="epsilon_greedy", model_name="task1-eps.pt", gamma=0.8, n_rounds=args.rounds, match_id=f"task1-{unique_id()}"),
        # EnvVariables(policy="decay_greedy", model_name="task1-decay-trained.pt", gamma=0.8, n_rounds=args.rounds, match_id=f"task1-{unique_id()}"),
    ]

    play_game(agents=agents, scenario="classic", envs=envs)


if __name__ == "__main__":
    main()
