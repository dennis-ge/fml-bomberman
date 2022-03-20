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


def decay():
    os.environ["POLICY"] = "decay_greedy"


def create_match_name(agents: str):
    return f"{unique_id()}-{'-vs-'.join(agents.split(' '))}"


EnvVariables = namedtuple("EnvironmentalVariables", ["policy", "model_name", "n_rounds", "gamma"])


def set_env(env: EnvVariables):
    os.environ["POLICY"] = env.policy
    os.environ["MODEL_NAME"] = env.model_name
    os.environ["N_ROUNDS"] = env.n_rounds
    os.environ["GAMMA"] = env.gamma


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
    game_args.extend(["--train", str(len(iteration.agents.split(" ")))])
    set_env(env)
    play(game_args)


def play_game(env: EnvVariables, scenario: str, n_rounds: int, all_agents: List[str], model_name):
    """
    play x iterations of the game with the given settings
    """
    def execute(**kwargs):
        logger.info("Executing iteration", extra=kwargs)
        it = GameIteration(**kwargs)
        play_iteration(it, env)

    for agent in all_agents:
        mn = create_match_name(agent)
        execute(agents=agent, match_name=mn,
                n_rounds=n_rounds, scenario=scenario, save_stats=f"results/{TIMESTAMP}-{mn}.json",
                log_dir=os.path.dirname(os.path.abspath(__file__)) + "/logs", seed=False)
        shutil.copy2(f'dump/{model_name}', f'agent_code/{agent}/models/{model_name}')


#
# Hyperparameter tuning
#
def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")

    model_name = "task1-trained.pt"
    n_rounds=1000
    env = EnvVariables(policy="epsilon_greedy", model_name=model_name, gamma="0.8", n_rounds=f"{n_rounds}")
    # all_agents = ["task1 task1_double_q", "task1", "task1_double_q"]
    play_game(env=env, scenario="classic", n_rounds=n_rounds, all_agents=["task1"], model_name=model_name)


if __name__ == "__main__":
    main()
