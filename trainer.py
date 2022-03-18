import logging
import os
import random
import shutil
import string
from collections import namedtuple
from datetime import datetime, timezone

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


def match_id(chars=string.ascii_uppercase + string.digits, N=10):
    return ''.join(random.choice(chars) for _ in range(N))


def decay():
    os.environ["POLICY"] = "decay_greedy"


def create_match_name(agents: str):
    return f"{match_id()}-{'-vs-'.join(agents.split(' '))}"


EnvVariables = namedtuple("EnvironmentalVariables", ["policy", "model_name"])


def set_env(env: EnvVariables):
    os.environ["POLICY"] = env.policy
    os.environ["MODEL_NAME"] = env.model_name


#
# Game related
#


GameIteration = namedtuple("GameIteration", ["agents", "match_name", "n_rounds", "log_dir", "save_stats",
                                             "scenario", "seed"])


def play_iteration(iteration: GameIteration):
    game_args = ["play", "--no-gui"]
    game_args.extend(["--agents", *iteration.agents.split(" ")])
    game_args.extend(["--n-rounds", str(iteration.n_rounds)])
    game_args.extend(["--log-dir", iteration.log_dir])
    game_args.extend(["--scenario", iteration.scenario])
    game_args.extend(["--seed", str(42)]) if iteration.seed else []
    game_args.extend(["--match-name", iteration.match_name])
    game_args.extend(["--save-stats", iteration.save_stats])
    game_args.extend(["--train", str(len(iteration.agents.split(" ")))])

    play(game_args)


def play_game():
    scenario = "classic"
    # all_agents = ["task1 task1_double_q", "task1", "task1_double_q"]
    all_agents = ["task1"]
    iteration_count = len(all_agents)
    model_name="task1-trained.pt"
    env = EnvVariables(policy="epsilon_greedy", model_name=model_name)

    def execute(**kwargs):
        logger.info("Executing iteration", extra=kwargs)
        it = GameIteration(**kwargs)
        play_iteration(it)

    for i in range(iteration_count):
        mn = create_match_name(all_agents[i])
        set_env(env)
        execute(agents=all_agents[i], match_name=mn,
                n_rounds=1000, scenario=scenario, save_stats=f"results/{TIMESTAMP}-{mn}.json",
                log_dir=os.path.dirname(os.path.abspath(__file__)) + "/logs", seed=False)
        shutil.copy2(f'dump/{model_name}', f'agent_code/{all_agents[i]}/models/{model_name}')

#
# Hyperparameter tuning
#
def main():
    play_game()


if __name__ == "__main__":
    main()
