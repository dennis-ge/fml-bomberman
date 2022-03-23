import events as e

# Feature 1
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
# Feature 2
DID_NOT_COLLECT_COIN = "DID_NOT_COLLECT_COIN"
# Feature 3
VALID_ACTION = "VALID_ACTION"
INVALID_ACTION = "INVALID_ACTION"
# Feature 4
MOVED_OUT_OF_BLAST_RADIUS = "MOVED_OUT_OF_BLAST_RADIUS"
STAYED_IN_BLAST_RADIUS = "STAYED_IN_BLAST_RADIUS"
# Feature 5
MOVED_TOWARDS_BOMB_FIELDS = "MOVED_TOWARDS_BOMB_FIELDS"
MOVED_AWAY_FROM_BOMB_FIELDS = "MOVED_AWAY_FROM_BOMB_FIELDS"
# Feature 6
PLACED_BOMB_NEXT_TO_CRATE = "PLACED_BOMB_NEXT_TO_CRATE"
# DID_NOT_PLACED_BOMB_NEXT_TO_CRATE = "DID_NOT_PLACED_BOMB_NEXT_TO_CRATE"
# Feature 7
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"
# Feature 8
STAYED_OUT_OF_BOMB_RADIUS = "STAYED_OUT_OF_BOMB_RADIUS"
MOVED_INTO_BOMB_RADIUS = "MOVED_INTO_BOMB_RADIUS"
# Feature 9
PLACED_BOMB_NEXT_TO_OPPONENT = "PLACED_BOMB_NEXT_TO_OPPONENT"
# DID_NOT_PLACED_BOMB_NEXT_TO_OPPONENT = "DID_NOT_PLACED_BOMB_NEXT_TO_OPPONENT"
PLACED_USELESS_BOMB = "SET_USELESS_BOMB"
# Feature 10
MOVED_AWAY_FROM_DANGEROUS_ENEMY = "MOVED_AWAY_FROM_DANGEROUS_ENEMY"
MOVED_TOWARDS_DANGEROUS_ENEMY = "MOVED_TOWARDS_DANGEROUS_ENEMY"
# Feature 11
MOVED_TOWARDS_ENEMY = "MOVED_TOWARDS_ENEMY"
MOVED_AWAY_FROM_ENEMY = "MOVED_AWAY_FROM_ENEMY"
# Feature 12
KILLED_ENEMY_IN_SAFE_DEAD = "KILLED_ENEMY_IN_SAFE_DEAD"
DID_NOT_KILL_ENEMY_IN_SAFE_DEAD = "DID_NOT_KILL_ENEMY_IN_SAFE_DEAD"

ranges = {
    "small_pos": [2, 6, 10],
    "small_pos_2": [8, 12, 16],
    "medium_pos": [20, 30],
    "high_pos": [50, 60],
    "small_neg": [-2, -6, -10],
    "small_neg_2": [-8, -12, -16],
    "medium_neg": [-20, -30],
    "high_neg": [-50, -60],
}

REWARDS_LIST = [
    # [Name, default, possible range]
    [e.OPPONENT_ELIMINATED, 0, [2]],
    [VALID_ACTION, 0, [2]],
    [MOVED_TOWARDS_CRATE, 2, ranges["small_pos"]],
    [e.CRATE_DESTROYED, 5, ranges["small_pos"]],
    [MOVED_AWAY_FROM_DANGEROUS_ENEMY, 5, ranges["small_pos"]],
    [PLACED_BOMB_NEXT_TO_CRATE, 5, ranges["small_pos"]],
    [e.BOMB_DROPPED, 10, ranges["small_pos"]],
    [e.COIN_FOUND, 10, ranges["small_pos"]],
    [MOVED_TOWARDS_ENEMY, 10, ranges["small_pos_2"]],
    [MOVED_TOWARDS_COIN, 10, ranges["small_pos_2"]],
    # [PLACED_BOMB_NEXT_TO_OPPONENT,10],
    [MOVED_AWAY_FROM_BOMB_FIELDS, 30, ranges["medium_pos"]],
    [STAYED_OUT_OF_BOMB_RADIUS, 30, ranges["medium_pos"]],
    [e.COIN_COLLECTED, 35, ranges["medium_pos"]],
    [MOVED_OUT_OF_BLAST_RADIUS, 40, ranges["medium_pos"]],
    [e.SURVIVED_ROUND, 50, ranges["high_pos"]],
    [KILLED_ENEMY_IN_SAFE_DEAD, 50, ranges["high_pos"]],
    [e.KILLED_OPPONENT, 50, ranges["high_pos"]],
    # Negative
    [e.MOVED_UP, -1, [-1]],
    [e.MOVED_RIGHT, -1, [-1]],
    [e.MOVED_DOWN, -1, [-1]],
    [e.MOVED_LEFT, -1, [-1]],
    [e.BOMB_EXPLODED, -1, [-1]],
    [MOVED_AWAY_FROM_CRATE, -2, ranges["small_neg"]],
    [MOVED_TOWARDS_DANGEROUS_ENEMY, -5, ranges["small_neg_2"]],
    [INVALID_ACTION, -15, [-15]],
    [PLACED_USELESS_BOMB, -20, ranges["medium_neg"]],
    [e.WAITED, -20, [-20]],
    [MOVED_AWAY_FROM_COIN, -15, ranges["small_neg_2"]],
    [DID_NOT_COLLECT_COIN, -25, ranges["medium_neg"]],
    [DID_NOT_KILL_ENEMY_IN_SAFE_DEAD, -30, ranges["medium_neg"]],
    [e.GOT_KILLED, -50, [-50]],
    [e.INVALID_ACTION, -50, [-50]],
    [MOVED_TOWARDS_BOMB_FIELDS, -55, ranges["medium_neg"]],
    [MOVED_INTO_BOMB_RADIUS, -60, ranges["medium_neg"]],
    [STAYED_IN_BLAST_RADIUS, -80, ranges["medium_neg"]],
    [e.KILLED_SELF, -200, [-100, -200]],
]
