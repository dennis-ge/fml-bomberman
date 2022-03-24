import events as e

# Feature 1
MOVED_TOWARDS_COIN_1 = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN_1 = "MOVED_AWAY_FROM_COIN"
# Feature 2
DID_NOT_COLLECT_COIN_2 = "DID_NOT_COLLECT_COIN"
# Feature 3
VALID_ACTION_3 = "VALID_ACTION"
INVALID_ACTION_3 = "INVALID_ACTION"
# Feature 4
MOVED_OUT_OF_BLAST_RADIUS_4 = "MOVED_OUT_OF_BLAST_RADIUS"
STAYED_IN_BLAST_RADIUS_4 = "STAYED_IN_BLAST_RADIUS"
# Feature 5
MOVED_TOWARDS_BOMB_FIELDS_5 = "MOVED_TOWARDS_BOMB_FIELDS"
MOVED_AWAY_FROM_BOMB_FIELDS_5 = "MOVED_AWAY_FROM_BOMB_FIELDS"
# Feature 6
STAYED_OUT_OF_BOMB_RADIUS_6 = "STAYED_OUT_OF_BOMB_RADIUS"
MOVED_INTO_BOMB_RADIUS_6 = "MOVED_INTO_BOMB_RADIUS"
# Feature 7
PLACED_BOMB_NEXT_TO_CRATE_7 = "PLACED_BOMB_NEXT_TO_CRATE"
# DID_NOT_PLACED_BOMB_NEXT_TO_CRATE = "DID_NOT_PLACED_BOMB_NEXT_TO_CRATE"
# Feature 8
MOVED_TOWARDS_CRATE_8 = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE_8 = "MOVED_AWAY_FROM_CRATE"
# Feature 9
PLACED_BOMB_NEXT_TO_OPPONENT_9 = "PLACED_BOMB_NEXT_TO_OPPONENT"
# DID_NOT_PLACED_BOMB_NEXT_TO_OPPONENT = "DID_NOT_PLACED_BOMB_NEXT_TO_OPPONENT"
PLACED_USELESS_BOMB_7_9 = "SET_USELESS_BOMB"
# Feature 10
MOVED_AWAY_FROM_DANGEROUS_ENEMY_10 = "MOVED_AWAY_FROM_DANGEROUS_ENEMY"
MOVED_TOWARDS_DANGEROUS_ENEMY_10 = "MOVED_TOWARDS_DANGEROUS_ENEMY"
# Feature 11
MOVED_TOWARDS_ENEMY_11 = "MOVED_TOWARDS_ENEMY"
MOVED_AWAY_FROM_ENEMY_11 = "MOVED_AWAY_FROM_ENEMY"
# Feature 12
KILLED_ENEMY_IN_SAFE_DEAD_12 = "KILLED_ENEMY_IN_SAFE_DEAD"
DID_NOT_KILL_ENEMY_IN_SAFE_DEAD_12 = "DID_NOT_KILL_ENEMY_IN_SAFE_DEAD"

ranges = {
    "small_pos": [2, 6, 10],
    "small_pos_2": [8, 12, 16],
    "medium_pos": [20, 30, 40],
    "high_pos": [50, 60],
    "small_neg": [-2, -6, -10],
    "small_neg_2": [-8, -12, -16],
    "medium_neg": [-20, -30, 40],
    "high_neg": [-50, -60],
}

REWARDS = {
    # Name: [default, possible range]
    e.OPPONENT_ELIMINATED: [0, [0]],
    e.MOVED_UP: [-5, [-1, -5]],
    e.MOVED_RIGHT: [-5, [-1, -5]],
    e.MOVED_DOWN: [-5, [-1, -5]],
    e.MOVED_LEFT: [-5, [-1, -5]],
    e.BOMB_DROPPED: [10, ranges["small_pos"]],
    e.BOMB_EXPLODED: [2, [-1, 2]],
    e.COIN_FOUND: [5, ranges["small_pos"]],
    e.CRATE_DESTROYED: [5, ranges["small_pos"]],
    e.COIN_COLLECTED: [30, ranges["medium_pos"]],
    e.WAITED: [-25, [-25, -15]],
    e.SURVIVED_ROUND: [50, ranges["high_pos"]],
    e.GOT_KILLED: [-50, [-50]],
    e.INVALID_ACTION: [-50, [-50]],
    e.KILLED_OPPONENT: [70, ranges["high_pos"]],
    e.KILLED_SELF: [-100, [-100, -200]],

    MOVED_TOWARDS_COIN_1: [12, ranges["small_pos_2"]],
    MOVED_AWAY_FROM_COIN_1: [-15, ranges["small_neg_2"]],
    DID_NOT_COLLECT_COIN_2: [-30, ranges["medium_neg"]],
    #  [VALID_ACTION_3: [ 2, [2]],
    #  [INVALID_ACTION_3: [ -15, [-15]],
    MOVED_OUT_OF_BLAST_RADIUS_4: [40, ranges["medium_pos"]],
    STAYED_IN_BLAST_RADIUS_4: [-45, ranges["medium_neg"]],
    MOVED_TOWARDS_BOMB_FIELDS_5: [-45, ranges["medium_neg"]],
    MOVED_AWAY_FROM_BOMB_FIELDS_5: [40, ranges["medium_pos"]],
    STAYED_OUT_OF_BOMB_RADIUS_6: [40, ranges["medium_pos"]],
    MOVED_INTO_BOMB_RADIUS_6: [-45, ranges["medium_neg"]],
    PLACED_BOMB_NEXT_TO_CRATE_7: [20, ranges["small_pos_2"]],
    MOVED_TOWARDS_CRATE_8: [5, ranges["small_pos"]],
    MOVED_AWAY_FROM_CRATE_8: [-5, ranges["small_neg"]],
    PLACED_BOMB_NEXT_TO_OPPONENT_9: [40, [30, 40]],
    PLACED_USELESS_BOMB_7_9: [-50, ranges["medium_neg"]],
    MOVED_AWAY_FROM_DANGEROUS_ENEMY_10: [5, ranges["small_pos"]],
    MOVED_TOWARDS_DANGEROUS_ENEMY_10: [-5, ranges["small_neg_2"]],
    # MOVED_TOWARDS_ENEMY_11: [10, ranges["small_pos_2"]],
    #  [MOVED_AWAY_FROM_ENEMY_11: [ -10, ranges["small_pos_2"]],
    KILLED_ENEMY_IN_SAFE_DEAD_12: [70, ranges["high_pos"]],
    DID_NOT_KILL_ENEMY_IN_SAFE_DEAD_12: [-30, ranges["medium_neg"]],
}
