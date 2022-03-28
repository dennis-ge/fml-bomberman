import events as e

# Feature 1
MOVED_TOWARDS_COIN_1 = "MOVED_TOWARDS_COIN_1"
MOVED_AWAY_FROM_COIN_1 = "MOVED_AWAY_FROM_COIN_1"
# Feature 2
DID_NOT_COLLECT_COIN_2 = "DID_NOT_COLLECT_COIN_2"
# Feature 3
VALID_ACTION_3 = "VALID_ACTION_3"
INVALID_ACTION_3 = "INVALID_ACTION_3"
# Feature 4
MOVED_OUT_OF_BLAST_RADIUS_4 = "MOVED_OUT_OF_BLAST_RADIUS_4"
STAYED_IN_BLAST_RADIUS_4 = "STAYED_IN_BLAST_RADIUS_4"
# Feature 5
MOVED_TOWARDS_BOMB_POSITIONS_5 = "MOVED_TOWARDS_bomb_positions_5"
MOVED_AWAY_FROM_BOMB_POSITIONS_5 = "MOVED_AWAY_FROM_bomb_positions_5"
# Feature 6
STAYED_OUT_OF_BOMB_POS_6 = "STAYED_OUT_OF_BOMB_RADIUS_6"
MOVED_INTO_BOMB_POS_6 = "MOVED_INTO_BOMB_RADIUS_6"
# Feature 7
PLACED_BOMB_NEXT_TO_CRATE_7 = "PLACED_BOMB_NEXT_TO_CRATE_7"
# DID_NOT_PLACED_BOMB_NEXT_TO_CRATE = "DID_NOT_PLACED_BOMB_NEXT_TO_CRATE"
# Feature 8
MOVED_TOWARDS_CRATE_8 = "MOVED_TOWARDS_CRATE_8"
MOVED_AWAY_FROM_CRATE_8 = "MOVED_AWAY_FROM_CRATE_8"
# Feature 9
PLACED_BOMB_NEXT_TO_OPPONENT_9 = "PLACED_BOMB_NEXT_TO_OPPONENT_9"
# DID_NOT_PLACED_BOMB_NEXT_TO_OPPONENT = "DID_NOT_PLACED_BOMB_NEXT_TO_OPPONENT"
PLACED_USELESS_BOMB_7_9 = "PLACED_USELESS_BOMB_7_9"
# Feature 10
MOVED_AWAY_FROM_DANG_OPPONENT_10 = "MOVED_AWAY_FROM_DANGEROUS_OPPONENT_10"
MOVED_TOWARDS_DANG_OPPONENT_10 = "MOVED_TOWARDS_DANGEROUS_OPPONENT_10"
# Feature 11
MOVED_TOWARDS_OPPONENT_11 = "MOVED_TOWARDS_OPPONENT_11"
MOVED_AWAY_FROM_OPPONENT_11 = "MOVED_AWAY_FROM_OPPONENT_11"
# Feature 12
KILLED_OPPONENT_IN_TRAP_12 = "KILLED_OPPONENT_IN_TRAP_12"
DID_NOT_KILL_OPPONENT_IN_TRAP_12 = "DID_NOT_KILL_OPPONENT_IN_TRAP_12"
# Feature 13
MOVED_INTO_DANG_POS_13 = "MOVED_INTO_DANGEROUS_POSITION_13"
DID_NOT_MOVE_INTO_DANG_POS_13 = "DID_NOT_MOVE_INTO_DANGEROUS_POSITION_13"

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
    # ratio between occurence of positive and negative rewards

    # Name: [default, possible range]
    e.OPPONENT_ELIMINATED: [0, [0]],
    e.MOVED_UP: [-1, [-1, -5]],
    e.MOVED_RIGHT: [-1, [-1, -5]],
    e.MOVED_DOWN: [-1, [-1, -5]],
    e.MOVED_LEFT: [-1, [-1, -5]],
    e.BOMB_DROPPED: [2, ranges["small_pos"]],
    e.COIN_FOUND: [0, ranges["small_pos"]],
    e.CRATE_DESTROYED: [0, ranges["small_pos"]],
    e.WAITED: [-1, [-25, -15]],
    e.SURVIVED_ROUND: [50, ranges["high_pos"]],
    e.GOT_KILLED: [0, [-50]],
    e.KILLED_OPPONENT: [0, ranges["high_pos"]],
    e.KILLED_SELF: [-500, [-100, -200]],

    MOVED_TOWARDS_COIN_1: [3, ranges["small_pos_2"]],
    MOVED_AWAY_FROM_COIN_1: [-9, ranges["small_neg_2"]],
    e.COIN_COLLECTED: [20, ranges["medium_pos"]],
    DID_NOT_COLLECT_COIN_2: [-40, ranges["medium_neg"]],
    VALID_ACTION_3: [20, [2]],
    INVALID_ACTION_3: [-20, [-15]],
    MOVED_OUT_OF_BLAST_RADIUS_4: [55, ranges["medium_pos"]],
    STAYED_IN_BLAST_RADIUS_4: [-40, ranges["medium_neg"]],
    # MOVED_TOWARDS_BOMB_POSITIONS_5: [-27, ranges["medium_neg"]],
    # MOVED_AWAY_FROM_BOMB_POSITIONS_5: [35, ranges["medium_pos"]],
    STAYED_OUT_OF_BOMB_POS_6: [55, ranges["medium_pos"]],
    MOVED_INTO_BOMB_POS_6: [-40, ranges["medium_neg"]],
    PLACED_BOMB_NEXT_TO_CRATE_7: [80, ranges["small_pos_2"]],
    MOVED_TOWARDS_CRATE_8: [5, ranges["small_pos"]],
    MOVED_AWAY_FROM_CRATE_8: [-5, ranges["small_neg"]],
    PLACED_BOMB_NEXT_TO_OPPONENT_9: [100, [30, 40]],
    PLACED_USELESS_BOMB_7_9: [-80, ranges["medium_neg"]],
    MOVED_AWAY_FROM_DANG_OPPONENT_10: [6, ranges["small_pos"]],
    MOVED_TOWARDS_DANG_OPPONENT_10: [-10, ranges["small_neg_2"]],
    MOVED_TOWARDS_OPPONENT_11: [5, ranges["small_pos_2"]],
    MOVED_AWAY_FROM_OPPONENT_11: [-5, ranges["small_pos_2"]],
    KILLED_OPPONENT_IN_TRAP_12: [200, ranges["high_pos"]],
    DID_NOT_KILL_OPPONENT_IN_TRAP_12: [-200, ranges["medium_neg"]],
    MOVED_INTO_DANG_POS_13: [-100, ranges["medium_neg"]],
    DID_NOT_MOVE_INTO_DANG_POS_13: [80, ranges["medium_neg"]],
}
