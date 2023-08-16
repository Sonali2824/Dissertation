# Modifed Dellacherie heuristic reward
"""
Landing Height: The height at which the current piece fell.
2. Eroded pieces: The contribution of the last piece to the cleared lines time the
number of cleared lines. --- not used
3. Row Transitions: The number of filled cells adjacent to the empty cells summed
over all rows.
4. Column Transitions: The same as Row Transitions but along columns.
5. Holes: A cell is considered to be a hole if it is empty and the cell above it is
occupied.
6. Cumulative Wells:A well is a succession of empty cells and the cells to the left
and right are occupied.
So cumulative wells is the sum of accumulated depths of the wells.
"""
def dellacherie_heurisitic_reward(landing_height, row_transitions, column_transitions, holes, wells, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += - 4 * holes - wells - row_transitions - column_transitions - landing_height
    return reward

# near_bot_heuristic_reward
"""
a) Aggregate Height This tells how high the grid and it is computed by summing the heights of all columns.
b) Complete Lines Since the goal of the AI agent is clearing lines so we
want to maximize this value.
c) Holes
d) Bumpiness The bumpiness of a grid represents the variation of its column
heights. It is computed by summing up the absolute differences between
all two adjacent columns
"""
def near_bot_heuristic_reward(agg_height, completed_lines, holes, bumpiness, validity, is_not_alive):
    reward = 0
    if validity:
        reward +=  -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += -0.510066 * (agg_height) + 0.760666 * (completed_lines) - 0.35663 * (holes) - 0.184483 * (bumpiness)
    return reward

# el_tetris_heurisitic_reward
"""
El - Tetris
There are six features in total, formally outlined as follows:

Landing Height: The height where the piece is put (= the height of the column + (the height of the piece / 2))
Rows eliminated: The number of rows eliminated.
Row Transitions: The total number of row transitions. A row transition occurs when an empty cell is adjacent to a filled cell on the same row and vice versa.
Column Transitions: The total number of column transitions. A column transition occurs when an empty cell is adjacent to a filled cell on the same column and vice versa.
Number of Holes: A hole is an empty cell that has at least one filled cell above it in the same column.
Well Sums: A well is a succession of empty cells such that their left cells and right cells are both filled.
"""
def el_tetris_heurisitic_reward(landing_height, cleared_lines, row_transitions, column_transitions, holes, wells, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += -4.500158825082766 * landing_height + 3.4181268101392694 * cleared_lines - 3.2178882868487753 * row_transitions - 9.348695305445199 * column_transitions - 7.899265427351652 * holes - 3.3855972247263626 * wells
    return reward

# Thiam, Kessler and Schwenker Evaluation
"""
Average Height
2. Holes
3. Quadratic Unevenness This feature is the sum of the squared values of the
differences of the neighbouring columns. It is a good indicator about how the
pieces is distributed.
The weighted evaluation function 
"""
def thiam_kessler_Schwenker_evaluation(average_height_t, holes_t, QU_t, average_height_t_1, holes_t_1, QU_t_1, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += (-5 * average_height_t - 16 * holes_t - QU_t) - (-5 * average_height_t_1 - 16 * holes_t_1 - QU_t_1)
    return reward

# Modified Thiam, Kessler and Schwenker Evaluation
def thiam_kessler_Schwenker_evaluation_modified(average_height_t, holes_t, QU_t, cleared_lines_t, wells_t, average_height_t_1, holes_t_1, QU_t_1, cleared_lines_t_1, wells_t_1, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += (-5 * average_height_t - 16 * holes_t - QU_t + 10 * cleared_lines_t - wells_t) - (-5 * average_height_t_1 - 16 * holes_t_1 - QU_t_1 + 10 * cleared_lines_t_1 - wells_t_1)
    return reward

# yan et al reward function adaptation
def yan_et_al_evaluation(completed_lines_t, holes_t, bumpiness_t, aggregate_height_t, completed_lines_t_1, holes_t_1, bumpiness_t_1, aggregate_height_t_1, alpha, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward +=  alpha + (completed_lines_t - completed_lines_t_1) - (aggregate_height_t - aggregate_height_t_1) - (holes_t - holes_t_1 )
    return reward

# Conventional reward function
def conventional_reward_function(cleared_lines, width, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += cleared_lines ** 2 * width + 1
    return reward

# Conventional Modified reward function
def conventional_modified_reward_function(cleared_lines, width, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += cleared_lines ** 2 * width 
    return reward

# Hanyuan et al evaluation function
def hanyuan_et_al_evaluation_function(completed_lines, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2
    if is_not_alive:
        reward += -2
    reward += 100 * (completed_lines ** 2) + 1
    return reward

def hanyuan_et_al_evaluation_function_modified(completed_lines, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -20
    if is_not_alive:
        reward += -20
    reward += 10 * (completed_lines) + 0.5
    return reward

def ulm_uni_evaluation_function(average_height, holes, QU, validity, is_not_alive):
    reward = 0
    if validity:
        reward += -2 #-10
    if is_not_alive:
        reward += -2 #-10
    reward += 5 * (average_height)+ 16 * (holes) + QU 
    return reward

# 11 reward functions
# cleared_lines, holes, bumpiness, height, row_transitions, column_transitions, cumulative_wells, QU, self.selected_tetrimino, landing_height
def calculate_reward(reward_type, previous_features, current_features, validity, is_not_alive):
    alpha = 1
    cleared_lines_t, holes_t, bumpiness_t, height_t, row_transitions_t, column_transitions_t, wells_t, QU_t, selected_tetrimino_t, landing_height_t = current_features
    cleared_lines_t_1, holes_t_1, bumpiness_t_1, height_t_1, row_transitions_t_1, column_transitions_t_1, wells_t_1, QU_t_1, selected_tetrimino_t_1, landing_height_t_1 = previous_features
    
    if reward_type == 1:
        return near_bot_heuristic_reward(height_t, cleared_lines_t, holes_t, bumpiness_t, validity, is_not_alive)
    elif reward_type == 2:
        return el_tetris_heurisitic_reward(landing_height_t, cleared_lines_t, row_transitions_t, column_transitions_t, holes_t, wells_t, validity, is_not_alive)
    elif reward_type ==3:
        return thiam_kessler_Schwenker_evaluation(height_t//10, holes_t, QU_t, height_t_1//10, holes_t_1, QU_t_1, validity, is_not_alive)
    elif reward_type == 4:
        return thiam_kessler_Schwenker_evaluation_modified(height_t//10, holes_t, QU_t, cleared_lines_t, wells_t, height_t_1//10, holes_t_1, QU_t_1, cleared_lines_t_1, wells_t_1, validity, is_not_alive)
    elif reward_type == 5:
        return yan_et_al_evaluation(cleared_lines_t, holes_t, bumpiness_t, height_t, cleared_lines_t_1, holes_t_1, bumpiness_t_1, height_t_1, alpha, validity, is_not_alive) # this wrt futures but changed to previous
    elif reward_type == 6:
        return conventional_reward_function(cleared_lines_t, 10, validity, is_not_alive)
    elif reward_type == 7:
        return hanyuan_et_al_evaluation_function(cleared_lines_t, validity, is_not_alive)
    elif reward_type == 8:
        return hanyuan_et_al_evaluation_function_modified(cleared_lines_t, validity, is_not_alive)
    elif reward_type == 9:
        return ulm_uni_evaluation_function(height_t//10, holes_t, QU_t, validity, is_not_alive)
    elif reward_type == 10:
        return dellacherie_heurisitic_reward(landing_height_t, row_transitions_t, column_transitions_t, holes_t, wells_t, validity, is_not_alive)
    elif reward_type == 11:
        return conventional_modified_reward_function(cleared_lines_t, 10, validity, is_not_alive)
    





    

