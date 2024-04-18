import math
import random
import time

import numpy as np
from Utility import de_vectorize_index_to_4D, get_vectorized_index_from_4D


def crew_member_step(bot_posn: tuple[int, int], crew_posn: tuple[int, int], ship_layout: list[list[str]]) -> tuple[
    int, int]:
    crew_valid_next_positions = get_valid_crew_member_moves(crew_posn, bot_posn, ship_layout)
    next_crew_posn = random.choice(crew_valid_next_positions)
    ship_layout[next_crew_posn[0]][next_crew_posn[1]] = 'C'
    ship_layout[crew_posn[0]][crew_posn[1]] = 'O'
    return next_crew_posn


def get_valid_crew_member_moves(crew_posn: tuple[int, int],
                                bot_posn: tuple[int, int],
                                ship_layout: list[list[str]]) -> list[tuple[int, int]]:
    ship_dim = len(ship_layout)
    x, y = crew_posn
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    valid_next_positions = []
    for dx, dy in directions:
        if 0 <= x + dx < ship_dim and 0 <= y + dy < ship_dim and ship_layout[x + dx][y + dy] != '#':
            valid_next_positions.append((x + dx, y + dy))
    if bot_posn in valid_next_positions:
        valid_next_positions.remove(bot_posn)
    return valid_next_positions


def get_valid_bot_moves(bot_posn: tuple[int, int], ship_layout: list[list[str]]) -> list[tuple[int, int]]:
    directions = [(0, 0), (-1, 0), (-1, 1), (1, 0), (1, 1), (0, 1), (1, 1), (0, -1), (1, -1)]
    ship_dim = len(ship_layout)
    valid_bot_moves = []
    for dx, dy in directions:
        x = bot_posn[0] + dx
        y = bot_posn[1] + dy
        if 0 <= x + dx < ship_dim and 0 <= y + dy < ship_dim and ship_layout[x][y] != '#':
            valid_bot_moves.append((x + dx, y + dy))
    return valid_bot_moves


def get_action_space_by_bot_posn(ship_layout: list[list[str]]) -> dict[int, list[tuple[int, int]]]:
    '''
    :param ship_layout:
    :return: vectorized action space based on the bot's position, so size is [dim*dim,1]
    '''
    action_space = {}
    ship_dim = len(ship_layout)
    for i in range(ship_dim):
        for j in range(ship_dim):
            action_space[i * ship_dim + j] = get_valid_bot_moves((i, j), ship_layout)
    return action_space


def initialize_random_policy(ship_layout: list[list[str]],
                             action_space_by_bot_posn: dict[int, list[tuple[int, int]]]) -> dict[int, tuple[int, int]]:
    ship_dim = len(ship_layout)
    policy = {}
    for i in range(ship_dim * ship_dim):
        for j in range(ship_dim * ship_dim):
            state_index = 121 * i + j
            policy[state_index] = random.choice(action_space_by_bot_posn[i])
    return policy


def get_rewards(ship_layout: list[list[str]]) -> np.ndarray:
    ship_dim = len(ship_layout)
    rewards = np.zeros((ship_dim, ship_dim, ship_dim, ship_dim), float)
    for i1 in range(ship_dim):
        for j1 in range(ship_dim):
            for i2 in range(ship_dim):
                for j2 in range(ship_dim):
                    rewards[i1][j1][i2][j2] = -1.0
                    if i2 == 5 and j2 == 5 or ship_layout[i1][j1] == '#' or ship_layout[i2][j2] == '#':
                        rewards[i1][j1][i2][j2] = 0.0
    return rewards.flatten()


def initialize_values(ship_dim: int) -> np.ndarray:
    # values are of dimension 14641. Originally of the dimension 4D with 11 * 11 * 11 * 11, the first two indexes
    # indicating the bot position and the last two indexes indicate the crew member's position. The values are
    # vectorized to a 1D array for simple matrix multiplication during policy iteration value updates
    return np.zeros((ship_dim * ship_dim * ship_dim * ship_dim), float)


# def calculate_q_function(ship_layout, policy):
#     pass
#     # state is of the size 4 with the first 2 indexes indicating the position of the bot and the second two indicating
#     # the position of the crew member. So the value matrix will be a 11*11*11*11 matrix, which when vectorized is 14641
#     # P matrix should be a transition matrix depending on both the state and the action performed in that state.
#     # So if I

def policy_iteration(ship_layout):
    delta = 0.01
    ship_dim = len(ship_layout)
    action_space = get_action_space_by_bot_posn(ship_layout)
    rewards = get_rewards(ship_layout)
    current_policy = initialize_random_policy(ship_layout, action_space)
    print('Initialized policy with random actions from the valid action space')
    current_values = initialize_values(ship_dim)
    print('Initialized values 0')
    timestep = 0
    while True:
        print(f'Running policy iteration, iteration number:{timestep}')
        start_time = time.time()
        prev_values = current_values
        transition_prob = get_transition_by_policy(current_policy, ship_layout)
        print(f'Calculated transition prob and its shape is {transition_prob.shape} in {time.time()-start_time} seconds')
        current_values = update_values_by_policy(rewards, transition_prob)
        current_delta = np.max(abs(current_values - prev_values))
        print(f'Calculated values for the current policy and got delta:{current_delta} in {time.time()-start_time} '
              f'seconds')
        print(f'Update values of the shape:{current_values.shape}')
        if current_delta < delta:
            break
        current_policy = update_policy_by_current_values(current_values, rewards, action_space,
                                                         ship_layout)
        print(f'Updated policy based on the critic values in {time.time()-start_time} seconds')
        print(f'Completed iteration {timestep} in {time.time()-start_time} seconds.')
        timestep += 1
    return current_policy


def update_policy_by_current_values(current_values, rewards, action_space, ship_layout):
    ship_dim = len(ship_layout)
    policy: dict[int, tuple[int, int]] = {}
    for i in range(ship_dim * ship_dim * ship_dim * ship_dim):
        policy[i] = get_argmax_action(current_values, i, action_space, rewards, ship_layout)
    return policy


def get_argmax_action(current_values, current_state, action_space, rewards, ship_layout):
    ship_dim = len(ship_layout)
    bot_x, bot_y, crew_x, crew_y = de_vectorize_index_to_4D(current_state)
    valid_actions = action_space[bot_x * ship_dim + bot_y]
    max_value = float('-inf')
    argmax_action = None
    for action in valid_actions:
        next_crew_x, next_crew_y = action
        valid_crew_moves = get_valid_crew_member_moves((next_crew_x, next_crew_y), (bot_x, bot_y), ship_layout)
        value = rewards[current_state]
        for next_crew_x, next_crew_y in valid_crew_moves:
            j = get_vectorized_index_from_4D((next_crew_x, next_crew_y, next_crew_x, next_crew_y))
            value += (current_values[j] / len(valid_crew_moves))
        if value > max_value:
            max_value = value
            argmax_action = action
    return argmax_action


def update_values_by_policy(rewards: np.ndarray, transition_prob: np.ndarray) -> np.ndarray:
    '''
    :param rewards:
    :param transition_prob:
    :return:
    '''
    identity_size = len(transition_prob)
    identity = np.identity(identity_size)
    factor = (identity - transition_prob)
    factor_inv = np.linalg.inv(factor)
    return np.matmul(factor_inv, rewards)


def get_transition_by_policy(current_policy: dict[int,tuple[int,int]], ship_layout: list[list[str]]):
    '''
    :param current_policy:
    :param ship_layout:
    :return:
    '''
    ship_dim = len(ship_layout)
    number_of_states = int(math.pow(ship_dim, 4))  # number of states is 11 * 11 * 11 * 11
    transition_probs = np.zeros((number_of_states, number_of_states), float)
    for i in range(number_of_states):
        bot_x, bot_y, crew_x, crew_y = de_vectorize_index_to_4D(i)
        if ship_layout[bot_x][bot_y] == '#' or ship_layout[crew_x][crew_y] == '#':
            continue
        next_bot_x, next_bot_y = current_policy[i]
        valid_crew_moves = get_valid_crew_member_moves((crew_x, crew_y),
                                                       (next_bot_x, next_bot_y), ship_layout)
        for next_crew_x, next_crew_y in valid_crew_moves:
            j = get_vectorized_index_from_4D((next_bot_x, next_bot_y, next_crew_x, next_crew_y))
            transition_probs[i][j] = 1 / len(valid_crew_moves)
    return transition_probs

# def value_iteration(ship_layout):
