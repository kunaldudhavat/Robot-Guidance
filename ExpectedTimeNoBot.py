import numpy as np
from Utility import de_vectorize_index_to_2D, is_neighbor, get_num_open_neighbors


def get_rewards(ship):
    rewards = np.where(ship != 'O', 0.0, -1.0)
    return rewards


def get_transition_probability(ship: list[list[str]]):
    transition_probs = np.zeros((121, 121), float)
    for i in range(121):
        for j in range(121):
            x, y = de_vectorize_index_to_2D(i)
            x1, y1 = de_vectorize_index_to_2D(j)
            if ship[x][y] != 'O' or ship[x1][y1] != 'O':
                continue
            if is_neighbor(de_vectorize_index_to_2D(i), de_vectorize_index_to_2D(j)):
                transition_probs[i][j] = 1 / get_num_open_neighbors(de_vectorize_index_to_2D(i), ship)
    return transition_probs


def evaluate_expected_values(ship):
    rewards = get_rewards(ship)
    rewards = rewards.flatten()
    transition_prob = get_transition_probability(ship)
    I = np.identity(121, float)
    factor = (I - transition_prob)
    factor_inv = np.linalg.inv(factor)
    expected_time_no_bot = np.matmul(factor_inv, rewards.T)
    expected_time_no_bot = expected_time_no_bot.reshape((len(ship), len(ship)))
    print(f'Calculated the expected values')
    expected_time_no_bot = np.where(ship == '#', float('-inf'), expected_time_no_bot)
    expected_time_no_bot = expected_time_no_bot * -1
    print(expected_time_no_bot)
