import random
import string
import time

from Ship import get_ship
import numpy as np
from ExpectedTimeWithBot import save_policy_to_csv


def decode_state(state):
    # bot_id = state // 121
    # crew_id = state % 121
    # bot_x, bot_y = bot_id // 11, bot_id % 11
    # crew_x, crew_y = crew_id // 11, crew_id % 11
    bot_x = state//(121*11)
    state = state%(121*11)
    bot_y = state//121
    state = state % 121
    crew_x = state // 11
    crew_y = state % 11
    return ((bot_x, bot_y), (crew_x, crew_y))

class Value_Iteration():
    def __init__(self):
        random.seed(10)
        self.ship = get_ship()
        # Constants
        self.N_STATES = 14641
        self.ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'NE', 'NW', 'SE', 'SW', 'STAY']
        self.GAMMA = 1  # Discount factor
        self.THRESHOLD = 0.0  # Convergence THRESHOLD

        # Ship dimensions and blocked cells setup
        self.SHIP_SIZE = (11, 11)  # Assuming an 11x11 grid
        self.blocked_cells = [(i, j) for i in range(11) for j in range(11) if self.ship[i][j]=='#']
        self.teleport_pad = (5, 5)  # Example position

    # Helper functions
    def is_valid_position(self,position):
        x, y = position
        return 0 <= x < 11 and 0 <= y < 11 and position not in self.blocked_cells

    def manhattan_distance(self,pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def encode_state(self,bot_pos, crew_pos):
        bot_id = bot_pos[0] * 11 + bot_pos[1]
        crew_id = crew_pos[0] * 11 + crew_pos[1]
        return bot_id * 121 + crew_id

    def calculate_new_position(self, position, action):
        x, y = position
        if action == 'NORTH':
            return (x, y - 1)
        elif action == 'SOUTH':
            return (x, y + 1)
        elif action == 'EAST':
            return (x + 1, y)
        elif action == 'WEST':
            return (x - 1, y)
        elif action == 'NE':
            return (x + 1, y - 1)
        elif action == 'NW':
            return (x - 1, y - 1)
        elif action == 'SE':
            return (x + 1, y + 1)
        elif action == 'SW':
            return (x - 1, y + 1)
        elif action == 'STAY':
            return (x, y)
        else:
            return position  # Invalid action

    def reward(self,bot_pos, crew_pos):
        if crew_pos == self.teleport_pad:
            return 0  # High reward for success
        else:
            return -1  # Small penalty for each move

    def value_iteration(self):
        utilities = np.zeros(self.N_STATES)
        policy = ['STAY'] * self.N_STATES  # Default policy
        iteration_num = 1
        # Value iteration
        while True:
            start_time = time.time()
            delta = 0
            for state in range(self.N_STATES):
                bot_pos, crew_pos = decode_state(state)
                if bot_pos == crew_pos or bot_pos in self.blocked_cells or crew_pos in self.blocked_cells or crew_pos==self.teleport_pad:
                    continue
                max_utility = -float('inf')
                best_action = 'STAY'
                for action in self.ACTIONS:
                    new_bot_pos = self.calculate_new_position(bot_pos, action)
                    if not self.is_valid_position(new_bot_pos):
                        continue
                    sum_utility = 0
                    distances = []
                    for crew_action in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
                        new_crew_pos = self.calculate_new_position(crew_pos, crew_action)
                        if self.is_valid_position(new_crew_pos):
                            distance = self.manhattan_distance(new_bot_pos, new_crew_pos)
                            distances.append((distance, new_crew_pos))

                    if self.manhattan_distance(bot_pos, crew_pos) == 1:
                        # Crew tries to maximize distance
                        max_distance = max(distances, key=lambda x: x[0])
                        for distance, position in distances:
                            if distance == max_distance[0]:
                                next_state = self.encode_state(new_bot_pos, position)
                                sum_utility += 1.0 / len([d for d, _ in distances if d == max_distance[0]]) * utilities[next_state]
                    else:
                        # Random movement
                        for _, position in distances:
                            next_state = self.encode_state(new_bot_pos, position)
                            sum_utility += 1.0 / len(distances) * utilities[next_state]

                    # Calculate expected utility
                    if sum_utility > max_utility:
                        max_utility = sum_utility
                        best_action = action
                # Update utility and policy
                new_utility = self.reward(bot_pos, crew_pos) + self.GAMMA * max_utility
                delta = (delta + abs(new_utility - utilities[state]))
                utilities[state] = new_utility
                policy[state] = best_action
            print(f'Completed {iteration_num}th iteration in {time.time() - start_time} seconds')
            print(f'Delta achieved in the first iteration is {delta}')
            iteration_num+=1
            if delta <= self.THRESHOLD:
                break
        print("Optimal policy computed.")
        return policy

def get_save_optimal_policy():
    vt = Value_Iteration()
    optimal_policy = vt.value_iteration()
    decoded_policy = decode_policy(optimal_policy)
    save_policy_to_csv(decoded_policy,'value_iteration.csv')

def decode_policy(policy):
    decoded_policy = np.ndarray((11,11,11,11),dtype='object')
    # print
    for x in range(14641):
        (i,j),(k,l) = decode_state(x)
        decoded_policy[i][j][k][l] = policy[x]
    return decoded_policy

if __name__ == '__main__':
    get_save_optimal_policy()
