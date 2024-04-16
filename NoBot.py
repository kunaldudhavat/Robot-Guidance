import numpy as np
from scipy import linalg


def get_accessible_neighbors(ship, x, y):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < ship.shape[0] and 0 <= ny < ship.shape[1] and ship[nx, ny] == 'O':
            neighbors.append((nx, ny))
    return neighbors


def solve_expected_times(ship: np.ndarray):
    grid_size = ship.shape[0]
    A = np.zeros((grid_size ** 2, grid_size ** 2))
    b = np.zeros(grid_size ** 2)

    for i in range(grid_size):
        for j in range(grid_size):
            if ship[i, j] == 'T':
                # Teleport pad cell
                A[i * grid_size + j, i * grid_size + j] = 1
                b[i * grid_size + j] = 0
            elif ship[i, j] == '#':
                # Blocked cell
                continue  # Blocked cells are not part of the system
            else:
                # Open cell
                neighbors = get_accessible_neighbors(ship, i, j)
                A[i * grid_size + j, i * grid_size + j] = 1
                for (ni, nj) in neighbors:
                    A[i * grid_size + j, ni * grid_size + nj] = -1 / len(neighbors)
                b[i * grid_size + j] = 1

    # Solve the system of linear equations
    T_flattened = linalg.solve(A, b)
    # Reshape the solution to match the grid
    T_nobot = T_flattened.reshape((grid_size, grid_size))
    return T_nobot


# def solve_expected_times(ship, blocked_cells):
#     # Initialize the matrix A and vector b for the linear equations system
#     grid_size = ship.shape[0]
#     A = np.zeros((grid_size ** 2, grid_size ** 2))
#     b = np.zeros(grid_size ** 2)
#
#     # Populate the matrix A and vector b based on the neighbors and blocked cells
#     for i in range(grid_size):
#         for j in range(grid_size):
#             if (i, j) in blocked_cells or (i, j) == (5, 5):
#                 # Blocked cells and teleport pad have fixed expected time
#                 A[i * grid_size + j, i * grid_size + j] = 1
#                 if (i, j) == (5, 5):
#                     b[i * grid_size + j] = 0  # Teleport pad has expected time of 0
#             else:
#                 # For other cells, calculate the equation based on accessible neighbors
#                 neighbors = get_accessible_neighbors(i, j)
#                 A[i * grid_size + j, i * grid_size + j] = 1
#                 for (ni, nj) in neighbors:
#                     A[i * grid_size + j, ni * grid_size + nj] = -1 / len(neighbors)
#                 b[i * grid_size + j] = 1
#
#     # Solve the system of linear equations A * T = b
#     T_flattened = linalg.solve(A, b)
#     # Reshape the solution to match the grid
#     T_nobot = T_flattened.reshape((grid_size, grid_size))
#     return T_nobot