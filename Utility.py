def is_neighbor(cell1: tuple[int, int], cell2: tuple[int, int]):
    x, y = cell1
    x1, y1 = cell2
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dx = x1 - x
    dy = y1 - y
    if (dx, dy) in directions:
        return True
    return False


def get_num_open_neighbors(position: tuple[int, int], ship_layout: list[list[str]]) -> float:
    # Define the possible directions (up, down, left, right)
    ship_dim = len(ship_layout)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors: list[tuple[int, int]] = []
    for dx, dy in directions:
        neighbor_x, neighbor_y = position[0] + dx, position[1] + dy
        if 0 <= neighbor_x < ship_dim and 0 <= neighbor_y < ship_dim and ship_layout[neighbor_x][neighbor_y] != '#':
            neighbors.append((neighbor_x, neighbor_y))
    return len(neighbors)


def de_flatten_index(flattened_index):
    return flattened_index // 11, flattened_index % 11
