import random
from tkinter import Tk, ttk
import numpy as np

from NoBot import solve_expected_times
from Ship import get_ship


def show_tkinter(ship: np.ndarray):
    """
    :param ship: layout of the ship as a 2D matrix with each element representing whether the cell at that
                        coordinates is open/blocked/occupied by crew/teleport pad
    :return: None
    """
    root = Tk()
    table = ttk.Frame(root)
    table.grid()
    ship_dimension = len(ship)
    for row in range(ship_dimension):
        for col in range(ship_dimension):
            label = ttk.Label(table, text=ship[row][col], borderwidth=1, relief="solid")
            label.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
    root.mainloop()


def run_simulation_for_fixed_ship():
    random.seed(10)
    ship = get_ship()
    show_tkinter(ship)
    t_no_bot_grid = solve_expected_times(ship)
    # for row in t_no_bot_grid:
    #     print(["{:.2f}".format(t) if t > 0 else 'Blocked' for t in row])
    show_tkinter(t_no_bot_grid)