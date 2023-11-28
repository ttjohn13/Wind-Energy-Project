import numpy as np
from dataclasses import dataclass


@dataclass
class CoordNOrder:
    coord: np.ndarray
    order: int



def individualCreation(num_ones, y_size, x_size):
    total_elements = y_size * x_size
    zeros_array = np.zeros((total_elements, 1))
    random_indices = np.random.choice(total_elements, num_ones, replace=False)
    zeros_array[random_indices, 0] = 1
    return zeros_array


def gridcreate(individual, x_size, y_size, diam):
    grid = np.reshape(individual, (x_size, y_size))
    x_lin = np.arange(x_size) * diam
    y_lin = np.arange(y_size) * diam
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    return grid, x_grid, y_grid


def turbinelocate(grid, x_grid, y_grid):
    turbine_index = grid == 1
    x_coord = x_grid[turbine_index].reshape((-1, 1))
    y_coord = y_grid[turbine_index].reshape((-1, 1))
    turbine_coord = np.hstack((x_coord, y_coord))
    return turbine_coord


def turbineorder(turbine_coord):
    ordered_coord = np.sort(turbine_coord, axis=0)
    loc_and_order = []

    for i in range(len(ordered_coord[:, 0])):
        coordinate = ordered_coord[i, :].reshape((1, 2))
        if i>0:
            previous_coord = loc_and_order[i-1].coord
            previous_order = loc_and_order[i-1].order

            if previous_coord[0, 0] < coordinate[0, 0]:
                current_order = previous_order + 1
            elif previous_coord[0, 0] == coordinate[0, 0]:
                current_order = previous_order
            else:
                raise Exception("X coordinate not greater or equal to previous coordinate")

        else:
            current_order = 1

        current_coord_and_order = CoordNOrder(coordinate, current_order)
        loc_and_order.append(current_coord_and_order)

    return loc_and_order




individual_1 = individualCreation(8, 5, 5)

grid_1, x_grid_1, y_grid_1 = gridcreate(individual_1, 5, 5, 52)
turbine_locations = turbinelocate(grid_1, x_grid_1, y_grid_1)
ordered_turbine_locations = turbineorder(turbine_locations)
