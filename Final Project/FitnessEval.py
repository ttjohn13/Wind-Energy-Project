import numpy as np

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


individual_1 = individualCreation(8, 5, 5)

grid_1, x_grid_1, y_grid_1 = gridcreate(individual_1, 5, 5, 52)
turbine_locations = turbinelocate(grid_1, x_grid_1, y_grid_1)
