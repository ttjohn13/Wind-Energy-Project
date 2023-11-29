import numpy as np
from dataclasses import dataclass


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
    ordered_coord = turbine_coord[np.argsort(turbine_coord[:, 0])]
    loc_and_order = []
    previous_order = 0

    for i in range(len(ordered_coord[:, 0])):
        coordinate = ordered_coord[i, :].reshape((1, 2))
        if i > 0:
            previous_coord = ordered_coord[i-1, :].reshape((1, 2))
            if previous_coord[0, 0] < coordinate[0, 0]:
                current_order = previous_order + 1
                previous_order = current_order
            elif previous_coord[0, 0] == coordinate[0, 0]:
                current_order = previous_order
            else:
                raise Exception("X coordinate not greater or equal to previous coordinate")

        else:
            current_order = 1
            previous_order = current_order

        loc_and_order.append(current_order)

    loc_and_order = np.vstack(loc_and_order)

    return ordered_coord, loc_and_order.squeeze()


def wakeeffectsmatrix(turbine_ordered_coord, diam, alpha):
    effect_matrix_size = len(turbine_ordered_coord[:, 0])
    effect_matrix = np.zeros((effect_matrix_size, effect_matrix_size), dtype=np.int_)
    for i in range(effect_matrix_size):
        if i != effect_matrix_size-1:
            for j in range(i+1, effect_matrix_size):
                delta_x = turbine_ordered_coord[i, 0] - turbine_ordered_coord[j, 0]
                delta_y = turbine_ordered_coord[i, 1] - turbine_ordered_coord[j, 1]
                if delta_x < 0:
                    diam_wake = diam - 2 * alpha * delta_x
                    tip_edge_ydist = np.abs(delta_y) - diam/2
                    if tip_edge_ydist < diam_wake/2:
                        effect_matrix[i, j] = 1
                        effect_matrix[j, i] = -1

    return effect_matrix


def turb_wind_speed(uinf, effect_matrix, turb_ordered_coord, turb_order, diam, Ct, alpha):
    number_of_turb = len(turb_order)
    wind_speed_at_turb =[]
    for i in range(number_of_turb):
        vi = []
        indices = np.where(effect_matrix[i, :] == -1)[0]
        for j in indices:
            delta_x = turb_ordered_coord[i, 0] - turb_ordered_coord[j, 0]
            delta_y = turb_ordered_coord[i, 1] - turb_ordered_coord[j, 1]
            s = delta_x / diam
            if s >= 3:
                v1_part = uinf * (1 - (1 - np.sqrt((1-Ct)))/(1+2*alpha * s)**2)
                vi.append(v1_part)
            elif s < 3:
                v1_part = uinf * np.sqrt(1 - Ct)
                vi.append(v1_part)
        if not vi:
            ui = uinf

        else:
            vi = np.stack(vi)
            ui = uinf * (1 - np.sqrt(np.sum((1 - vi/uinf))**2))

        wind_speed_at_turb.append(ui)

    wind_speed_at_turb = np.stack(wind_speed_at_turb)
    return wind_speed_at_turb.squeeze()

individual_1 = individualCreation(8, 5, 5)

grid_1, x_grid_1, y_grid_1 = gridcreate(individual_1, 5, 5, 52)
turbine_locations = turbinelocate(grid_1, x_grid_1, y_grid_1)
ordered_turb_coord, order_num = turbineorder(turbine_locations)
alpha_1 = 0.5 / np.log(55/0.04)
M_effect = wakeeffectsmatrix(ordered_turb_coord, 52, alpha_1)
turbine_wind_speeds = turb_wind_speed(15, M_effect, ordered_turb_coord, order_num, 52, 0.5, alpha_1)
