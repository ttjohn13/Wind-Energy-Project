import numpy as np
import scipy as sp


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


def power_and_correctness(wind_speed_at_turbine, power_curve, target_turb):
    cut_in = np.min(power_curve[:, 0])
    cut_out = np.max(power_curve[:, 0])
    power_of_each_turb = []
    interpolation_object = sp.interpolate.interp1d(power_curve[:, 0], power_curve[:, 1])
    for wind_speed in wind_speed_at_turbine:
        if (wind_speed >= cut_in) and (wind_speed <= cut_out):
            pi = interpolation_object(wind_speed)
        else:
            pi = 0

        power_of_each_turb.append(pi)
    power_farm = sum(power_of_each_turb)
    number_of_turb = len(wind_speed_at_turbine)
    if number_of_turb != target_turb:
        max_power = np.max(power_curve[:, 1])
        error_from_target = np.abs(number_of_turb - target_turb)
        power_farm = power_farm - error_from_target * max_power
    return power_farm


individual_1 = individualCreation(8, 5, 5)

grid_1, x_grid_1, y_grid_1 = gridcreate(individual_1, 5, 5, 52)
turbine_locations = turbinelocate(grid_1, x_grid_1, y_grid_1)
ordered_turb_coord, order_num = turbineorder(turbine_locations)
alpha_1 = 0.5 / np.log(55/0.04)
M_effect = wakeeffectsmatrix(ordered_turb_coord, 52, alpha_1)
turbine_wind_speeds = turb_wind_speed(15, M_effect, ordered_turb_coord, order_num, 52, 0.5, alpha_1)

power_curve = np.array([[3, 1.7], [3.5, 15.3], [4, 30.8], [4.5, 53.7], [5, 77.4], [5.5, 106.2], [6, 139.7],
                          [6.5, 171.4], [7, 211.6], [7.5, 248.6], [8, 294.1], [8.5, 378.8], [9, 438.9], [9.5, 496.4],
                          [10, 578.4], [10.5, 629.8], [11, 668], [11.5, 742.4], [12, 783.6], [12.5, 801.3], [13, 819.4],
                          [13.5, 831.7], [14, 841.8], [14.5, 849.6], [15, 850.4], [15.5, 851.5], [16, 851.9], [25, 851.9]])
fitness = power_and_correctness(turbine_wind_speeds, power_curve, 7)


def farm_pow_calc(individual, number_of_turbines, x_size, y_size, diameter, alpha, ct, turb_power_curve, u_infinity):
    grid, x_grid, y_grid = gridcreate(individual, x_size, y_size, diameter)
    turbine_locations_coord = turbinelocate(grid, x_grid, y_grid)
    order_turb_coord, turb_order_num = turbineorder(turbine_locations_coord)
    effect_matrix = wakeeffectsmatrix(order_turb_coord, diameter, alpha)
    wind_speed_turbines = turb_wind_speed(u_infinity, effect_matrix, order_turb_coord, turb_order_num, diameter, ct, alpha)
    fitness_of_individual = power_and_correctness(wind_speed_turbines, turb_power_curve, number_of_turbines)
    return (fitness_of_individual,)
