import numpy as np
import scipy as sp
import pygad as pg

gene_space = [0, 1]  # possible values for each gene space

target_number_of_turbines = 15
turb_diameter = 52  # diameter of Vestas V52 rotor (m)
wind_speed = 15  # oncoming constant wind speed
# Create a grid of diameter spaced points that a turbine could be
x_points = 25
y_points = 25
x_coordinates = np.arange(x_points) * turb_diameter
y_coordinate = np.arange(y_points) * turb_diameter

number_of_genes = len(x_coordinates) * len(y_coordinate)  # number of points in wind farm space



def fitness_func(ga_instance, solution, solution_idx):
    number_of_turbines = np.sum(solution)
    turbine_count_error = np.abs(target_number_of_turbines-number_of_turbines)

