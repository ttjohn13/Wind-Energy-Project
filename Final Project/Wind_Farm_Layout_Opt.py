import numpy as np
import scipy as sp
import pygad as pg

gene_space = [0, 1]  # possible values for each gene space

target_number_of_turbines = 15

def fitness_func(ga_instance, solution, solution_idx):
    number_of_turbines = np.sum(solution)
    turbine_count_error = np.abs(target_number_of_turbines-number_of_turbines)

