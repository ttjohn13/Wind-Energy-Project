import random

import numpy as np
from deap import base, creator, tools

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

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # a max will be found
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)  # individuals will be a numpy array


def cxTwoPointCopy(ind1, ind2):
    """Deals with issue of numpy array viewing inside deap module"""
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size-1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 +=1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


def evalOneMax(individual):
    """Function for fitness_eval
    Going to need a penalty term for the number of turbines"""
    fitness_eval = np.sum(individual)
    return fitness_eval


def individualCreation(iclS, num_ones, y_size, x_size):
    total_elements = y_size * x_size
    zeros_array = np.zeros((total_elements, 1))
    random_indices = np.random.choice(total_elements, num_ones, replace=False)
    zeros_array[random_indices, 1] = 1
    iclS(zeros_array)
    return zeros_array


toolbox = base.Toolbox()
toolbox.register("individual", individualCreation, creator.Individual, num_ones=target_number_of_turbines,
                 y_size=y_points, x_size=x_points)
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.001)



