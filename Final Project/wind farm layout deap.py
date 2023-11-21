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
    return (fitness_eval, )


def individualCreation(iclS, num_ones, y_size, x_size):
    total_elements = y_size * x_size
    zeros_array = np.zeros((total_elements, 1))
    random_indices = np.random.choice(total_elements, num_ones, replace=False)
    zeros_array[random_indices, 0] = 1
    return iclS(zeros_array)


toolbox = base.Toolbox()
toolbox.register("individual", individualCreation, creator.Individual, num_ones=target_number_of_turbines,
                 y_size=y_points, x_size=x_points)
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def main():
    pop = toolbox.population(n=25)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    cross_prob = 0.5
    mutate_prob = 0.1

    fits = [ind.fitness.values[0] for ind in pop]
    generations = 0

    while generations<1000:
        generations += 1
        print("-- Generation %i --" % generations)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random()< cross_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutate_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)

main()



