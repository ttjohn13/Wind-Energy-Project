import random

import numpy
import numpy as np
from deap import base, creator, tools
from FitnessEval import farm_pow_calc, gridcreate, turbinelocate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib


gene_space = [0, 1]  # possible values for each gene space

target_number_of_turbines = 25
turb_diameter = 52  # diameter of Vestas V52 rotor (m)
wind_speed = 15  # oncoming constant wind speed
# Create a grid of diameter spaced points that a turbine could be
x_points = 10
y_points = 10
x_coordinates = np.arange(x_points) * turb_diameter
y_coordinate = np.arange(y_points) * turb_diameter
power_curve = np.array([[3, 1.7], [3.5, 15.3], [4, 30.8], [4.5, 53.7], [5, 77.4], [5.5, 106.2], [6, 139.7],
                          [6.5, 171.4], [7, 211.6], [7.5, 248.6], [8, 294.1], [8.5, 378.8], [9, 438.9], [9.5, 496.4],
                          [10, 578.4], [10.5, 629.8], [11, 668], [11.5, 742.4], [12, 783.6], [12.5, 801.3], [13, 819.4],
                          [13.5, 831.7], [14, 841.8], [14.5, 849.6], [15, 850.4], [15.5, 851.5], [16, 851.9], [25, 851.9]])
alpha_1 = 0.5 / np.log(55/0.04)
Ct = 0.3

number_of_genes = len(x_coordinates) * len(y_coordinate)  # number of points in wind farm space

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # a max will be found
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)  # individuals will be a numpy array


def cxTwoPointCopy(ind1, ind2, number_of_turbines):
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
    
    sum1 = sum(ind1)
    sum2 = sum(ind2)
    if sum1 > number_of_turbines:
        indices = np.where(ind1 == 1)[0]
        to_turn = sum1 - number_of_turbines
        random_flip_index = np.random.choice(indices, int(to_turn), replace=False)
        ind1[random_flip_index] = 0
    elif sum1 < number_of_turbines:
        indices = np.where(ind1 == 0)[0]
        to_turn = number_of_turbines - sum1
        random_flip_index = np.random.choice(indices, int(to_turn), replace=False)
        ind1[random_flip_index] = 1
    
    if sum2 > number_of_turbines:
        indices = np.where(ind2 == 1)[0]
        to_turn = sum2 - number_of_turbines
        random_flip_index = np.random.choice(indices, int(to_turn), replace=False)
        ind2[random_flip_index] = 0
    elif sum2 < number_of_turbines:
        indices = np.where(ind2 == 0)[0]
        to_turn = number_of_turbines - sum2
        random_flip_index = np.random.choice(indices, int(to_turn), replace=False)
        ind2[random_flip_index] = 1

    return ind1, ind2


def evalOneMax(individual, number_of_turbines, x_size, y_size, diameter, alpha, ct, turb_power_curve, u_infinity):
    """Function for fitness_eval
    Going to need a penalty term for the number of turbines"""
    fitness_eval = farm_pow_calc(individual, number_of_turbines, x_size, y_size, diameter, alpha, ct, turb_power_curve, u_infinity)
    return fitness_eval


def individualCreation(iclS, num_ones, y_size, x_size):
    total_elements = y_size * x_size
    zeros_array = np.zeros((total_elements, 1))
    random_indices = np.random.choice(total_elements, num_ones, replace=False)
    zeros_array[random_indices, 0] = 1
    return iclS(zeros_array)


def mutate_func(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = not individual[i]
            del individual.fitness.values


toolbox = base.Toolbox()
toolbox.register("individual", individualCreation, creator.Individual, num_ones=target_number_of_turbines,
                 y_size=y_points, x_size=x_points)
toolbox.register("evaluate", evalOneMax, number_of_turbines=target_number_of_turbines, x_size=x_points, y_size=y_points,
                 diameter=turb_diameter, alpha=alpha_1, ct=Ct, turb_power_curve=power_curve, u_infinity=wind_speed)
toolbox.register("mate", cxTwoPointCopy, number_of_turbines=target_number_of_turbines)
toolbox.register("mutate", mutate_func, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def main():
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(key=lambda indi: indi.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    cross_prob = 0.5
    mutate_prob = 0.1

    fits = [ind.fitness.values[0] for ind in pop]
    generations = 0
    hof_track_indiv = []
    hof_track_fit = []

    while generations<200:
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

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        hof_indiv = hof.items.copy()
        hof_fit = hof.keys.copy()
        hof_track_indiv.append(hof_indiv)
        hof_track_fit.append(hof_fit[0].values[0])

        record = stats.compile(pop)
        logbook.record(gen=generations, **record)

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)

    return pop, fits, logbook, hof_track_indiv, hof_track_fit


population, fitness_of_pop, tracking, tracking_indiv, tracking_fit = main()

#%%
def update(frame):
    best_individual = tracking_indiv[frame*2]
    production = tracking_fit[frame*2]
    grid, xgrid, ygrid = gridcreate(best_individual, x_points, y_points, turb_diameter)
    coord = turbinelocate(grid, xgrid, ygrid)
    ax2.clear()
    ax2.scatter(coord[:, 0], coord[:, 1], s=50, c='red', marker='o')
    ax2.set_ylabel('y (m)')
    ax2.set_xlabel('x (m)')
    ax2.set_aspect('equal')
    ax2.set_title(f'Best Individual: {production:.0f} kW')
    ax2.set_xlim([-turb_diameter, turb_diameter * (1 + x_points)])
    ax2.set_ylim([-turb_diameter, turb_diameter * (1 + y_points)])

fig2, ax2 = plt.subplots()
num_frames = 200//2
ani = FuncAnimation(fig2, update, frames=num_frames, repeat=False)
matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\trevo\\Downloads\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe"
writer = animation.FFMpegWriter(fps=4)
ani.save('GA Wind farm layout progression.mp4', writer=writer)

#%% post processing
best_fit_pop = tracking_indiv[-1]
fitness = tracking_fit[-1]

grid_turb, x_grid, y_grid = gridcreate(best_fit_pop, x_points, y_points, turb_diameter)
turbine_coord = turbinelocate(grid_turb, x_grid, y_grid)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f'Turbine locations: {fitness: .0f} kW')
ax.set_ylabel('y (m)')
ax.set_xlabel('x (m)')
ax.set_xlim([-turb_diameter, turb_diameter*(1 + x_points)])
ax.set_ylim([-turb_diameter, turb_diameter*(1 + y_points)])
ax.scatter(turbine_coord[:, 0], turbine_coord[:, 1], s=50, c='red', marker='o')
ax.set_aspect('equal')
plt.show()

#%% Logged stats and plots

gen, avg, maximum = tracking.select('gen', 'avg', 'max')

fig1, ax1 = plt.subplots()
avg_line = ax1.plot(gen, avg, label='Average Population Fittness')
max_line = ax1.plot(gen, maximum, label='Best Fit Individual')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Farm production (kW)')
ax1.legend()
ax1.set_title('GA performance')
plt.show()


