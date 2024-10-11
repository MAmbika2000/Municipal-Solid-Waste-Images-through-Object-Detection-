import time
import numpy as np


def MBO(mine_positions, objective_function, ub, lb, num_iterations):
    num_dimensions, num_mines = mine_positions.shape[0], mine_positions.shape[1]
    Convergence_curve = np.zeros((num_iterations, 1))

    fitness = np.zeros((num_dimensions))
    for i in range(num_dimensions):
        fitness[i] = objective_function(mine_positions[i,:])

    ct = time.time()
    for _ in range(num_iterations):
        trial_fitness = np.zeros(num_mines)
        for i in range(num_mines):
            # Select a mine position randomly
            mine = mine_positions[i]

            # Create a trial solution by perturbing the mine position
            trial_solution = mine + np.random.uniform(-1, 1, size=num_dimensions)

            # Clip the trial solution to ensure it stays within the bounds
            trial_solution = np.clip(trial_solution, ub[:, 0], lb[:, 1])

            # Evaluate the objective function for the trial solution
            trial_fitness = objective_function(trial_solution)

            # Compare the fitness of the trial solution with the current mine
            if trial_fitness < objective_function(mine):
                mine_positions[i] = trial_solution
        Convergence_curve[_, 1] = np.min(trial_fitness)

    # Find the best mine position and its fitness
    best_mine = None
    best_fitness = np.inf
    for mine in mine_positions:
        fitness = objective_function(mine)
        if fitness < best_fitness:
            best_mine = mine
            best_fitness = fitness

    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_mine, ct
