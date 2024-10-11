import time

import numpy as np


def boundary_check(population, lower_bound, upper_bound):
    population = np.where(population < lower_bound, lower_bound, population)
    population = np.where(population > upper_bound, upper_bound, population)
    return population


#### Artificial Gorilla Troops Optimizer (AGTO)
def AGTO(X, fobj, lower_bound, upper_bound, max_iter):
    # Initialize Silverback

    pop_size, variables_no = X.shape[0], X.shape[1]
    convergence_curve = np.zeros(max_iter)
    Silverback = np.zeros(variables_no)
    Silverback_Score = np.inf
    Pop_Fit = np.zeros(pop_size)

    for i in range(pop_size):
        Pop_Fit[i] = fobj(X[i, :])
        if Pop_Fit[i] < Silverback_Score:
            Silverback_Score = Pop_Fit[i]
            Silverback = np.copy(X[i, :])

    GX = np.copy(X)
    lb = np.ones(variables_no) * lower_bound
    ub = np.ones(variables_no) * upper_bound

    # Controlling parameters
    p = 0.03
    Beta = 3
    w = 0.8
    ct = time.time()
    # Main loop
    for It in range(max_iter):
        a = (np.cos(2 * np.random.rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * np.random.rand() - 1)

        # Exploration
        for i in range(pop_size):
            if np.random.rand() < p:
                GX[i, :] = (ub - lb) * np.random.rand() + lb
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(-a, a, variables_no)
                    H = Z * X[i, :]
                    GX[i, :] = (np.random.rand() - a) * X[np.random.randint(0, pop_size), :] + C * H
                else:
                    GX[i, :] = X[i, :] - C * (C * (X[i, :] - GX[np.random.randint(0, pop_size), :]) +
                                              np.random.rand() * (X[i, :] - GX[np.random.randint(0, pop_size), :]))

        GX = boundary_check(GX, lower_bound, upper_bound)

        # Group formation operation
        for i in range(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]

        # Exploitation
        for i in range(pop_size):
            if a >= w:
                g = 2 ** C
                delta = (np.abs(np.mean(GX)) ** g) ** (1 / g)
                GX[i, :] = C * delta * (X[i, :] - Silverback) + X[i, :]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.randn(1, variables_no)
                else:
                    h = np.random.randn(1, 1)
                r1 = np.random.rand()
                GX[i, :] = Silverback - (Silverback * (2 * r1 - 1) - X[i, :] * (2 * r1 - 1)) * (Beta * h)

        GX = boundary_check(GX, lower_bound, upper_bound)

        # Group formation operation
        for i in range(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]

        convergence_curve[It] = Silverback_Score
    ct = time.time() - ct
    return Silverback_Score, convergence_curve, Silverback, ct
