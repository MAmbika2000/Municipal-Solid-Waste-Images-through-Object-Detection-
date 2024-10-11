import time
import numpy as np

#Dung Beetle Optimizer (DBO)

def DBO(x, fobj, lb, ub, max_iter):
    pop, dim = x.shape[0], x.shape[1]

    # Initialization
    x = np.zeros((pop, dim))
    fit = np.zeros(pop)
    Convergence = np.zeros(max_iter)
    for i in range(pop):
        x[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        fit[i] = fobj(x[i, :])
    pFit = fit.copy()
    pX = x.copy()
    XX = pX.copy()

    # Find the global optimum fitness value and position
    fMin = np.min(fit)
    bestI = np.argmin(fit)
    bestX = x[bestI, :]
    ct = time.time()
    # Start updating the solutions
    for t in range(max_iter):
        fmax, B = np.max(fit), np.argmax(fit)
        worse = x[B, :]
        r = np.random.rand(1)
        for i in range(pop):
            if r < 0.9:
                a = 1 if np.random.rand(1) > 0.1 else -1
                x[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])  # Equation (1)
            else:
                aaa = np.random.randint(0, 181, 1)[0]
                if aaa == 0 or aaa == 90 or aaa == 180:
                    x[i, :] = pX[i, :]
                theta = aaa * np.pi / 180
                x[i, :] = pX[i, :] + np.tan(theta) * np.abs(pX[i, :] - XX[i, :])  # Equation (2)
            x[i, :] = np.clip(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        # Find the current optimum fitness value and position
        fMMin = np.min(fit)
        bestII = np.argmin(fit)
        bestXX = x[bestII, :]

        R = 1 - t / max_iter
        Xnew1 = bestXX * (1 - R)
        Xnew2 = bestXX * (1 + R)  # Equation (3)
        Xnew1 = np.clip(Xnew1, lb, ub)
        Xnew2 = np.clip(Xnew2, lb, ub)

        Xnew11 = bestX * (1 - R)
        Xnew22 = bestX * (1 + R)  # Equation (5)
        Xnew11 = np.clip(Xnew11, lb, ub)
        Xnew22 = np.clip(Xnew22, lb, ub)

        # Equation (4)
        for i in range(pop + 1, 12):
            x[i, :] = bestXX + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (pX[i, :] - Xnew2))
            x[i, :] = np.clip(x[i, :], Xnew1, Xnew2)
            fit[i] = fobj(x[i, :])

        # Equation (6)
        for i in range(12, 19):
            x[i, :] = pX[i, :] + (
                        (np.random.randn(1)) * (pX[i, :] - Xnew11) + (np.random.rand(1, dim)) * (pX[i, :] - Xnew22))
            x[i, :] = np.clip(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        # Equation (7)
        for j in range(19, pop):
            x[j, :] = bestX + np.random.randn(1, dim) * (
                        (np.abs((pX[j, :] - bestXX))) + (np.abs((pX[j, :] - bestX)))) / 2
            x[j, :] = np.clip(x[j, :], lb, ub)
            fit[j] = fobj(x[j, :])

        for j in range(20, pop):  # Equation (7)
            x[j, :] = bestX + np.random.randn(1, dim) * ((np.abs(pX[j, :] - bestXX)) + (np.abs(pX[j, :] - bestX))) / 2
            x[j, :] = np.clip(x[j, :], lb, ub)
            fit[j] = fobj(x[j, :])

        # Update the individual's best fitness value and the global best fitness value
        XX = pX.copy()
        for i in range(pop):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i, :] = x[i, :]

            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i, :]
        Convergence[t] = fMin
    ct = time.time() - ct
    return fMin, Convergence, bestX, ct
