import time

import numpy as np


def fun_checkpositions(dim, vec_pos, var_no_group, lb, ub):
    Lb = np.ones(dim) * lb
    Ub = np.ones(dim) * ub
    for i in range(var_no_group):
        vec_pos[i, :] = np.maximum(vec_pos[i, :], Lb)
        vec_pos[i, :] = np.minimum(vec_pos[i, :], Ub)
    return vec_pos


def AOA(Materials_no, fobj, lb, ub, Max_iter):
    N, dim = Materials_no.shape[0], Materials_no.shape[1]
    # Initialization
    C1 = 2
    C2 = 6
    C3 = 5
    C4 = 7
    u = 0.9
    l = 0.1

    X = lb + np.random.rand(Materials_no, dim) * (ub - lb)  # Initial positions Eq. (4)
    den = np.random.rand(Materials_no, dim)  # Eq. (5)
    vol = np.random.rand(Materials_no, dim)
    acc = lb + np.random.rand(Materials_no, dim) * (ub - lb)  # Eq. (6)

    Y = np.array([fobj(x) for x in X])
    Scorebest = np.min(Y)
    Score_index = np.argmin(Y)
    Xbest = X[Score_index, :]
    den_best = den[Score_index, :]
    vol_best = vol[Score_index, :]
    acc_best = acc[Score_index, :]
    acc_norm = acc

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        TF = np.exp((t - Max_iter) / Max_iter)  # Eq. (8)
        TF = min(TF, 1)

        d = np.exp((Max_iter - t) / Max_iter) - (t / Max_iter)  # Eq. (9)

        acc = acc_norm
        r = np.random.rand()

        for i in range(Materials_no):
            den[i, :] += r * (den_best - den[i, :])  # Eq. (7)
            vol[i, :] += r * (vol_best - vol[i, :])
            if TF < 0.45:  # collision
                mr = np.random.randint(Materials_no)
                acc_temp = (den[mr, :] + (vol[mr, :] * acc[mr, :])) / (
                            np.random.rand() * den[i, :] * vol[i, :])  # Eq. (10)
            else:
                acc_temp = (den_best + (vol_best * acc_best)) / (np.random.rand() * den[i, :] * vol[i, :])  # Eq. (11)

        acc_norm = ((u * (acc_temp - np.min(acc_temp))) / (np.max(acc_temp) - np.min(acc_temp))) + l  # Eq. (12)

        Xnew = np.zeros_like(X)

        for i in range(Materials_no):
            if TF < 0.4:
                for j in range(dim):
                    mrand = np.random.randint(Materials_no)
                    Xnew[i, j] = X[i, j] + C1 * np.random.rand() * acc_norm[i, j] * (
                                X[mrand, j] - X[i, j]) * d  # Eq. (13)
            else:
                for j in range(dim):
                    p = 2 * np.random.rand() - C4  # Eq. (15)
                    T = C3 * TF
                    T = min(T, 1)
                    if p < 0.5:
                        Xnew[i, j] = Xbest[j] + C2 * np.random.rand() * acc_norm[i, j] * (
                                    T * Xbest[j] - X[i, j]) * d  # Eq. (14)
                    else:
                        Xnew[i, j] = Xbest[j] - C2 * np.random.rand() * acc_norm[i, j] * (T * Xbest[j] - X[i, j]) * d

        Xnew = fun_checkpositions(dim, Xnew, Materials_no, lb, ub)

        for i in range(Materials_no):
            v = fobj(Xnew[i, :])
            if v < Y[i]:
                X[i, :] = Xnew[i, :]
                Y[i] = v

        var_Ybest = np.min(Y)
        var_index = np.argmin(Y)
        Convergence_curve.append(var_Ybest)

        if var_Ybest < Scorebest:
            Scorebest = var_Ybest
            Score_index = var_index
            Xbest = X[var_index, :]
            den_best = den[Score_index, :]
            vol_best = vol[Score_index, :]
            acc_best = acc_norm[Score_index, :]

    return Xbest, Convergence_curve, Scorebest, ct
