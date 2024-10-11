import numpy as np
from Evaluation import net_evaluation
from Global_Vars import Global_vars
from Model_MTYOLOV5 import Model_Yolov5


def Objective_function(Soln):
    Image = Global_vars.Image
    Target = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            pred = Model_Yolov5(Image, sol)
            Eval = net_evaluation(pred, Target)
            Fitn[i] = 1 / (Eval[4] + Eval[6])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        pred = Model_Yolov5(Image, sol)
        Eval = net_evaluation(pred, Target)
        Fitn = 1 / (Eval[4] + Eval[6])
        return Fitn
