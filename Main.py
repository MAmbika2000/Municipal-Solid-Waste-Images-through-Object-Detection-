import os
import numpy as np
import cv2 as cv
from AGTO import AGTO
from AOA import AOA
from DBO import DBO
from Global_Vars import Global_vars
from Image_Results import Image_Result
from MBO import MBO
from Model_DenseNet import Model_DenseNet
from Model_Inception import Model_Inception
from Model_MTYOLOV5 import Model_Yolov5
from Model_MobileNet import Model_MobileNet
from Model_PROPOSED import Model_PROPOSED
from Model_Resnet import Model_RESNET
from PROPOSED import PROPOSED
from Plot_Results import *
from numpy import matlib
from Objfun_Detect import Objective_function
from PIL import Image


def Open_Image(File):
    image = Image.open(File)
    image = np.uint8(image)
    image = cv.resize(image, (512, 512))
    return image


def Read_image(Directory):
    Images = []
    Target = []
    in_folder = os.listdir(Directory)
    for j in range(len(in_folder)):
        indir = Directory + in_folder[j] + '/'
        in_fold = os.listdir(indir)
        for k in range(len(in_fold)):
            print(j, k)
            filename = indir + '/' + in_fold[k]
            image = Open_Image(filename)
            Images.append(image)
            Target.append(j)
            target = Target
            Targ = np.asarray(target)
            uni = np.unique(Targ)
            tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
            for i in range(len(uni)):
                ind = np.where((Targ == uni[i]))
                tar[ind[0], i] = 1
    return Images, tar


# Read Dataset
an = 0
if an == 1:
    images, target = Read_image('./Datasets/Dataset 1/Garbage classification/')
    np.save('Images.npy', images)
    np.save('Target.npy', target)


##Optimization for Segmentation
an = 0
if an == 1:
    Image = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_vars.Image = Image
    Global_vars.Target = Target
    Npop = 10
    Ch_len = 3  # optimized
    xmin = matlib.repmat(([5, 5, 300]), Npop,
                         1)
    xmax = matlib.repmat(([255, 50, 1000]), Npop, 1)
    initsol = np.zeros((xmax.shape))
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = Objective_function
    Max_iter = 50

    print("MBO...")
    [bestfit1, fitness1, bestsol1, time1] = MBO(initsol, fname, xmin, xmax, Max_iter)

    print("AOA...")
    [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)

    print("AGTO...")
    [bestfit3, fitness3, bestsol3, time3] = AGTO(initsol, fname, xmin, xmax, Max_iter)

    print("DBO...")
    [bestfit4, fitness4, bestsol4, time4] = DBO(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    best = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])

    np.save('Best_Sol_Opt.npy', best)


# Segmentation Yolov5
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Sol = np.load('Best_Sol_Opt.npy', allow_pickle=True)
    Seg = []
    for i in range(len(Images)):
        Segmented = Model_Yolov5(Images[i], Sol)
        Seg.append(Segmented)
    np.save('Segmentation.npy', Seg)


### Classification
an = 0
if an == 1:
    Feature = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Feat = Feature
    EVAL = []
    Activation = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax']
    for learn in range(len(Activation)):
        Act = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:Act, :]
        Train_Target = Target[:Act, :]
        Test_Data = Feat[Act:, :]
        Test_Target = Target[Act:, :]
        Eval = np.zeros((5, 14))
        Eval[0, :] = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[1, :] = Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[2, :] = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[3, :] = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[4, :] = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target)
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)

plot_results()
plot_roc()
plot_results_conv()
plot_results_seg()
Plot_Confusion()
Image_Result()


