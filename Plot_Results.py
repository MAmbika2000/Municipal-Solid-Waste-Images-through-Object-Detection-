import warnings

warnings.filterwarnings("ignore")
from itertools import cycle
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics
import seaborn as sns


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_results():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Classifier = ['TERMS', 'ResNet', 'Inception', 'MobileNet', 'DenseNet', 'D-RAN']
    eval = np.load('Eval_All.npy', allow_pickle=True)
    value = eval[4, :, 4:]
    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[j, :])
    print('-------------------------------------------------- ', '', 'Classifier Comparison'
                                                                     '--------------------------------------------------')
    print(Table)

    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(6)
        ax.bar(X + 0.00, Graph[:, 0], edgecolor='k', hatch='..', color='#00ffff', width=0.10, label="ResNet")
        ax.bar(X + 0.10, Graph[:, 1], edgecolor='k', hatch='..', color='lime', width=0.10, label="Inception")
        ax.bar(X + 0.20, Graph[:, 2], edgecolor='k', hatch='..', color='#be03fd', width=0.10, label="MobileNet")
        ax.bar(X + 0.30, Graph[:, 3], edgecolor='k', hatch='..', color='#0485d1', width=0.10, label="DenseNet")
        ax.bar(X + 0.40, Graph[:, 4], edgecolor='w', hatch='//', color='k', width=0.10, label="D-RAN")
        plt.xticks(X + 0.25, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax'))
        plt.xlabel('Activation Functions')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path = "./Results/%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def plot_roc():
    lw = 2
    cls = ['ResNet', 'Inception', 'MobileNet', 'DenseNet', 'D-RAN']
    colors = cycle(["m", "b", "r", "lime", "k"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for j, color in zip(range(5), colors):
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[3, j], Predicted[3, j])
        auc = metrics.roc_auc_score(Actual[3, j], Predicted[3, j])
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[j]
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def plot_results_seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    value_all = Eval_all[:]
    stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
    for i in range(4, value_all[0].shape[1] - 9):
        for j in range(value_all.shape[0] + 4):
            if j < value_all.shape[0]:
                stats[i, j, 0] = np.max(value_all[j][:, i])
                stats[i, j, 1] = np.min(value_all[j][:, i])
                stats[i, j, 2] = np.mean(value_all[j][:, i])
                stats[i, j, 3] = np.median(value_all[j][:, i])
                stats[i, j, 4] = np.std(value_all[j][:, i])

        X = np.arange(stats.shape[2])
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.bar(X + 0.00, stats[i, 0, :], color='b', width=0.10, label="MBO-A-ViT-YoloV5")
        ax.bar(X + 0.10, stats[i, 1, :], color='#fe01b1', width=0.10, label="AOA-A-ViT-YoloV5")
        ax.bar(X + 0.20, stats[i, 2, :], color='lime', width=0.10, label="AGTO-A-ViT-YoloV5")
        ax.bar(X + 0.30, stats[i, 3, :], color='#ed0dd9', width=0.10, label="DBO-A-ViT-YoloV5")
        ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="E-DBO-A-ViT-YoloV5")
        plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
        plt.xlabel('Statisticsal Analysis')
        plt.ylabel(Terms[i - 4])
        plt.legend(loc=1)
        path = "./Results/seg_%s_Alg.png" % (Terms[i - 4])
        plt.savefig(path)
        plt.show()

        X = np.arange(stats.shape[2])
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.bar(X + 0.00, stats[i, 5, :], color='#a2cffe', width=0.10, label="UNet")
        ax.bar(X + 0.10, stats[i, 6, :], color='#fe01b1', width=0.10, label="Res-Unet")
        ax.bar(X + 0.20, stats[i, 7, :], color='lime', width=0.10, label="Trans-Unet")
        ax.bar(X + 0.30, stats[i, 8, :], color='#bc13fe', width=0.10, label="Yolov5")
        ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="E-DBO-A-ViT-YoloV5")
        plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
        plt.xlabel('Statisticsal Analysis')
        plt.ylabel(Terms[i - 4])
        plt.legend(loc=1)
        path = "./Results/seg_%s_met.png" % (Terms[i - 4])
        plt.savefig(path)
        plt.show()


def plot_results_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'MBO-A-ViT-YoloV5', 'AOA-A-ViT-YoloV5', 'AGTO-A-ViT-YoloV5', 'DBO-A-ViT-YoloV5',
                 'E-DBO-A-ViT-YoloV5']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):
        Conv_Graph[j, :] = stats(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- ', 'Statistical Report ',
          '--------------------------------------------------')
    print(Table)
    length = np.arange(50)
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='m', linewidth=3, marker='h', markerfacecolor='#6dedfd', markersize=12,
             label='MBO-A-ViT-YoloV5')
    plt.plot(length, Conv_Graph[1, :], color='y', linewidth=3, marker='p', markerfacecolor='green',
             markersize=12,
             label='AOA-A-ViT-YoloV5')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan',
             markersize=12,
             label='AGTO-A-ViT-YoloV5')
    plt.plot(length, Conv_Graph[3, :], color='lime', linewidth=3, marker='o', markerfacecolor='magenta',
             markersize=12,
             label='DBO-A-ViT-YoloV5')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12,
             label='E-DBO-A-ViT-YoloV5')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png")
    plt.show()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    for n in range(1):
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n]).argmax(axis=1), np.asarray(Predict[n]).argmax(axis=1))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
                    yticklabels=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'], ax=ax)

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        path = "./Results/Confusion.png"
        plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    plot_results()
    plot_roc()
    plot_results_conv()
    plot_results_seg()
    Plot_Confusion()
