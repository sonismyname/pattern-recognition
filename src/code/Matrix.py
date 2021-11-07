from sklearn.metrics import classification_report as clr
import sklearn.metrics as mt
import numpy as np
import matplotlib.pyplot as plt


def Evalution(true_label=None, predict_label=None):
    if len(true_label) != len(predict_label) or true_label is None:
        print('Không cùng chiều')
    else:

        # matrix = mt.accuracy_score(true_label, predict_label)
        matrix = clr(true_label, predict_label)
        return matrix


def Evalution_value(true_label=None, predict_label=None):
    if len(true_label) != len(predict_label) or true_label is None:
        print('Không cùng chiều')
    else:
        value_eval = {}
        f1_score_cl2 = mt.f1_score(true_label, predict_label, pos_label=2)
        f1_score_cl4 = mt.f1_score(true_label, predict_label, pos_label=4)
        accuracy = mt.accuracy_score(true_label, predict_label)
        presision_cl2 = mt.precision_score(true_label, predict_label, pos_label=2)
        presision_cl4 = mt.precision_score(true_label, predict_label, pos_label=4)
        recall_cl2 = mt.recall_score(true_label, predict_label, pos_label=2)
        recall_cl4 = mt.recall_score(true_label, predict_label, pos_label=4)
        value_eval['f1_score_benign'] = f1_score_cl2
        value_eval['f1_score_malignant'] = f1_score_cl4
        value_eval['accuracy'] = accuracy
        value_eval['presision_benign'] = presision_cl2
        value_eval['presision_malignant'] = presision_cl4
        value_eval['recall_benign'] = recall_cl2
        value_eval['recall_malignant'] = recall_cl4
        return value_eval

def Timeline_accuracy(accuracies):
    points = []
    for i in range(len(accuracies)):
        points.append(accuracies[i])

    plt.figure(0, figsize=(8, 8))
    points = np.copy(points)
    plt.plot(points[:, 0], points[:, 1])
    plt.plot(points[:, 0], points[:, 1], 'x', c='red')
    plt.xlim(2, 9)
    plt.ylim(0.8, 1)
    plt.show()

def Timeline_f1_score(f1_score_cl2, f1_score_cl4):
    points_1 = []
    points_2 = []
    for i in range(len(f1_score_cl2)):
        points_1.append(f1_score_cl2[i])
        points_2.append(f1_score_cl4[i])

    plt.figure(1, figsize=(8, 8))
    points_1 = np.copy(points_1)
    points_2 = np.copy(points_2)
    plt.plot(points_1[:, 0], points_1[:, 1], c='green')
    plt.plot(points_1[:, 0], points_1[:, 1], 'x', c='red')
    plt.plot(points_2[:, 0], points_2[:, 1], c='yellow')
    plt.plot(points_2[:, 0], points_2[:, 1], 'x', c='black')
    plt.xlim(2, 9)
    plt.ylim(0.5, 1)
    plt.show()

def Timeline_precision(precision_cl2, precision_cl4):
    points_1 = []
    points_2 = []
    for i in range(len(precision_cl2)):
        points_1.append(precision_cl2[i])
        points_2.append(precision_cl4[i])

    plt.figure(2, figsize=(8, 8))
    points_1 = np.copy(points_1)
    points_2 = np.copy(points_2)
    plt.plot(points_1[:, 0], points_1[:, 1], c='green')
    plt.plot(points_1[:, 0], points_1[:, 1], 'x', c='red')
    plt.plot(points_2[:, 0], points_2[:, 1], c='yellow')
    plt.plot(points_2[:, 0], points_2[:, 1], 'x', c='black')
    plt.xlim(2, 9)
    plt.ylim(0.5, 1)
    plt.show()

def Timeline_recall(recall_cl2, recall_cl4):
    points_1 = []
    points_2 = []
    for i in range(len(recall_cl2)):
        points_1.append(recall_cl2[i])
        points_2.append(recall_cl4[i])

    plt.figure(3, figsize=(8, 8))
    points_1 = np.copy(points_1)
    points_2 = np.copy(points_2)
    plt.plot(points_1[:, 0], points_1[:, 1], c='green')
    plt.plot(points_1[:, 0], points_1[:, 1], 'x', c='red')
    plt.plot(points_2[:, 0], points_2[:, 1], c='yellow')
    plt.plot(points_2[:, 0], points_2[:, 1], 'x', c='black')
    plt.xlim(2, 9)
    plt.ylim(0.5, 1)
    plt.show()
