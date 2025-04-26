#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:28
# @Author  : ywh
# @File    : evaluation.py
# @Software: PyCharm

from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
import numpy as np


def scores(y_test, y_pred, th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    #如果 item 的预测值小于阈值 th，则将对应的 y_predlabel 标记为 0，否则标记为 1。这是将概率预测转换为二元分类标签的常见操作，可以根据设定的阈值将连续性的预测结果转换为离散的类别标签。
    y_test = np.array([(0 if item < 1 else 1) for item in y_test])
    y_predlabel = np.array(y_predlabel)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    #混淆矩阵（Confusion Matrix）: 混淆矩阵是一个展示模型预测结果的矩阵，包括 True Positive（真正例）、True Negative（真负例）、False Positive（假正例）和 False Negative（假负例）。
    SP = tn * 1.0 / ((tn + fp) * 1.0)#特异性（Specificity，SP）: 表示预测为负例中真实负例的比例，计算公式为 tn / (tn + fp)。
    SN = tp * 1.0 / ((tp + fn) * 1.0)#灵敏度（Sensitivity，SN）或者召回率（Recall）: 表示预测为正例中真实正例的比例，计算公式为 tp / (tp + fn)。
    MCC = matthews_corrcoef(y_test, y_predlabel)#马修斯相关系数（Matthews Correlation Coefficient，MCC）: 衡量分类模型的预测结果与真实结果之间的相关性，取值范围为[-1, 1]。
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)#精准率（Precision）: 表示预测为正例中实际为正例的比例，计算公式为 tp / (tp + fp)。
    F1 = f1_score(y_test, y_predlabel)#F1 分数（F1 Score）: 综合考虑了精度和召回率的指标，是精度和召回率的调和平均值。
    Acc = accuracy_score(y_test, y_predlabel)#准确率（Accuracy）: 分类模型预测正确的样本数占总样本数的比例。
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)#ROC 曲线下面积（ROC AUC）: ROC 曲线下方的面积，用于衡量二分类模型预测结果的性能。
    AUPR = auc(recall_aupr, precision_aupr)#PR 曲线下面积（AUPR）: Precision-Recall 曲线下的面积，也是衡量二分类模型性能的指标之一。
    return Recall, SN, SP, MCC, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp

#准确率
def Aiming(y_hat, y):#y_hat 和 y 分别表示预测的标签和真实的标签
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape#其中 n 表示样本数量，m 表示标签数量。

    sorce_k = 0 #初始化变量 sorce_k，用于存储瞄准率的累加和。
    for v in range(n):#对每个样本进行循环。
        union = 0
        intersection = 0#初始化变量 union 和 intersection，分别用于记录预测标签和真实标签的并集和交集数量。
        for h in range(m):#对每个标签进行循环
            if y_hat[v, h] == 1 or y[v, h] == 1:#如果预测标签或真实标签中有一个为1，则表示预测或真实标签存在。
                union += 1#更新并集数量。
            if y_hat[v, h] == 1 and y[v, h] == 1:#如果预测标签和真实标签都为1，则表示存在交集。
                intersection += 1#更新交集数量。
        if intersection == 0:#如果交集数量为0，则跳过当前样本，因为此时无法计算瞄准率。
            continue
        sorce_k += intersection / sum(y_hat[v])#累加当前样本的瞄准率，计算方式为交集数量除以第 v 个样本的预测标签向量。
    return sorce_k / n

#召回率
def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n

#用于计算多标签分类任务中每个样本的Jaccard指数（交并比，IoU），然后将这些指数在所有样本上取平均，得到整体的准确度
def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    same
    '''

    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):#比较预测标签y_hat和真实标签y中每个样本的元素是否完全相同来判断预测是否正确
            score_k += 1
    return score_k / n

#这段代码实现了一个计算二进制标签之间汉明损失（Hamming Loss）的函数。汉明损失用于衡量预测值与真实值之间的不匹配程度，通常用于多标签分类问题。
def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        sorce_k += (union - intersection) / m
    return sorce_k / n


def evaluate(y_hat, y):
    score_label = y_hat
    aiming_list = []
    coverage_list = []
    accuracy_list = []
    absolute_true_list = []
    absolute_false_list = []
    # getting prediction label
    for i in range(len(score_label)):#这段代码是对预测标签进行处理，将小于0.5的值设为0，大于等于0.5的值设为1，以确定每个样本的最终预测标签。
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5:  # throld
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    y_hat = score_label

    aiming = Aiming(y_hat, y)
    aiming_list.append(aiming)
    coverage = Coverage(y_hat, y)
    coverage_list.append(coverage)
    accuracy = Accuracy(y_hat, y)
    accuracy_list.append(accuracy)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_true_list.append(absolute_true)
    absolute_false = AbsoluteFalse(y_hat, y)
    absolute_false_list.append(absolute_false)
    #这部分代码调用了四个评估函数（Aiming、Coverage、Accuracy和AbsoluteTrue），分别计算瞄准率、覆盖率、准确率和绝对真实率。然后将它们的结果分别添加到对应的列表中。
    return dict(aiming=aiming, coverage=coverage, accuracy=accuracy, absolute_true=absolute_true,
                absolute_false=absolute_false)




