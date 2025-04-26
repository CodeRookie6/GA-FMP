
import time
import torch
import math
import numpy as np
from sklearn import metrics
from torch import nn
#from torchinfo import summary
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
import evaluation
import matplotlib.pyplot as plt
import os
from data_feature import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import time
import torch
import math
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import shap

def evaluate(model, datadl, device="cpu"):
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in datadl:
            if len(batch) == 10:
                x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features = batch
                batch_batch = torch.zeros(gnn_features.size(0), dtype=torch.long).to(device)  # 假设只有一个图
            elif len(batch) == 11:
                x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features, batch_batch = batch
                batch_batch = batch_batch.to(device)
            else:
                raise ValueError("Unexpected batch size")

            x = x.to(device).long()
            y = y.to(device).long()
            features = {
                'aai': aai.to(device),
                'paac': paac.to(device),
                'pc6': pc6.to(device),
                'blosum62': blosum62.to(device),
                'aac': aac.to(device)
            }
            edge_index = edge_index.to(device)
            gnn_features = gnn_features.to(device)
            y_hat = model(x, z, features, edge_index, gnn_features, batch_batch)
            #predictions.extend(y_hat.squeeze().tolist())
            predictions.extend(y_hat.tolist())
           # print(predictions)
            labels.extend(y.tolist())

    scores = evaluation.evaluate(np.array(predictions), np.array(labels))
    return scores

def scoring(y_true, y_score):
    threshold = 0.5
    y_pred = [int(i >= threshold) for i in y_score]
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    if confusion_matrix.size == 1:
        # Handle cases where only one class is present
        tn, fp, fn, tp = 0, 0, 0, 0
        if y_true[0] == 0:
            tn = 1
        else:
            tp = 1
    else:
        tn, fp, fn, tp = confusion_matrix.flatten()
    sen = tp / (fn + tp) if (fn + tp) > 0 else 0
    spe = tn / (fp + tn) if (fp + tn) > 0 else 0
    pre = metrics.precision_score(y_true, y_pred, zero_division=0)
    auc = metrics.roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0
    pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(rc, pr)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    mcc = metrics.matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
    acc = metrics.accuracy_score(y_true, y_pred)
    return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)

class DataTrain:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler

        self.device = device
        self.model.to(self.device)

    def train_step(self, train_iter, epochs=None, plot_picture=True):
        x_plot = []
        y_plot = []
        epochTrainLoss = []
        steps = 1
        for epoch in range(1, epochs+1):
            start = time.time()
            total_loss = 0
            for batch in train_iter:
                if len(batch) == 10:
                    x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features = batch
                    batch_batch = torch.zeros(gnn_features.size(0), dtype=torch.long).to(self.device)  # 假设只有一个图
                elif len(batch) == 11:
                    x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features, batch_batch = batch
                    batch_batch = batch_batch.to(self.device)
                else:
                    raise ValueError("Unexpected batch size")

                self.model.train()  # 进入训练模式
                x = x.to(self.device).long()
                y = y.to(self.device).float()
                features = {
                    'aai': aai.to(self.device),
                    'paac': paac.to(self.device),
                    'pc6': pc6.to(self.device),
                    'blosum62': blosum62.to(self.device),
                    'aac': aac.to(self.device)
                }
                edge_index = edge_index.to(self.device)
                gnn_features = gnn_features.to(self.device)

                y_hat = self.model(x, z, features, edge_index, gnn_features, batch_batch)
                loss = self.criterion(y_hat, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                        self.lr_scheduler.step()
                    else:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                x_plot.append(epoch)
                y_plot.append(self.lr_scheduler(steps) if self.lr_scheduler else 0)
                total_loss += loss.item()
                steps += 1

            finish = time.time()

            print(f'[ Epoch {epoch} ', end='')
            print(f"运行时间 {finish - start:.2f}s")
            print(f'loss={total_loss / len(train_iter):.8f} ]')

        if plot_picture:
            plt.plot(x_plot, y_plot, 'r')
            plt.title('lr value of LambdaLR with (Cos_warmup)')
            plt.xlabel('step')
            plt.ylabel('lr')
            plt.savefig('./result/Cos_warmup.jpg')

def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer_, lr_lambda, last_epoch)

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        if self.warmup_steps == 0:
            return self.base_lr_orig
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, step):
        epoch = step  # 这里假设 step 与 epoch 同步
        if epoch <= self.warmup_steps and self.warmup_steps > 0:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update and self.max_steps > 0:
            return self.final_lr + (self.base_lr_orig - self.final_lr) * \
                   (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.final_lr






