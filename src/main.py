#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/26 21:04
# @Author : fhh
# @FileName: model.py
# @Software: PyCharm


# import os
# import csv
# import pandas as pd
# import torch
#
# from model import *
# from torch.utils.data import DataLoader
# from loss_functions import *
# from train import *
# import random
#
#
#
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
#              'AVP',
#              'BBP', 'BIP',
#              'CPP', 'DPPIP',
#              'QSP', 'SBP', 'THP']
#
#
# def PadEncode(data, label, max_len):
#     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
#     data_e, label_e, seq_length, temp = [], [], [], []
#     sign, b = 0, 0
#     for i in range(len(data)):
#         length = len(data[i])
#         elemt, st = [], data[i].strip()
#         for j in st:
#             if j not in amino_acids:
#                 sign = 1
#                 break
#             index = amino_acids.index(j)
#             elemt.append(index)
#             sign = 0
#
#         if length <= max_len and sign == 0:
#             temp.append(elemt)
#             seq_length.append(len(temp[b]))
#             b += 1
#             elemt += [0] * (max_len - length)
#             data_e.append(elemt)
#             label_e.append(label[i])
#     return np.array(data_e), np.array(label_e), np.array(seq_length)
#
#
#
#
# def getSequenceData(first_dir, file_name):
#     # getting sequence data and label
#     data, label = [], []
#     path = "{}/{}.txt".format(first_dir, file_name)
#
#     with open(path) as f:
#         for each in f:
#             each = each.strip()
#             if each[0] == '>':
#                 label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
#             else:
#                 data.append(each)
#
#     return data, label
#
# #函数接受训练集标签 y_train 和测试集标签 y_test 作为输入
# def staticTrainAndTest(y_train, y_test):
#     #创建了两个长度为 filenames 长度的全零数组 data_size_tr 和 data_size_te，用于存储训练集和测试集中每个类别的样本数量。
#     data_size_tr = np.zeros(len(filenames))
#     data_size_te = np.zeros(len(filenames))
#
#     #通过两层循环遍历训练集 y_train 中的每个样本和每个类别，并判断是否大于 0。如果大于 0，表示该样本属于该类别，因此对应类别的 data_size_tr 加一，
#     # 用于统计该类别在训练集中的样本数量。
#     for i in range(len(y_train)):
#         for j in range(len(y_train[i])):
#             if y_train[i][j] > 0:
#                 data_size_tr[j] += 1
#
#     for i in range(len(y_test)):
#         for j in range(len(y_test[i])):
#             if y_test[i][j] > 0:
#                 data_size_te[j] += 1
#
#     return data_size_tr
#
#
#
#
#
# def main(num, data):
#     first_dir = 'dataset'
#
#     max_length = 50  # the longest length of the peptide sequence
#
#     #调用 getSequenceData 函数，分别获取训练集和测试集的序列数据和标签数据。
#     # getting train data and test data
#
#     train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
#     test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')
#
#     #将训练集和测试集的标签数据转换为 NumPy 数组。
#     # Converting the list collection to an array
#     y_train = np.array(train_sequence_label)
#     y_test = np.array(test_sequence_label)
#
#     #调用 PadEncode 函数对训练集和测试集的序列数据进行填充和编码处理，并获取填充后的序列数据、标签数据以及序列长度。
#     # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
#     x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
#     x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
#
#
#
#     # x_train_np = np.array(x_train)
#     # y_train_np = np.array(y_train)
#     # y_test_np = np.array(y_test)
#     # x_test_np = np.array(x_test)
#     # data_size = staticTrainandTest(y_train_np, y_test_np)
#
#     #将训练集和测试集的序列数据、序列长度以及标签数据转换为 PyTorch 张量。
#     x_train = torch.LongTensor(x_train)  # torch.Size([7872, 50])
#     x_test = torch.LongTensor(x_test)  # torch.Size([1969, 50])
#     train_length = torch.LongTensor(train_length)
#     y_test = torch.Tensor(y_test)
#     y_train = torch.Tensor(y_train)
#     test_length = torch.LongTensor(test_length)
#
#     #将训练集和测试集的数据集封装为 PyTorch 的 DataLoader 对象，并指定批量大小、是否打乱数据以及是否使用固定内存。
#     """Create a dataset and split it"""
#     dataset_train = list(zip(x_train, y_train, train_length))
#     dataset_test = list(zip(x_test, y_test, test_length))
#     dataset_train = DataLoader(dataset_train, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#     dataset_test = DataLoader(dataset_test, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#
#     #设置了保存模型的路径，并将测试集数据集保存为名为 tea_data[num].h5 的文件。
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'data', 'tea_data' + str(num) + '.h5')
#     torch.save(dataset_test, each_model)
#     # 设置训练参数
#     vocab_size = 50
#     output_size = 21
#
#     # 类别权重
#     # class_weights = []  # 类别权重
#     # sumx = sum(data_size)
#     #
#     # m1 = (np.max(data_size) / sumx)
#     # for m in range(len(data_size)):
#     #     # x = {m: 18*math.pow(int((math.log((sumx / data_size[m]), 2))),2)}
#     #     # x = int(sumx / (data_size[m]))
#     #     # x = int((math.log((sumx / data_size[m]), 2)))
#     #     # x = 8 * math.pow(int((math.log((sumx / data_size[m]), 2))), 2)
#     #     x = math.pow(int((math.log((sumx / data_size[m]), 2))), 2)
#     #     class_weights.append(x)  # 更新权重
#     # class_weights = torch.Tensor(class_weights)
#
#     # 初始化参数训练模型相关参数
#     model = ETFC(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'],
#                  data['num_heads'])
#     rate_learning = data['learning_rate']
#     optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
#     lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
#
#     # criterion = nn.BCEWithLogitsLoss()
#     # BCELoss https://www.jianshu.com/p/63e255e3232f
#     #criterion = BCEFocalLoss(gamma=10)
#     # criterion = BCEFocalLoss(class_weight=class_weights)
#     #criterion = GHMC(label_weight=class_weights)
#     #criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.2, reduction='sum')
#     #criterion = GHMC(label_weight=class_weights, class_weight=class_weights)
#     #criterion = BinaryDiceLoss()
#    # criterion = DCSLoss() #会丢失loss
#
#     """添加的损失函数"""
#     #criterion = ComboLoss()
#     #criterion = TverskyLoss()#没用
#     #criterion = DiceLoss()#loss反向增大
#     #criterion = FocalLoss()
#     #criterion = LDAM_loss(max_m=0.5, class_weight="balanced")#coverage明显增加，但其他指标下降
#     #criterion = APLLoss()
#     #criterion = AsymmetricLoss()
#
#
#
#
#     """"""
#
#     criterion = FocalDiceLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight'])
#
#     # 创建初始化训练类
#     Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)
#
#     a = time.time()
#     Train.train_step(dataset_train, epochs=data['epochs'], plot_picture=False)
#     b = time.time()
#     test_score = evaluate(model, dataset_test, device=DEVICE)
#     runtime = b - a
#
#     "-------------------------------------------保存模型参数-----------------------------------------------"
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
#     torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
#     "---------------------------------------------------------------------------------------------------"
#
#     "-------------------------------------------输出模型结果-----------------------------------------------"
#     print(f"runtime:{runtime:.3f}s")
#     print("测试集：")
#     print(f'aiming: {test_score["aiming"]:.3f}')
#     print(f'coverage: {test_score["coverage"]:.3f}')
#     print(f'accuracy: {test_score["accuracy"]:.3f}')
#     print(f'absolute_true: {test_score["absolute_true"]:.3f}')
#     print(f'absolute_false: {test_score["absolute_false"]:.3f}')
#     "---------------------------------------------------------------------------------------------------"
#
#     "-------------------------------------------保存模型结果-----------------------------------------------"
#     title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
#
#     model_name = "tea-CELoss"
#
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     content = [[model_name, '%.3f' % test_score["aiming"],
#                 '%.3f' % test_score["coverage"],
#                 '%.3f' % test_score["accuracy"],
#                 '%.3f' % test_score["absolute_true"],
#                 '%.3f' % test_score["absolute_false"],
#                 '%.3f' % runtime,
#                 now]]
#
#     path = "{}/{}.csv".format('result', 'teacher')
#
#     if os.path.exists(path):
#         data1 = pd.read_csv(path, header=None)
#         one_line = list(data1.iloc[0])
#         if one_line == title:
#             with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
#                 writer = csv.writer(t)  # 这一步是创建一个csv的写入器
#                 writer.writerows(content)  # 写入样本数据
#         else:
#             with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
#                 writer = csv.writer(t)  # 这一步是创建一个csv的写入器
#                 writer.writerow(title)  # 写入标签
#                 writer.writerows(content)  # 写入样本数据
#     else:
#         with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
#             writer = csv.writer(t)  # 这一步是创建一个csv的写入器
#
#             writer.writerow(title)  # 写入标签
#             writer.writerows(content)  # 写入样本数据
#     "---------------------------------------------------------------------------------------------------"
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     clip_pos = 0.7
#     #clip_neg = 0.5
#     clip_neg = 0.5
#
#     #pos_weight = 0.3
#     pos_weight = 0.3
#
#     #batch_size = 200
#     batch_size = 200
#     #epochs = 200
#     epochs = 200
#
#     #learning_rate = 0.0018
#     learning_rate = 0.0018
#
#     embedding_size = 256
#     #dropout = 0.6
#     dropout = 0.6
#     fan_epochs = 1
#     num_heads = 8
#
#
#     para = {'clip_pos': clip_pos,
#             'clip_neg': clip_neg,
#             'pos_weight': pos_weight,
#             'batch_size': batch_size,
#             'epochs': epochs,
#             'learning_rate': learning_rate,
#             'embedding_size': embedding_size,
#             'dropout': dropout,
#             'fan_epochs': fan_epochs,
#             'num_heads': num_heads}
#     for i in range(10):
#         main(i, para)



"""四个特征用这个"""
# import datetime
# import os
# import csv
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from loss_functions import FocalDiceLoss
# from train import DataTrain, evaluate, CosineScheduler
# from data_feature import AAI_embedding, PAAC_embedding, PC6_embedding, BLOSUM62_embedding
# from model import *
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
#              'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']
#
# def PadEncode(data, label, max_len):
#     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
#     data_e, label_e, seq_length, temp = [], [], [], []
#     sign, b = 0, 0
#     for i in range(len(data)):
#         length = len(data[i])
#         elemt, st = [], data[i].strip()
#         #st = augment_sequence(st)#数据增强，随机替换氨基酸序列
#         for j in st:
#             if j not in amino_acids:
#                 sign = 1
#                 break
#             index = amino_acids.index(j)
#             elemt.append(index)
#             sign = 0
#
#         if length <= max_len and sign == 0:
#             temp.append(elemt)
#             seq_length.append(len(temp[b]))
#             b += 1
#             elemt += [0] * (max_len - length)
#             data_e.append(elemt)
#             label_e.append(label[i])
#     return np.array(data_e), np.array(label_e), np.array(seq_length)
#
#
# #会打印读取到的序列和标签总数
# def getSequenceData(first_dir, file_name):
#     data, label = [], []
#     path = "{}/{}.txt".format(first_dir, file_name)
#
#     with open(path, 'r') as f:
#         for each in f:
#             each = each.strip()
#             if each:  # Check if line is not empty
#                 if each[0] == '>':  # This assumes that labels are prefixed with '>'
#                     try:
#                         numeric_label = np.array(list(map(int, each[1:])), dtype=int)  # Convert string labels to numeric vectors
#                         label.append(numeric_label)
#                     except ValueError:
#                         print("Invalid label encountered and skipped:", each[1:])
#                 else:
#                     data.append(each)
#
#     print("Total sequences read:", len(data))
#     print("Total labels read:", len(label))
#     return data, label
#
# def staticTrainAndTest(y_train, y_test):
#     data_size_tr = np.zeros(len(filenames))
#     data_size_te = np.zeros(len(filenames))
#
#     for i in range(len(y_train)):
#         for j in range(len(y_train[i])):
#             if y_train[i][j] > 0:
#                 data_size_tr[j] += 1
#
#     for i in range(len(y_test)):
#         for j in range(len(y_test[i])):
#             if y_test[i][j] > 0:
#                 data_size_te[j] += 1
#
#     return data_size_tr
#
# def main(num, data):
#     first_dir = 'dataset'
#     max_length = 50
#
#     train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
#     test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')
#
#
#     y_train = np.array(train_sequence_label)
#     y_test = np.array(test_sequence_label)
#
#     x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
#     x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
#
#
#     train_features_aai = AAI_embedding(train_sequence_data, max_len=max_length)
#     test_features_aai = AAI_embedding(test_sequence_data, max_len=max_length)
#     train_features_paac = PAAC_embedding(train_sequence_data, max_len=max_length)
#     test_features_paac = PAAC_embedding(test_sequence_data, max_len=max_length)
#     train_features_pc6 = PC6_embedding(train_sequence_data, max_len=max_length)
#     test_features_pc6 = PC6_embedding(test_sequence_data, max_len=max_length)
#     train_features_blosum62 = BLOSUM62_embedding(train_sequence_data, max_len=max_length)
#     test_features_blosum62 = BLOSUM62_embedding(test_sequence_data, max_len=max_length)
#     train_features_aac = AAC_embedding(train_sequence_data)
#     test_features_aac = AAC_embedding(test_sequence_data)
#
#
#
#
#     train_features = {'aai': train_features_aai, 'paac': train_features_paac, 'pc6': train_features_pc6, 'blosum62': train_features_blosum62,'aac':train_features_aac}
#     test_features = {'aai': test_features_aai, 'paac': test_features_paac, 'pc6': test_features_pc6, 'blosum62': test_features_blosum62,'aac':test_features_aac}
#
#     x_train = torch.LongTensor(x_train)
#     x_test = torch.LongTensor(x_test)
#     train_length = torch.LongTensor(train_length)
#     y_test = torch.Tensor(y_test)
#     y_train = torch.Tensor(y_train)
#     test_length = torch.LongTensor(test_length)
#
#
#     dataset_train = list(zip(x_train, y_train, train_length, train_features['aai'], train_features['paac'], train_features['pc6'], train_features['blosum62'],train_features['aac']))
#     dataset_test = list(zip(x_test, y_test, test_length, test_features['aai'], test_features['paac'], test_features['pc6'], test_features['blosum62'],test_features['aac']))
#     dataset_train = DataLoader(dataset_train, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#     dataset_test = DataLoader(dataset_test, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'data', 'tea_data' + str(num) + '.h5')
#     torch.save(dataset_test, each_model)
#
#     vocab_size = 50
#     output_size = 21
#
#     model = ETFC(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'], data['num_heads'])
#     rate_learning = data['learning_rate']
#     #optimizer = torch.optim.NAdam(model.parameters(), lr=rate_learning)
#     optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
#     lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
#     #criterion = LDAM_loss(max_m=0.5, class_weight="balanced")
#     #criterion = FocalDiceLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight'])
#     #criterion = AsymmetricLoss()
#     #criterion = FocalLoss()
#     #criterion = ComboLoss()
#     #criterion = nn.CrossEntropyLoss()#没效果
#     #criterion = APLLoss()
#     #criterion = PartialSelectiveLoss()
#     #criterion = FocalLDAMLoss()
#     #criterion = FocalAsymmetricLoss()
#     criterion = MarginalFocalDiceLoss()
#
#
#
#
#     Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)
#
#     a = time.time()
#     Train.train_step(dataset_train, epochs=data['epochs'], plot_picture=True)
#     b = time.time()
#     test_score = evaluate(model, dataset_test, device=DEVICE)
#     runtime = b - a
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
#     torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
#
#     print(f"runtime:{runtime:.3f}s")
#     print("测试集：")
#     print(f'aiming: {test_score["aiming"]:.3f}')
#     print(f'coverage: {test_score["coverage"]:.3f}')
#     print(f'accuracy: {test_score["accuracy"]:.3f}')
#     print(f'absolute_true: {test_score["absolute_true"]:.3f}')
#     print(f'absolute_false: {test_score["absolute_false"]:.3f}')
#
#     title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
#     model_name = "tea-CELoss"
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     content = [[model_name, '%.3f' % test_score["aiming"], '%.3f' % test_score["coverage"], '%.3f' % test_score["accuracy"],
#                 '%.3f' % test_score["absolute_true"], '%.3f' % test_score["absolute_false"], '%.3f' % runtime, now]]
#
#     path = "{}/{}.csv".format('result', 'teacher')
#     if os.path.exists(path):
#         data1 = pd.read_csv(path, header=None)
#         one_line = list(data1.iloc[0])
#         if one_line == title:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerows(content)
#         else:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerow(title)
#                 writer.writerows(content)
#     else:
#         with open(path, 'a+', newline='') as t:
#             writer = csv.writer(t)
#             writer.writerow(title)
#             writer.writerows(content)
#
#
# if __name__ == '__main__':
#     clip_pos = 0.7
#     clip_neg = 0.5
#     #pos_weight = 0.3#本地
#     pos_weight = 0.7
#
#     batch_size = 256
#     #epochs = 256#本地
#     epochs = 256
#
#      # batch_size = 200#本地
#     # epochs = 200#本地
#
#     learning_rate = 0.0018
#
#
#
#     embedding_size = 256
#
#
#     dropout = 0.6
#     fan_epochs = 1
#     num_heads = 8
#
#     para = {'clip_pos': clip_pos, 'clip_neg': clip_neg, 'pos_weight': pos_weight, 'batch_size': batch_size,
#             'epochs': epochs, 'learning_rate': learning_rate, 'embedding_size': embedding_size, 'dropout': dropout,
#             'fan_epochs': fan_epochs, 'num_heads': num_heads}
#     for i in range(10):
#         main(i, para)





#粒子群进化算法，没什么用
# import datetime
# import os
# import csv
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from loss_functions import FocalDiceLoss, CombinedLDAMFocalDiceLoss  # 确保导入您的损失函数
# from train import DataTrain, evaluate, CosineScheduler
# from data_feature import AAI_embedding, PAAC_embedding, PC6_embedding, BLOSUM62_embedding, AAC_embedding
# from model import ETFC
# import time
# import evaluation  # 确保导入 evaluation.py
# from sklearn.model_selection import train_test_split
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP',
#              'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']
#
# def PadEncode(data, label, max_len):
#     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
#     data_e, label_e, seq_length, temp = [], [], [], []
#     sign, b = 0, 0
#     for i in range(len(data)):
#         length = len(data[i])
#         elemt, st = [], data[i].strip()
#         for j in st:
#             if j not in amino_acids:
#                 sign = 1
#                 break
#             index = amino_acids.index(j)
#             elemt.append(index)
#             sign = 0
#
#         if length <= max_len and sign == 0:
#             temp.append(elemt)
#             seq_length.append(len(temp[b]))
#             b += 1
#             elemt += [0] * (max_len - length)
#             data_e.append(elemt)
#             label_e.append(label[i])
#     return np.array(data_e), np.array(label_e), np.array(seq_length)
#
# def getSequenceData(first_dir, file_name):
#     data, label = [], []
#     path = "{}/{}.txt".format(first_dir, file_name)
#
#     with open(path, 'r') as f:
#         for each in f:
#             each = each.strip()
#             if each:
#                 if each[0] == '>':
#                     try:
#                         numeric_label = np.array(list(map(int, each[1:])), dtype=int)
#                         label.append(numeric_label)
#                     except ValueError:
#                         print("Invalid label encountered and skipped:", each[1:])
#                 else:
#                     data.append(each)
#
#     print("Total sequences read:", len(data))
#     print("Total labels read:", len(label))
#     return data, label
#
# def staticTrainAndTest(y_train, y_test):
#     data_size_tr = np.zeros(len(filenames))
#     data_size_te = np.zeros(len(filenames))
#
#     for i in range(len(y_train)):
#         for j in range(len(y_train[i])):
#             if y_train[i][j] > 0:
#                 data_size_tr[j] += 1
#
#     for i in range(len(y_test)):
#         for j in range(len(y_test[i])):
#             if y_test[i][j] > 0:
#                 data_size_te[j] += 1
#
#     return data_size_tr
#
# # 定义粒子类
# class Particle:
#     def __init__(self, dim, bounds, batch_size_candidates):
#         self.position = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
#         self.velocity = np.random.uniform(-1, 1, dim)
#         self.best_position = self.position.copy()
#         self.best_score = float('inf')
#         self.batch_size_candidates = batch_size_candidates
#
#     def update_velocity(self, global_best_position, w=0.7, c1=1.0, c2=1.0, vmax=5):
#         r1 = np.random.rand(len(self.position))
#         r2 = np.random.rand(len(self.position))
#         self.velocity = (w * self.velocity +
#                          c1 * r1 * (self.best_position - self.position) +
#                          c2 * r2 * (global_best_position - self.position))
#         # 限制速度
#         self.velocity = np.clip(self.velocity, -vmax, vmax)
#
#     def update_position(self, bounds):
#         self.position += self.velocity
#         self.position = np.clip(self.position, bounds[:, 0], bounds[:, 1])
#         # 映射 batch_size
#         self.position[1] = self.map_batch_size(self.position[1])
#
#     def map_batch_size(self, value):
#         # 将连续值映射到最接近的候选 batch_size
#         return min(self.batch_size_candidates, key=lambda x: abs(x - value))
#
#
#
#
# class PSO:
#     def __init__(self, model_class, dataset_train, dataset_val,dataset_test, num_particles, max_iter, bounds, data, batch_size_candidates, fitness_epochs=10):
#         self.model_class = model_class
#         self.dataset_train = dataset_train
#         self.dataset_val = dataset_val
#         self.dataset_test = dataset_test
#         self.num_particles = num_particles
#         self.max_iter = max_iter
#         self.bounds = bounds
#         self.data = data
#         self.batch_size_candidates = batch_size_candidates
#         self.fitness_epochs = fitness_epochs
#         self.particles = [Particle(len(bounds), bounds, self.batch_size_candidates) for _ in range(self.num_particles)]
#         self.global_best_position = None
#         self.global_best_score = float('inf')
#
#     def fitness(self, params):
#         lr, batch_size, embedding_size, dropout = params
#         batch_size = int(batch_size)
#         embedding_size = int(embedding_size)
#         num_heads = self.data['num_heads']
#         if embedding_size % num_heads != 0:
#             embedding_size = (embedding_size // num_heads) * num_heads
#
#         model = self.model_class(vocab_size=50, embedding_size=embedding_size, output_size=21, dropout=dropout,
#                                  fan_epoch=self.data['fan_epochs'], num_heads=num_heads)
#         model.to(DEVICE)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         criterion = CombinedLDAMFocalDiceLoss()
#
#         train_loader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True,
#                                   drop_last=True)
#
#         # 训练模型
#         Train = DataTrain(model, optimizer, criterion, device=DEVICE)
#         Train.train_step(train_loader, epochs=self.fitness_epochs, plot_picture=False)
#
#         # 在验证集上评估模型
#         val_loader = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True)
#         val_score = evaluate(model, val_loader, device=DEVICE)
#
#         # 定义指标权重（根据任务需求调整）
#         weights = {
#             'aiming': 0.4,
#             'coverage': 0.2,
#             'accuracy': 0.2,
#             'absolute_true': 0.1,
#             'absolute_false': -0.1  # 由于 absolute_false 是损失，需要取反或调整权重
#         }
#
#         # 计算综合适应度值
#         fitness_value = -(
#                 weights['aiming'] * val_score['aiming'] +
#                 weights['coverage'] * val_score['coverage'] +
#                 weights['accuracy'] * val_score['accuracy'] +
#                 weights['absolute_true'] * val_score['absolute_true'] -
#                 weights['absolute_false'] * val_score['absolute_false']  # 如果 absolute_false 是损失，取负
#         )
#         return fitness_value
#
#     def evaluate_on_validation_set(self, best_params):
#         lr, batch_size, embedding_size, dropout = best_params
#         batch_size = int(batch_size)
#         embedding_size = int(embedding_size)
#         num_heads = self.data['num_heads']
#         if embedding_size % num_heads != 0:
#             embedding_size = (embedding_size // num_heads) * num_heads
#
#         # 打印当前超参数
#         print(f"当前评估的超参数：")
#         print(f"学习率：{lr:.6f}, 批量大小：{batch_size}, 嵌入维度：{embedding_size}, Dropout：{dropout:.2f}")
#
#         # 构建模型
#         model = self.model_class(vocab_size=50, embedding_size=embedding_size, output_size=21, dropout=dropout,
#                                  fan_epoch=self.data['fan_epochs'], num_heads=num_heads)
#         model.to(DEVICE)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         criterion = CombinedLDAMFocalDiceLoss()
#
#         # 重新创建数据加载器
#         train_loader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
#         val_loader = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=True, pin_memory=True,
#                                   drop_last=True)
#         test_loader = DataLoader(self.dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True)
#
#         # 训练模型
#         Train = DataTrain(model, optimizer, criterion, device=DEVICE)
#         Train.train_step(train_loader, epochs=self.fitness_epochs, plot_picture=False)
#
#         # 在验证集上评估模型
#         val_score = evaluate(model, val_loader, device=DEVICE)
#
#         # 打印验证集上的结果
#         print("在验证集上的评估结果：")
#         print(f'aiming: {val_score["aiming"]:.3f}')
#         print(f'coverage: {val_score["coverage"]:.3f}')
#         print(f'accuracy: {val_score["accuracy"]:.3f}')
#         print(f'absolute_true: {val_score["absolute_true"]:.3f}')
#         print(f'absolute_false: {val_score["absolute_false"]:.3f}')
#         print("-" * 50)
#
#         return val_score
#
#
#
#     def optimize(self):  # 添加验证集的 DataLoader
#         for iter_num in range(self.max_iter):
#             print(f"PSO优化 - 迭代次数：{iter_num + 1}/{self.max_iter}")
#             for particle in self.particles:
#                 score = self.fitness(particle.position)
#                 if score < particle.best_score:
#                     particle.best_score = score
#                     particle.best_position = particle.position.copy()
#                 if score < self.global_best_score:
#                     self.global_best_score = score
#                     self.global_best_position = particle.position.copy()
#             for particle in self.particles:
#                 particle.update_velocity(self.global_best_position)
#                 particle.update_position(self.bounds)
#
#             # 打印当前最优超参数
#             best_lr, best_batch_size, best_embedding_size, best_dropout = self.global_best_position
#             print(
#                 f"当前最优超参数：学习率={best_lr:.6f}, 批量大小={int(best_batch_size)}, 嵌入维度={int(best_embedding_size)}, Dropout={best_dropout:.2f}")
#
#             # 在每次迭代后，评估当前全局最优模型在验证集上的性能
#             print(f"\n第 {iter_num + 1} 次迭代后，在验证集上的结果：")
#             self.evaluate_on_validation_set(self.global_best_position)  # 使用验证集进行评估
#
#         print("PSO优化完成")
#         print(f"最佳适应度值：{-self.global_best_score}")
#
#         # 打印最终最优的超参数
#         final_best_lr, final_best_batch_size, final_best_embedding_size, final_best_dropout = self.global_best_position
#         print(
#             f"\n最终最优超参数：学习率={final_best_lr:.6f}, 批量大小={int(final_best_batch_size)}, 嵌入维度={int(final_best_embedding_size)}, Dropout={final_best_dropout:.2f}")
#
#         return self.global_best_position
#
# def main_with_pso(num, data):
#     first_dir = 'dataset'
#     max_length = 50
#
#     train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train_original')
#     test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test_original')
#
#     y_train = np.array(train_sequence_label)
#     y_test = np.array(test_sequence_label)
#
#     x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
#     x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
#
#
#
#
#
#
#     # 特征嵌入
#     train_features_aai = AAI_embedding(train_sequence_data, max_len=max_length)
#     test_features_aai = AAI_embedding(test_sequence_data, max_len=max_length)
#     train_features_paac = PAAC_embedding(train_sequence_data, max_len=max_length)
#     test_features_paac = PAAC_embedding(test_sequence_data, max_len=max_length)
#     train_features_pc6 = PC6_embedding(train_sequence_data, max_len=max_length)
#     test_features_pc6 = PC6_embedding(test_sequence_data, max_len=max_length)
#     train_features_blosum62 = BLOSUM62_embedding(train_sequence_data, max_len=max_length)
#     test_features_blosum62 = BLOSUM62_embedding(test_sequence_data, max_len=max_length)
#     train_features_aac = AAC_embedding(train_sequence_data)
#     test_features_aac = AAC_embedding(test_sequence_data)
#
#     # 将特征转换为张量
#     # train_features = {
#     #     'aai': torch.from_numpy(train_features_aai).float() if isinstance(train_features_aai, np.ndarray) else train_features_aai.float(),
#     #     'paac': torch.from_numpy(train_features_paac).float() if isinstance(train_features_paac, np.ndarray) else train_features_paac.float(),
#     #     'pc6': torch.from_numpy(train_features_pc6).float() if isinstance(train_features_pc6, np.ndarray) else train_features_pc6.float(),
#     #     'blosum62': torch.from_numpy(train_features_blosum62).float() if isinstance(train_features_blosum62, np.ndarray) else train_features_blosum62.float(),
#     #     'aac': torch.from_numpy(train_features_aac).float() if isinstance(train_features_aac, np.ndarray) else train_features_aac.float()
#     # }
#     # test_features = {
#     #     'aai': torch.from_numpy(test_features_aai).float() if isinstance(test_features_aai, np.ndarray) else test_features_aai.float(),
#     #     'paac': torch.from_numpy(test_features_paac).float() if isinstance(test_features_paac, np.ndarray) else test_features_paac.float(),
#     #     'pc6': torch.from_numpy(test_features_pc6).float() if isinstance(test_features_pc6, np.ndarray) else test_features_pc6.float(),
#     #     'blosum62': torch.from_numpy(test_features_blosum62).float() if isinstance(test_features_blosum62, np.ndarray) else test_features_blosum62.float(),
#     #     'aac': torch.from_numpy(test_features_aac).float() if isinstance(test_features_aac, np.ndarray) else test_features_aac.float()
#     # }
#
#     train_features = {'aai': train_features_aai, 'paac': train_features_paac, 'pc6': train_features_pc6, 'blosum62': train_features_blosum62,'aac':train_features_aac}
#     test_features = {'aai': test_features_aai, 'paac': test_features_paac, 'pc6': test_features_pc6, 'blosum62': test_features_blosum62,'aac':test_features_aac}
#
#
#     x_train = torch.LongTensor(x_train)
#     x_test = torch.LongTensor(x_test)
#     train_length = torch.LongTensor(train_length)
#     y_test = torch.Tensor(y_test)
#     y_train = torch.Tensor(y_train)
#     test_length = torch.LongTensor(test_length)
#
#     # 按 8:2 划分训练集，创建新的训练集和验证集
#     x_train_new, x_val, y_train_new, y_val, train_length_new, val_length = train_test_split(
#         x_train, y_train, train_length, test_size=0.2, random_state=42)
#
#     # 相应划分特征
#     train_features_new = {key: value[:len(x_train_new)] for key, value in train_features.items()}
#     val_features = {key: value[len(x_train_new):] for key, value in train_features.items()}
#
#     # 创建新的训练集和验证集
#     dataset_train = list(zip(x_train_new, y_train_new, train_length_new, train_features_new['aai'],
#                              train_features_new['paac'], train_features_new['pc6'], train_features_new['blosum62'],
#                              train_features_new['aac']))
#     dataset_val = list(zip(x_val, y_val, val_length, val_features['aai'],
#                            val_features['paac'], val_features['pc6'], val_features['blosum62'], val_features['aac']))
#
#     # dataset_train = list(zip(x_train, y_train, train_length, train_features['aai'], train_features['paac'],
#     #                          train_features['pc6'], train_features['blosum62'], train_features['aac']))
#     dataset_test = list(zip(x_test, y_test, test_length, test_features['aai'], test_features['paac'],
#                             test_features['pc6'], test_features['blosum62'], test_features['aac']))
#
#     # 定义 batch_size 的候选值
#     batch_size_candidates = [32, 64, 128, 256]
#
#     # PSO 参数设置
#     bounds = np.array([
#         [0.0001, 0.005],   # 学习率，缩小范围
#         # [min(batch_size_candidates), max(batch_size_candidates)],  # batch_size 范围
#         [192,512],
#         [200, 512],        # embedding_size，集中在原始值附近
#         [0.4, 0.7]         # dropout，调整范围
#     ])
#     pso = PSO(model_class=ETFC, dataset_train=dataset_train, dataset_val=dataset_val,dataset_test=dataset_test,
#               num_particles=10, max_iter=10, bounds=bounds, data=data, batch_size_candidates=batch_size_candidates,
#               fitness_epochs=10)
#
#     # 优化并获取最佳参数
#     best_params = pso.optimize()
#
#     # 使用最佳参数进行正式训练
#     best_lr, best_batch_size, best_embedding_size, best_dropout = best_params
#     best_lr = float(best_lr)
#     best_batch_size = int(best_batch_size)
#     best_embedding_size = int(best_embedding_size)
#     best_dropout = float(best_dropout)
#
#     # 打印PSO选择的最优超参数
#     print("\nPSO优化选择的最优超参数：")
#     print(f"学习率（learning_rate）：{best_lr}")
#     print(f"批量大小（batch_size）：{best_batch_size}")
#     print(f"嵌入维度（embedding_size）：{best_embedding_size}")
#     print(f"Dropout率（dropout）：{best_dropout}\n")
#
#     num_heads = data['num_heads']
#     if best_embedding_size % num_heads != 0:
#         best_embedding_size = (best_embedding_size // num_heads) * num_heads
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'data', 'tea_data' + str(num) + '.h5')
#     torch.save(dataset_test, each_model)
#
#     model = ETFC(vocab_size=50, embedding_size=best_embedding_size, output_size=21, dropout=best_dropout,
#                  fan_epoch=data['fan_epochs'], num_heads=num_heads)
#     model.to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
#     lr_scheduler = CosineScheduler(10000, base_lr=best_lr, warmup_steps=500)
#     criterion = CombinedLDAMFocalDiceLoss()
#
#     train_loader = DataLoader(dataset_train, batch_size=best_batch_size, shuffle=True, pin_memory=True, drop_last=True)
#     test_loader = DataLoader(dataset_test, batch_size=best_batch_size, shuffle=False, pin_memory=True)
#
#     # 使用完整的 epoch 数进行正式训练
#     Train = DataTrain(model, optimizer, criterion, lr_scheduler=lr_scheduler, device=DEVICE)
#
#     a = time.time()
#     Train.train_step(train_loader, epochs=data['epochs'], plot_picture=True)
#     b = time.time()
#     test_score = evaluate(model, test_loader, device=DEVICE)
#     runtime = b - a
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
#     torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
#
#     # 打印和保存结果
#     print(f"runtime:{runtime:.3f}s")
#     print("测试集：")
#     print(f'aiming: {test_score["aiming"]:.3f}')
#     print(f'coverage: {test_score["coverage"]:.3f}')
#     print(f'accuracy: {test_score["accuracy"]:.3f}')
#     print(f'absolute_true: {test_score["absolute_true"]:.3f}')
#     print(f'absolute_false: {test_score["absolute_false"]:.3f}')
#
#
#     # ...（保存模型和结果的代码）
#
#     title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
#     model_name = "tea-CELoss"
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     content = [[model_name, '%.3f' % test_score["aiming"], '%.3f' % test_score["coverage"], '%.3f' % test_score["accuracy"],
#                 '%.3f' % test_score["absolute_true"], '%.3f' % test_score["absolute_false"], '%.3f' % runtime, now]]
#
#     path = "{}/{}.csv".format('result', 'teacher')
#     if os.path.exists(path):
#         data1 = pd.read_csv(path, header=None)
#         one_line = list(data1.iloc[0])
#         if one_line == title:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerows(content)
#         else:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerow(title)
#                 writer.writerows(content)
#     else:
#         with open(path, 'a+', newline='') as t:
#             writer = csv.writer(t)
#             writer.writerow(title)
#             writer.writerows(content)
#
#
# if __name__ == '__main__':
#     # PSO和模型初始化的超参数
#     clip_pos = 0.7
#     clip_neg = 0.5
#     pos_weight = 0.7
#     batch_size = 256        # 初始值，将由PSO优化
#     epochs = 100
#     learning_rate = 0.0018  # 初始值，将由PSO优化
#     embedding_size = 256    # 初始值，将由PSO优化
#     dropout = 0.6           # 初始值，将由PSO优化
#     fan_epochs = 1
#     num_heads = 8
#
#     para = {
#         'clip_pos': clip_pos,
#         'clip_neg': clip_neg,
#         'pos_weight': pos_weight,
#         'batch_size': batch_size,
#         'epochs': epochs,
#         'learning_rate': learning_rate,
#         'embedding_size': embedding_size,
#         'dropout': dropout,
#         'fan_epochs': fan_epochs,
#         'num_heads': num_heads
#     }
#
#     for i in range(10):
#         main_with_pso(i, para)








#GAN
# import datetime
# import os
# import csv
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from loss_functions import FocalDiceLoss
# from train import DataTrain, evaluate, CosineScheduler
# from data_feature import AAI_embedding, PAAC_embedding, PC6_embedding, BLOSUM62_embedding, AAC_embedding
# from model import ETFC, Generator, Discriminator
# import time
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
#              'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']
#
# def PadEncode(data, label, max_len):
#     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
#     data_e, label_e, seq_length, temp = [], [], [], []
#     sign, b = 0, 0
#     for i in range(len(data)):
#         length = len(data[i])
#         elemt, st = [], data[i].strip()
#         for j in st:
#             if j not in amino_acids:
#                 sign = 1
#                 break
#             index = amino_acids.index(j)
#             elemt.append(index)
#             sign = 0
#
#         if length <= max_len and sign == 0:
#             temp.append(elemt)
#             seq_length.append(len(temp[b]))
#             b += 1
#             elemt += [0] * (max_len - length)
#             data_e.append(elemt)
#             label_e.append(label[i])
#     return np.array(data_e), np.array(label_e), np.array(seq_length)
#
# def getSequenceData(first_dir, file_name):
#     data, label = [], []
#     path = "{}/{}.txt".format(first_dir, file_name)
#
#     with open(path, 'r') as f:
#         for each in f:
#             each = each.strip()
#             if each:
#                 if each[0] == '>':
#                     try:
#                         numeric_label = np.array(list(map(int, each[1:])), dtype=int)
#                         label.append(numeric_label)
#                     except ValueError:
#                         print("Invalid label encountered and skipped:", each[1:])
#                 else:
#                     data.append(each)
#
#     print("Total sequences read:", len(data))
#     print("Total labels read:", len(label))
#     return data, label
#
# def main(num, data):
#     first_dir = 'dataset'
#     max_length = 50
#
#     train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train_original')
#     test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test_original')
#
#     y_train = np.array(train_sequence_label)
#     y_test = np.array(test_sequence_label)
#
#     x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
#     x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
#
#     train_features_aai = AAI_embedding(train_sequence_data, max_len=max_length)
#     test_features_aai = AAI_embedding(test_sequence_data, max_len=max_length)
#     train_features_paac = PAAC_embedding(train_sequence_data, max_len=max_length)
#     test_features_paac = PAAC_embedding(test_sequence_data, max_len=max_length)
#     train_features_pc6 = PC6_embedding(train_sequence_data, max_len=max_length)
#     test_features_pc6 = PC6_embedding(test_sequence_data, max_len=max_length)
#     train_features_blosum62 = BLOSUM62_embedding(train_sequence_data, max_len=max_length)
#     test_features_blosum62 = BLOSUM62_embedding(test_sequence_data, max_len=max_length)
#     train_features_aac = AAC_embedding(train_sequence_data)
#     test_features_aac = AAC_embedding(test_sequence_data)
#
#     train_features = {'aai': train_features_aai, 'paac': train_features_paac, 'pc6': train_features_pc6, 'blosum62': train_features_blosum62, 'aac': train_features_aac}
#     test_features = {'aai': test_features_aai, 'paac': test_features_paac, 'pc6': test_features_pc6, 'blosum62': test_features_blosum62, 'aac': test_features_aac}
#
#     x_train = torch.LongTensor(x_train)
#     x_test = torch.LongTensor(x_test)
#     train_length = torch.LongTensor(train_length)
#     y_test = torch.Tensor(y_test)
#     y_train = torch.Tensor(y_train)
#     test_length = torch.LongTensor(test_length)
#
#     dataset_train = list(zip(x_train, y_train, train_length, train_features['aai'], train_features['paac'], train_features['pc6'], train_features['blosum62'], train_features['aac']))
#     dataset_test = list(zip(x_test, y_test, test_length, test_features['aai'], test_features['paac'], test_features['pc6'], test_features['blosum62'], test_features['aac']))
#     dataset_train = DataLoader(dataset_train, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#     dataset_test = DataLoader(dataset_test, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'data', 'tea_data' + str(num) + '.h5')
#     torch.save(dataset_test, each_model)
#
#     vocab_size = 50
#     output_size = 21
#
#     model = ETFC(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'], data['num_heads'])
#     rate_learning = data['learning_rate']
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
#     lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
#     #criterion = FocalDiceLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight'])
#     criterion = CombinedLDAMFocalDiceLoss()
#
#     # 初始化生成器和判别器
#     noise_dim = 100
#     seq_length = x_train.shape[1]
#     generator = Generator(noise_dim, seq_length, vocab_size).to(DEVICE)
#     discriminator = Discriminator(seq_length, vocab_size).to(DEVICE)
#
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=rate_learning)
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=rate_learning)
#     adversarial_loss = torch.nn.BCELoss()
#
#     # 使用修改后的 DataTrain 类
#     Train = DataTrain(model, optimizer, criterion, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, lr_scheduler, device=DEVICE)
#
#     a = time.time()
#     Train.train_step(dataset_train, epochs=data['epochs'], plot_picture=True)
#     b = time.time()
#     test_score = evaluate(model, dataset_test, device=DEVICE)
#     runtime = b - a
#
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
#     torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
#
#     print(f"runtime:{runtime:.3f}s")
#     print("测试集：")
#     print(f'aiming: {test_score["aiming"]:.3f}')
#     print(f'coverage: {test_score["coverage"]:.3f}')
#     print(f'accuracy: {test_score["accuracy"]:.3f}')
#     print(f'absolute_true: {test_score["absolute_true"]:.3f}')
#     print(f'absolute_false: {test_score["absolute_false"]:.3f}')
#
#     title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
#     model_name = "tea-CELoss"
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     content = [[model_name, '%.3f' % test_score["aiming"], '%.3f' % test_score["coverage"], '%.3f' % test_score["accuracy"],
#                 '%.3f' % test_score["absolute_true"], '%.3f' % test_score["absolute_false"], '%.3f' % runtime, now]]
#
#     path = "{}/{}.csv".format('result', 'teacher')
#     if os.path.exists(path):
#         data1 = pd.read_csv(path, header=None)
#         one_line = list(data1.iloc[0])
#         if one_line == title:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerows(content)
#         else:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerow(title)
#                 writer.writerows(content)
#     else:
#         with open(path, 'a+', newline='') as t:
#             writer = csv.writer(t)
#             writer.writerow(title)
#             writer.writerows(content)
#
# if __name__ == '__main__':
#     clip_pos = 0.7
#     clip_neg = 0.5
#     pos_weight = 0.7
#
#     batch_size = 256
#     epochs = 3
#     learning_rate = 0.0018
#     embedding_size = 256
#     dropout = 0.6
#     fan_epochs = 1
#     num_heads = 8
#
#     para = {'clip_pos': clip_pos, 'clip_neg': clip_neg, 'pos_weight': pos_weight, 'batch_size': batch_size,
#             'epochs': epochs, 'learning_rate': learning_rate, 'embedding_size': embedding_size, 'dropout': dropout,
#             'fan_epochs': fan_epochs, 'num_heads': num_heads}
#     for i in range(1):
#         main(i, para)




#用BAS（Beetle Antennae Search）进行超参优化，还有CNN,LSTM的超参也一起优化了，没什么用

# import datetime
# import os
# import csv
# import pandas as pd
# import numpy as np
# import torch
# import random  # 添加 random 模块
# from torch.utils.data import DataLoader
# from train import DataTrain, evaluate, CosineScheduler
# from data_feature import AAI_embedding, PAAC_embedding, PC6_embedding, BLOSUM62_embedding, AAC_embedding
# from model import ETFC
# import time
# from sklearn.model_selection import train_test_split
#
# # 设定随机种子
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     # 确保CuDNN的确定性
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# # 调用设置随机种子函数
# set_seed(42)
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
#              'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']
#
# class BeetleAntennaeSearch:
#     def __init__(self, func, dim, lower_bound, upper_bound, max_iter=20, init_position=None):
#         self.func = func
#         self.dim = dim
#         self.lower_bound = np.array(lower_bound)
#         self.upper_bound = np.array(upper_bound)
#         self.max_iter = max_iter
#         if init_position is None:
#             self.position = np.random.uniform(self.lower_bound, self.upper_bound)
#         else:
#             self.position = np.array(init_position)
#         self.best_position = self.position.copy()
#         self.best_score = self.func(self.position)
#
#     def optimize(self):
#         for i in range(self.max_iter):
#             # 随机方向
#             dir = np.random.randn(self.dim)
#             dir = dir / np.linalg.norm(dir)
#             # 触角长度
#             delta = 0.8 / (i + 1)
#             # 左右触角位置
#             left = self.position + delta * dir
#             right = self.position - delta * dir
#             # 保持在范围内
#             left = np.clip(left, self.lower_bound, self.upper_bound)
#             right = np.clip(right, self.lower_bound, self.upper_bound)
#             # 在触角位置评估函数
#             left_score = self.func(left)
#             right_score = self.func(right)
#             # 估计梯度
#             grad = (left_score - right_score) * dir / (2 * delta)
#             # 步长
#             step = 0.2 / (i + 1)
#             # 更新位置
#             self.position = self.position - step * grad
#             # 保持在范围内
#             self.position = np.clip(self.position, self.lower_bound, self.upper_bound)
#             # 评估新位置
#             score = self.func(self.position)
#             if score < self.best_score:
#                 self.best_score = score
#                 self.best_position = self.position.copy()
#             print(f"Iteration {i+1}, Best Score: {self.best_score}")
#         return self.best_position, self.best_score
#
# def PadEncode(data, label, max_len):
#     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
#     data_e, label_e, seq_length, temp = [], [], [], []
#     sign, b = 0, 0
#     for i in range(len(data)):
#         length = len(data[i])
#         elemt, st = [], data[i].strip()
#         for j in st:
#             if j not in amino_acids:
#                 sign = 1
#                 break
#             index = amino_acids.index(j)
#             elemt.append(index)
#             sign = 0
#
#         if length <= max_len and sign == 0:
#             temp.append(elemt)
#             seq_length.append(len(temp[b]))
#             b += 1
#             elemt += [0] * (max_len - length)
#             data_e.append(elemt)
#             label_e.append(label[i])
#     return np.array(data_e), np.array(label_e), np.array(seq_length)
#
# def getSequenceData(first_dir, file_name):
#     data, label = [], []
#     path = "{}/{}.txt".format(first_dir, file_name)
#
#     with open(path, 'r') as f:
#         for each in f:
#             each = each.strip()
#             if each:
#                 if each[0] == '>':
#                     try:
#                         numeric_label = np.array(list(map(int, each[1:])), dtype=int)
#                         label.append(numeric_label)
#                     except ValueError:
#                         print("Invalid label encountered and skipped:", each[1:])
#                 else:
#                     data.append(each)
#
#     print("Total sequences read:", len(data))
#     print("Total labels read:", len(label))
#     return data, label
#
# def main(num, data, use_full_training_data=False):
#     first_dir = 'dataset'
#     max_length = 50
#
#     train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train_original')
#     test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test_original')
#
#     y_train = np.array(train_sequence_label)
#     y_test = np.array(test_sequence_label)
#
#     if not use_full_training_data:
#         # 划分训练集和验证集
#         train_data, val_data, train_labels, val_labels = train_test_split(
#             train_sequence_data, y_train, test_size=0.2, random_state=42)
#     else:
#         train_data = train_sequence_data
#         train_labels = y_train
#         val_data = None
#         val_labels = None
#
#     # 处理训练数据
#     x_train, y_train, train_length = PadEncode(train_data, train_labels, max_length)
#     train_features_aai = AAI_embedding(train_data, max_len=max_length)
#     train_features_paac = PAAC_embedding(train_data, max_len=max_length)
#     train_features_pc6 = PC6_embedding(train_data, max_len=max_length)
#     train_features_blosum62 = BLOSUM62_embedding(train_data, max_len=max_length)
#     train_features_aac = AAC_embedding(train_data)
#
#     train_features = {'aai': train_features_aai, 'paac': train_features_paac, 'pc6': train_features_pc6, 'blosum62': train_features_blosum62,'aac':train_features_aac}
#
#     x_train = torch.LongTensor(x_train)
#     y_train = torch.Tensor(y_train)
#     train_length = torch.LongTensor(train_length)
#
#     dataset_train = list(zip(x_train, y_train, train_length, train_features['aai'], train_features['paac'], train_features['pc6'], train_features['blosum62'],train_features['aac']))
#     dataset_train = DataLoader(dataset_train, batch_size=int(data['batch_size']), shuffle=True, pin_memory=True)
#
#     if not use_full_training_data:
#         # 处理验证数据
#         x_val, y_val, val_length = PadEncode(val_data, val_labels, max_length)
#         val_features_aai = AAI_embedding(val_data, max_len=max_length)
#         val_features_paac = PAAC_embedding(val_data, max_len=max_length)
#         val_features_pc6 = PC6_embedding(val_data, max_len=max_length)
#         val_features_blosum62 = BLOSUM62_embedding(val_data, max_len=max_length)
#         val_features_aac = AAC_embedding(val_data)
#
#         val_features = {'aai': val_features_aai, 'paac': val_features_paac, 'pc6': val_features_pc6, 'blosum62': val_features_blosum62,'aac':val_features_aac}
#
#         x_val = torch.LongTensor(x_val)
#         y_val = torch.Tensor(y_val)
#         val_length = torch.LongTensor(val_length)
#
#         dataset_val = list(zip(x_val, y_val, val_length, val_features['aai'], val_features['paac'], val_features['pc6'], val_features['blosum62'],val_features['aac']))
#         dataset_val = DataLoader(dataset_val, batch_size=int(data['batch_size']), shuffle=False, pin_memory=True)
#     else:
#         dataset_val = None
#
#     # 处理测试数据
#     x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
#     test_features_aai = AAI_embedding(test_sequence_data, max_len=max_length)
#     test_features_paac = PAAC_embedding(test_sequence_data, max_len=max_length)
#     test_features_pc6 = PC6_embedding(test_sequence_data, max_len=max_length)
#     test_features_blosum62 = BLOSUM62_embedding(test_sequence_data, max_len=max_length)
#     test_features_aac = AAC_embedding(test_sequence_data)
#
#     test_features = {'aai': test_features_aai, 'paac': test_features_paac, 'pc6': test_features_pc6, 'blosum62': test_features_blosum62,'aac':test_features_aac}
#
#     x_test = torch.LongTensor(x_test)
#     y_test = torch.Tensor(y_test)
#     test_length = torch.LongTensor(test_length)
#
#     dataset_test = list(zip(x_test, y_test, test_length, test_features['aai'], test_features['paac'], test_features['pc6'], test_features['blosum62'],test_features['aac']))
#     dataset_test = DataLoader(dataset_test, batch_size=int(data['batch_size']), shuffle=False, pin_memory=True)
#
#     vocab_size = 50
#     output_size = 21
#
#     # 将卷积核大小和池化大小转换为整数
#     conv_channels = int(data['conv_channels'])
#     kernel_sizes = [int(k) for k in data['kernel_sizes']]
#     max_pool = int(data['max_pool'])
#
#     # 初始化模型
#     model = ETFC(
#         vocab_size=vocab_size,
#         embedding_size=int(data['embedding_size']),
#         output_size=output_size,
#         dropout=data['dropout'],
#         fan_epoch=int(data['fan_epochs']),
#         num_heads=int(data['num_heads']),
#         conv_channels=conv_channels,
#         kernel_sizes=kernel_sizes,
#         max_pool=max_pool,
#         # 添加LSTM超参数
#         lstm_hidden_size=int(data['lstm_hidden_size']),
#         lstm_num_layers=int(data['lstm_num_layers']),
#         lstm_dropout=data['lstm_dropout']
#     )
#
#     rate_learning = data['learning_rate']
#     optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
#     lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
#     criterion = CombinedLDAMFocalDiceLoss()
#
#     Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)
#
#     a = time.time()
#     Train.train_step(dataset_train, epochs=int(data['epochs']), plot_picture=False)
#     b = time.time()
#     runtime = b - a
#
#     if not use_full_training_data:
#         # 在验证集上评估
#         val_score = evaluate(model, dataset_val, device=DEVICE)
#         print(f"runtime:{runtime:.3f}s")
#         print("验证集：")
#         print(f'aiming: {val_score["aiming"]:.3f}')
#         print(f'coverage: {val_score["coverage"]:.3f}')
#         print(f'accuracy: {val_score["accuracy"]:.3f}')
#         print(f'absolute_true: {val_score["absolute_true"]:.3f}')
#         print(f'absolute_false: {val_score["absolute_false"]:.3f}')
#         return val_score["loss"]  # 返回验证集损失以供 BAS 优化
#     else:
#         PATH = os.getcwd()
#         each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
#         torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
#         # 在测试集上评估
#         test_score = evaluate(model, dataset_test, device=DEVICE)
#         print(f"runtime:{runtime:.3f}s")
#         print("测试集：")
#         print(f'aiming: {test_score["aiming"]:.3f}')
#         print(f'coverage: {test_score["coverage"]:.3f}')
#         print(f'accuracy: {test_score["accuracy"]:.3f}')
#         print(f'absolute_true: {test_score["absolute_true"]:.3f}')
#         print(f'absolute_false: {test_score["absolute_false"]:.3f}')
#         title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time', 'LSTM_Hidden_Size', 'LSTM_Num_Layers', 'LSTM_Dropout']
#         model_name = "tea-CELoss"
#         now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         content = [[model_name,
#                     '%.3f' % test_score["aiming"],
#                     '%.3f' % test_score["coverage"],
#                     '%.3f' % test_score["accuracy"],
#                     '%.3f' % test_score["absolute_true"],
#                     '%.3f' % test_score["absolute_false"],
#                     '%.3f' % runtime,
#                     now,
#                     int(data['lstm_hidden_size']),
#                     int(data['lstm_num_layers']),
#                     '%.3f' % data['lstm_dropout']
#                    ]]
#
#         path = "{}/{}.csv".format('result', 'teacher')
#         if os.path.exists(path):
#             data1 = pd.read_csv(path, header=None)
#             one_line = list(data1.iloc[0])
#             if one_line == title:
#                 with open(path, 'a+', newline='') as t:
#                     writer = csv.writer(t)
#                     writer.writerows(content)
#             else:
#                 with open(path, 'a+', newline='') as t:
#                     writer = csv.writer(t)
#                     writer.writerow(title)
#                     writer.writerows(content)
#         else:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerow(title)
#                 writer.writerows(content)
#
#         return test_score  # 返回测试集结果
#
# def bas_optimize(num):
#     # 定义需要优化的超参数，包括LSTM的超参数
#     lower_bounds = [1e-5, 0.1, 32, 16, 2, 64, 1, 0.0]  # 添加LSTM超参数的下界
#     upper_bounds = [1e-2, 0.7, 256, 128, 5, 512, 3, 0.5]  # 添加LSTM超参数的上界
#
#     def objective_function(params):
#         learning_rate, dropout, batch_size, conv_channels, max_pool, lstm_hidden_size, lstm_num_layers, lstm_dropout = params
#         batch_size = int(batch_size)
#         conv_channels = int(conv_channels)
#         max_pool = int(max_pool)
#         lstm_num_layers = int(lstm_num_layers)
#         lstm_hidden_size = int(lstm_hidden_size)
#         lstm_dropout = float(lstm_dropout)
#         data = {
#             'batch_size': batch_size,
#             'epochs': 5,  # 为了加快优化过程，可先设置较小的 epochs
#             'learning_rate': learning_rate,
#             'embedding_size': 256,
#             'dropout': dropout,
#             'fan_epochs': 1,
#             'num_heads': 8,
#             'conv_channels': conv_channels,
#             'kernel_sizes': [2, 4, 6, 8],  # 可以固定或设为超参数
#             'max_pool': max_pool,
#             'lstm_hidden_size': lstm_hidden_size,
#             'lstm_num_layers': lstm_num_layers,
#             'lstm_dropout': lstm_dropout
#         }
#         loss = main(num, data, use_full_training_data=False)
#         return loss
#
#     # 初始化 BAS 优化器
#     bas = BeetleAntennaeSearch(
#         func=objective_function,
#         dim=8,  # 优化8个超参数
#         lower_bound=lower_bounds,
#         upper_bound=upper_bounds,
#         max_iter=100  # 可以根据需要调整迭代次数
#     )
#
#     best_params, best_score = bas.optimize()
#     print("BAS 优化完成，最佳参数：", best_params)
#     print("最佳验证集损失：", best_score)
#
#     # 使用最佳超参数在全量训练集上训练，并在测试集上评估
#     learning_rate, dropout, batch_size, conv_channels, max_pool, lstm_hidden_size, lstm_num_layers, lstm_dropout = best_params
#     best_data = {
#         'batch_size': int(batch_size),
#         'epochs': 256,  # 可以根据需要调整训练轮数
#         'learning_rate': learning_rate,
#         'embedding_size': 256,
#         'dropout': dropout,
#         'fan_epochs': 1,
#         'num_heads': 8,
#         'conv_channels': int(conv_channels),
#         'kernel_sizes': [2, 4, 6, 8],
#         'max_pool': int(max_pool),
#         'lstm_hidden_size': int(lstm_hidden_size),
#         'lstm_num_layers': int(lstm_num_layers),
#         'lstm_dropout': float(lstm_dropout)
#     }
#
#     main(num, best_data, use_full_training_data=True)
#
# if __name__ == '__main__':
#     for i in range(10):
#         bas_optimize(i)



# import datetime
# import os
# import csv
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from loss_functions import FocalDiceLoss
# from train import DataTrain, evaluate, CosineScheduler
# from data_feature import AAI_embedding, PAAC_embedding, PC6_embedding, BLOSUM62_embedding
# from model import ETFC
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
#              'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']
#
# def PadEncode(data, label, max_len):
#     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
#     data_e, label_e, seq_length, temp = [], [], [], []
#     sign, b = 0, 0
#     for i in range(len(data)):
#         length = len(data[i])
#         elemt, st = [], data[i].strip()
#         #st = augment_sequence(st)#数据增强，随机替换氨基酸序列
#         for j in st:
#             if j not in amino_acids:
#                 sign = 1
#                 break
#             index = amino_acids.index(j)
#             elemt.append(index)
#             sign = 0
#
#         if length <= max_len and sign == 0:
#             temp.append(elemt)
#             seq_length.append(len(temp[b]))
#             b += 1
#             elemt += [0] * (max_len - length)
#             data_e.append(elemt)
#             label_e.append(label[i])
#     return np.array(data_e), np.array(label_e), np.array(seq_length)
#
#
# #会打印读取到的序列和标签总数
# def getSequenceData(first_dir, file_name):
#     data, label = [], []
#     path = "{}/{}.txt".format(first_dir, file_name)
#
#     with open(path, 'r') as f:
#         for each in f:
#             each = each.strip()
#             if each:  # Check if line is not empty
#                 if each[0] == '>':  # This assumes that labels are prefixed with '>'
#                     try:
#                         numeric_label = np.array(list(map(int, each[1:])), dtype=int)  # Convert string labels to numeric vectors
#                         label.append(numeric_label)
#                     except ValueError:
#                         print("Invalid label encountered and skipped:", each[1:])
#                 else:
#                     data.append(each)
#
#     print("Total sequences read:", len(data))
#     print("Total labels read:", len(label))
#     return data, label
#
# def staticTrainAndTest(y_train, y_test):
#     data_size_tr = np.zeros(len(filenames))
#     data_size_te = np.zeros(len(filenames))
#
#     for i in range(len(y_train)):
#         for j in range(len(y_train[i])):
#             if y_train[i][j] > 0:
#                 data_size_tr[j] += 1
#
#     for i in range(len(y_test)):
#         for j in range(len(y_test[i])):
#             if y_test[i][j] > 0:
#                 data_size_te[j] += 1
#
#     return data_size_tr
#
# def main(num, data):
#     first_dir = 'dataset'
#     max_length = 50
#
#     train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train_original')
#     test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test_original')
#
#
#     y_train = np.array(train_sequence_label)
#     y_test = np.array(test_sequence_label)
#
#     x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
#     x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)
#
#
#     train_features_aai = AAI_embedding(train_sequence_data, max_len=max_length)
#     test_features_aai = AAI_embedding(test_sequence_data, max_len=max_length)
#     train_features_paac = PAAC_embedding(train_sequence_data, max_len=max_length)
#     test_features_paac = PAAC_embedding(test_sequence_data, max_len=max_length)
#     train_features_pc6 = PC6_embedding(train_sequence_data, max_len=max_length)
#     test_features_pc6 = PC6_embedding(test_sequence_data, max_len=max_length)
#     train_features_blosum62 = BLOSUM62_embedding(train_sequence_data, max_len=max_length)
#     test_features_blosum62 = BLOSUM62_embedding(test_sequence_data, max_len=max_length)
#     train_features_aac = AAC_embedding(train_sequence_data)
#     test_features_aac = AAC_embedding(test_sequence_data)
#
#
#
#
#     train_features = {'aai': train_features_aai, 'paac': train_features_paac, 'pc6': train_features_pc6, 'blosum62': train_features_blosum62,'aac':train_features_aac}
#     test_features = {'aai': test_features_aai, 'paac': test_features_paac, 'pc6': test_features_pc6, 'blosum62': test_features_blosum62,'aac':test_features_aac}
#
#     x_train = torch.LongTensor(x_train)
#     x_test = torch.LongTensor(x_test)
#     train_length = torch.LongTensor(train_length)
#     y_test = torch.Tensor(y_test)
#     y_train = torch.Tensor(y_train)
#     test_length = torch.LongTensor(test_length)
#
#
#     dataset_train = list(zip(x_train, y_train, train_length, train_features['aai'], train_features['paac'], train_features['pc6'], train_features['blosum62'],train_features['aac']))
#     dataset_test = list(zip(x_test, y_test, test_length, test_features['aai'], test_features['paac'], test_features['pc6'], test_features['blosum62'],test_features['aac']))
#     dataset_train = DataLoader(dataset_train, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#     dataset_test = DataLoader(dataset_test, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'data', 'tea_data' + str(num) + '.h5')
#     torch.save(dataset_test, each_model)
#
#     vocab_size = 50
#     output_size = 21
#
#     model = ETFC(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'], data['num_heads'])
#     rate_learning = data['learning_rate']
#     #optimizer = torch.optim.NAdam(model.parameters(), lr=rate_learning)
#     optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
#     lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
#     #criterion = LDAM_loss(max_m=0.5, class_weight="balanced")
#     #criterion = FocalDiceLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight'])
#     #criterion = AsymmetricLoss()
#     #criterion = FocalLoss()
#     #criterion = ComboLoss()
#     #criterion = nn.CrossEntropyLoss()#没效果
#     #criterion = APLLoss()
#     #criterion = PartialSelectiveLoss()
#     #criterion = FocalLDAMLoss()
#     #criterion = FocalAsymmetricLoss()
#     criterion = CombinedLDAMFocalDiceLoss()
#
#
#
#
#     Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)
#
#     a = time.time()
#     Train.train_step(dataset_train, epochs=data['epochs'], plot_picture=True)
#     b = time.time()
#     test_score = evaluate(model, dataset_test, device=DEVICE)
#     runtime = b - a
#
#     PATH = os.getcwd()
#     each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
#     torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
#
#     print(f"runtime:{runtime:.3f}s")
#     print("测试集：")
#     print(f'aiming: {test_score["aiming"]:.3f}')
#     print(f'coverage: {test_score["coverage"]:.3f}')
#     print(f'accuracy: {test_score["accuracy"]:.3f}')
#     print(f'absolute_true: {test_score["absolute_true"]:.3f}')
#     print(f'absolute_false: {test_score["absolute_false"]:.3f}')
#
#     title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
#     model_name = "tea-CELoss"
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     content = [[model_name, '%.3f' % test_score["aiming"], '%.3f' % test_score["coverage"], '%.3f' % test_score["accuracy"],
#                 '%.3f' % test_score["absolute_true"], '%.3f' % test_score["absolute_false"], '%.3f' % runtime, now]]
#
#     path = "{}/{}.csv".format('result', 'teacher')
#     if os.path.exists(path):
#         data1 = pd.read_csv(path, header=None)
#         one_line = list(data1.iloc[0])
#         if one_line == title:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerows(content)
#         else:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerow(title)
#                 writer.writerows(content)
#     else:
#         with open(path, 'a+', newline='') as t:
#             writer = csv.writer(t)
#             writer.writerow(title)
#             writer.writerows(content)
#
#
# if __name__ == '__main__':
#     clip_pos = 0.7
#     clip_neg = 0.5
#     #pos_weight = 0.3#本地
#     pos_weight = 0.7
#
#     batch_size = 256
#     #epochs = 256#本地
#     epochs = 256
#
#      # batch_size = 200#本地
#     # epochs = 200#本地
#
#     learning_rate = 0.0018
#
#
#
#     embedding_size = 256
#
#
#     dropout = 0.6
#     fan_epochs = 1
#     num_heads = 8
#
#     para = {'clip_pos': clip_pos, 'clip_neg': clip_neg, 'pos_weight': pos_weight, 'batch_size': batch_size,
#             'epochs': epochs, 'learning_rate': learning_rate, 'embedding_size': embedding_size, 'dropout': dropout,
#             'fan_epochs': fan_epochs, 'num_heads': num_heads}
#     for i in range(10):
#         main(i, para)





#转成图卷积
# import os
# import math
# import time
# import datetime
# import csv
# import pandas as pd
# import numpy as np
# import torch
# from torch_geometric.data import Data as GeoData
# from torch_geometric.loader import DataLoader as GeoDataLoader
#
# from model import GNNModel  # 修改后的模型
# from train import DataTrain, CosineScheduler, evaluate
# from evaluation import evaluate as eval_metric
#
# # 确保使用GPU
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
#              'AVP',
#              'BBP', 'BIP',
#              'CPP', 'DPPIP',
#              'QSP', 'SBP', 'THP']
#
#
# def getSequenceData(first_dir, file_name):
#     # 获取序列数据和标签
#     data, label = [], []
#     path = "{}/{}.txt".format(first_dir, file_name)
#
#     with open(path) as f:
#         for each in f:
#             each = each.strip()
#             if each[0] == '>':
#                 label.append(np.array(list(each[1:]), dtype=int))  # 将字符串标签转换为数值向量
#             else:
#                 data.append(each)
#
#     return data, label
#
#
# def sequence_to_graph(sequence, label, max_len=50):
#     amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
#     node_features = []
#     edge_index = []
#     for i, aa in enumerate(sequence):
#         if aa not in amino_acids:
#             # 如果遇到不在氨基酸列表中的字符，跳过该序列
#             return None
#         node_features.append(amino_acids.index(aa))
#
#     # 构建边，连接序列中的相邻氨基酸
#     edge_index = []
#     for i in range(len(node_features) - 1):
#         edge_index.append([i, i + 1])
#         edge_index.append([i + 1, i])  # 无向图
#
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#
#     # 创建节点特征张量
#     x = torch.tensor(node_features, dtype=torch.long)
#
#     # 标签
#     y = torch.tensor(label, dtype=torch.float)
#
#     data = GeoData(x=x, edge_index=edge_index, y=y)
#
#     return data
#
#
# def create_graph_dataset(sequences, labels, max_len=50):
#     graph_list = []
#     for seq, lbl in zip(sequences, labels):
#         # 如果序列长度超过最大长度，裁剪到 max_len
#         if len(seq) > max_len:
#             seq = seq[:max_len]  # 裁剪序列
#
#         # 创建图数据
#         graph = sequence_to_graph(seq, lbl, max_len)
#         if graph is not None:
#             graph_list.append(graph)
#     return graph_list
#
#
# def main(num, data):
#     first_dir = 'dataset'
#
#     max_length = 50  # 氨基酸序列的最长长度
#
#     # 获取训练集和测试集的序列数据和标签数据
#     train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train_original')
#     test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test_original')
#     print(f"训练集序列数据数量: {len(train_sequence_data)}")
#     print(f"测试集序列数据数量: {len(test_sequence_data)}")
#
#
#     # 创建图数据集
#     train_graphs = create_graph_dataset(train_sequence_data, train_sequence_label, max_length)
#     test_graphs = create_graph_dataset(test_sequence_data, test_sequence_label, max_length)
#
#
#     # 打印读取到的图数据数量
#     print(f"训练集图数据数量: {len(train_graphs)}")
#     print(f"测试集图数据数量: {len(test_graphs)}")
#     # 计算并打印训练集和测试集中的总节点数
#     train_total_nodes = sum([g.num_nodes for g in train_graphs])
#     test_total_nodes = sum([g.num_nodes for g in test_graphs])
#     print(f"训练集总节点数: {train_total_nodes}")
#     print(f"测试集总节点数: {test_total_nodes}")
#
#     # 创建PyG的DataLoader
#     dataset_train = GeoDataLoader(train_graphs, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
#     dataset_test = GeoDataLoader(test_graphs, batch_size=data['batch_size'], shuffle=False, pin_memory=True)
#
#
#
#
#     # 设置保存模型的路径，并将测试集数据集保存为名为 tea_data[num].pt 的文件
#     PATH = os.getcwd()
#     each_model_data = os.path.join(PATH, 'result', 'Model', 'data', 'tea_data' + str(num) + '.h5')
#     torch.save(dataset_test, each_model_data)
#
#     # 设置训练参数
#     vocab_size = 50
#     output_size = len(filenames)
#
#     # 初始化参数训练模型相关参数
#     model = GNNModel(vocab_size=vocab_size, embedding_size=data['embedding_size'],
#                     output_size=output_size, dropout=data['dropout'],
#                     fan_epoch=data['fan_epochs'], num_heads=data['num_heads']).to(DEVICE)
#     rate_learning = data['learning_rate']
#     optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
#     lr_scheduler = CosineScheduler(max_update=10000, base_lr=rate_learning, warmup_steps=500)
#
#     # 定义损失函数
#     criterion = CombinedLDAMFocalDiceLoss()
#
#     # 创建训练类
#     Trainer = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)
#
#     # 记录训练时间
#     a = time.time()
#     Trainer.train_step(dataset_train, epochs=data['epochs'], plot_picture=False)
#     b = time.time()
#     runtime = b - a
#
#     # 评估模型
#     test_score = evaluate(model, dataset_test, device=DEVICE)
#
#     # 保存模型参数
#     each_model = os.path.join(PATH, 'result', 'Model', 'teacher', 'tea_model' + str(num) + '.h5')
#     torch.save(model.state_dict(), each_model)
#
#     # 输出模型结果
#     print(f"runtime:{runtime:.3f}s")
#     print("测试集：")
#     print(f'aiming: {test_score["aiming"]:.3f}')
#     print(f'coverage: {test_score["coverage"]:.3f}')
#     print(f'accuracy: {test_score["accuracy"]:.3f}')
#     print(f'absolute_true: {test_score["absolute_true"]:.3f}')
#     print(f'absolute_false: {test_score["absolute_false"]:.3f}')
#
#     # 保存模型结果
#     title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
#     model_name = "tea-GNN"
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     content = [[model_name, '%.3f' % test_score["aiming"],
#                 '%.3f' % test_score["coverage"],
#                 '%.3f' % test_score["accuracy"],
#                 '%.3f' % test_score["absolute_true"],
#                 '%.3f' % test_score["absolute_false"],
#                 '%.3f' % runtime,
#                 now]]
#
#     path = "{}/{}.csv".format('result', 'teacher')
#
#     if os.path.exists(path):
#         data1 = pd.read_csv(path, header=None)
#         one_line = list(data1.iloc[0])
#         if one_line == title:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerows(content)
#         else:
#             with open(path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerow(title)
#                 writer.writerows(content)
#     else:
#         with open(path, 'a+', newline='') as t:
#             writer = csv.writer(t)
#             writer.writerow(title)
#             writer.writerows(content)
#
#
# if __name__ == '__main__':
#     clip_pos = 0.7
#     clip_neg = 0.5
#     pos_weight = 0.3
#     batch_size = 256
#     epochs = 200
#     learning_rate = 0.0018
#     embedding_size = 256
#     dropout = 0.6
#     fan_epochs = 1
#     num_heads = 8
#
#     para = {'clip_pos': clip_pos,
#             'clip_neg': clip_neg,
#             'pos_weight': pos_weight,
#             'batch_size': batch_size,
#             'epochs': epochs,
#             'learning_rate': learning_rate,
#             'embedding_size': embedding_size,
#             'dropout': dropout,
#             'fan_epochs': fan_epochs,
#             'num_heads': num_heads}
#
#     for i in range(10):
#         main(i, para)



import os
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from loss_functions import *
from train import DataTrain, evaluate, CosineScheduler
from data_feature import AAI_embedding, PAAC_embedding, PC6_embedding, BLOSUM62_embedding, AAC_embedding
from model import *
import torch_geometric
from torch_geometric.data import Data as GeometricData
from torch.utils.data import random_split
import sys
import random
import time
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']





#进化算法
# param_space = {
#     'conv1_kernel': [2, 3, 4],
#     'conv2_kernel': [3, 4, 5],
#     'conv3_kernel': [4, 5, 6],
#     'conv4_kernel': [5, 6, 7, 8],
#     'out_channels': [32, 64, 96],
#     'max_pool': [2, 3, 4, 5]
# }

param_space = {
    # CNN
    'conv1_kernel': [2, 3, 4],
    'conv2_kernel': [3, 4, 5],
    'conv3_kernel': [4, 5, 6],
    'conv4_kernel': [5, 6, 7, 8],
    'out_channels': [32, 64, 96],
    'max_pool': [2, 3, 4, 5],

    # LSTM
    'lstm_hidden': [64, 128, 256],
    'lstm_layers': [1, 2],

    # GAT
    'gnn_hidden': [64, 128],
    'gnn_heads': [4, 8]
}

# def generate_candidate():
#     return {
#         'conv1_kernel': random.choice(param_space['conv1_kernel']),
#         'conv2_kernel': random.choice(param_space['conv2_kernel']),
#         'conv3_kernel': random.choice(param_space['conv3_kernel']),
#         'conv4_kernel': random.choice(param_space['conv4_kernel']),
#         'out_channels': random.choice(param_space['out_channels']),
#         'max_pool': random.choice(param_space['max_pool'])
#     }

def generate_candidate():
    while True:
        candidate = {
            'conv1_kernel': random.choice(param_space['conv1_kernel']),
            'conv2_kernel': random.choice(param_space['conv2_kernel']),
            'conv3_kernel': random.choice(param_space['conv3_kernel']),
            'conv4_kernel': random.choice(param_space['conv4_kernel']),
            'out_channels': random.choice(param_space['out_channels']),
            'max_pool': random.choice(param_space['max_pool']),
            'lstm_hidden': random.choice(param_space['lstm_hidden']),
            'lstm_layers': random.choice(param_space['lstm_layers']),
            'gnn_hidden': random.choice(param_space['gnn_hidden']),
            'gnn_heads': random.choice(param_space['gnn_heads'])
        }
        # 限制：不允许所有卷积核大小都一样
        ks = [candidate['conv1_kernel'], candidate['conv2_kernel'],
              candidate['conv3_kernel'], candidate['conv4_kernel']]
        if len(set(ks)) > 1:  # 至少有两个不一样
            return candidate

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child

def mutate(individual, mutation_rate=0.2):
    mutant = individual.copy()
    for key in mutant:
        if random.random() < mutation_rate:
            mutant[key] = random.choice(param_space[key])
    return mutant

# def evaluate_candidate(candidate, model, val_loader):
#     try:
#         model.conv1 = nn.Conv1d(model.embedding_size, candidate['out_channels'], candidate['conv1_kernel']).to(model.device)
#         model.conv2 = nn.Conv1d(model.embedding_size, candidate['out_channels'], candidate['conv2_kernel']).to(model.device)
#         model.conv3 = nn.Conv1d(model.embedding_size, candidate['out_channels'], candidate['conv3_kernel']).to(model.device)
#         model.conv4 = nn.Conv1d(model.embedding_size, candidate['out_channels'], candidate['conv4_kernel']).to(model.device)
#         model.MaxPool1d = nn.MaxPool1d(candidate['max_pool']).to(model.device)
#
#         dummy_input = torch.randn(1, model.embedding_size, 50).to(model.device)
#         cnn_out_dim = model.TextCNN(dummy_input).shape[1]
#         model.fan = FAN_encode(model.dropout_value, cnn_out_dim).to(model.device)
#         model.full3 = nn.Linear(cnn_out_dim, 1024).to(model.device)
#
#         score = evaluate(model, val_loader, device=model.device)
#         return score['accuracy']
#     except Exception as e:
#         print("[Warning] Invalid candidate:", candidate, "Error:", e)
#         return -1

def evaluate_candidate(candidate, model, val_loader):
    try:
        # LSTM 参数设置
        model.bilstm = nn.LSTM(
            input_size=model.embedding_size,
            hidden_size=candidate['lstm_hidden'],
            num_layers=candidate['lstm_layers'],
            batch_first=True,
            bidirectional=True
        ).to(model.device)

        lstm_out_dim = candidate['lstm_hidden'] * 2

        # CNN 参数设置
        model.conv1 = nn.Conv1d(lstm_out_dim, candidate['out_channels'], candidate['conv1_kernel']).to(model.device)
        model.conv2 = nn.Conv1d(lstm_out_dim, candidate['out_channels'], candidate['conv2_kernel']).to(model.device)
        model.conv3 = nn.Conv1d(lstm_out_dim, candidate['out_channels'], candidate['conv3_kernel']).to(model.device)
        model.conv4 = nn.Conv1d(lstm_out_dim, candidate['out_channels'], candidate['conv4_kernel']).to(model.device)
        model.MaxPool1d = nn.MaxPool1d(candidate['max_pool']).to(model.device)

        dummy_input = torch.randn(1, lstm_out_dim, 50).to(model.device)
        cnn_out_dim = model.TextCNN(dummy_input).shape[1]

        # GNN 参数设置
        model.gnn = GNN(
            input_dim=256,
            hidden_dim=candidate['gnn_hidden'],
            output_dim=256,
            num_heads=candidate['gnn_heads']
        ).to(model.device)

        # FAN 和 FC 重新定义
        model.fan = FAN_encode(model.dropout_value, cnn_out_dim).to(model.device)
        model.full3 = nn.Linear(cnn_out_dim, 1024).to(model.device)

        score = evaluate(model, val_loader, device=model.device)
        return score['accuracy']

    except Exception as e:
        print("[Warning] Invalid candidate:", candidate, "Error:", e)
        return -1

# def optimize_cnn_params(model, val_loader, population_size=8, generations=5):
#     population = [generate_candidate() for _ in range(population_size)]
#
#     for generation in range(generations):
#         scored = [(ind, evaluate_candidate(ind, model, val_loader)) for ind in population]
#         scored = sorted(scored, key=lambda x: x[1], reverse=True)
#         elites = [ind for ind, _ in scored[:population_size // 2]]
#
#         next_population = elites[:]
#         while len(next_population) < population_size:
#             p1, p2 = random.sample(elites, 2)
#             child = crossover(p1, p2)
#             child = mutate(child)
#             next_population.append(child)
#
#         population = next_population
#
#     best = max(population, key=lambda ind: evaluate_candidate(ind, model, val_loader))
#     return best


def optimize_model_hyperparams(model, val_loader, population_size=8, generations=5):
    # 初始化种群
    population = [generate_candidate() for _ in range(population_size)]

    for generation in range(generations):
        print(f"[Generation {generation + 1}] Evaluating candidates...")

        # 评估每个候选体的准确率
        scored_population = []
        for individual in population:
            accuracy = evaluate_candidate(individual, model, val_loader)
            scored_population.append((individual, accuracy))

        # 按准确率排序，保留精英个体
        scored_population.sort(key=lambda x: x[1], reverse=True)
        elites = [ind for ind, _ in scored_population[:population_size // 2]]

        # 打印当前最佳个体
        print(f"Best in generation {generation + 1}: {scored_population[0]}")

        # 生成下一代
        next_generation = elites[:]
        while len(next_generation) < population_size:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            next_generation.append(child)

        population = next_generation

    # 最终评估最佳个体
    best_candidate = max(population, key=lambda ind: evaluate_candidate(ind, model, val_loader))
    print("[Optimization Complete] Best Candidate:", best_candidate)
    return best_candidate

def PadEncode(data, label, max_len):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            temp.append(elemt)
            seq_length.append(len(temp[b]))
            b += 1
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    print(f"PadEncode: Filtered sequences: {len(data_e)}, Filtered labels: {len(label_e)}")
    return np.array(data_e), np.array(label_e), np.array(seq_length)




# 会打印读取到的序列和标签总数
def getSequenceData(first_dir, file_name):
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path, 'r') as f:
        for each in f:
            each = each.strip()
            if each:  # Check if line is not empty
                if each[0] == '>':  # This assumes that labels are prefixed with '>'
                    try:
                        numeric_label = np.array(list(map(int, each[1:])),
                                                 dtype=int)  # Convert string labels to numeric vectors
                        label.append(numeric_label)
                    except ValueError:
                        print("Invalid label encountered and skipped:", each[1:])
                else:
                    data.append(each)

    print("Total sequences read:", len(data))
    print("Total labels read:", len(label))

    # if len(label) > 0:
    #     print("Sample labels (first 5):", label[:5])
    #     label_array = np.vstack(label)
    #     print("Label distribution per class:")
    #     for i in range(label_array.shape[1]):
    #         print(f"Class {i}: {np.sum(label_array[:, i])} positives, {len(label_array) - np.sum(label_array[:, i])} negatives")

    return data, label

def staticTrainAndTest(y_train, y_test):
    data_size_tr = np.zeros(len(filenames))
    data_size_te = np.zeros(len(filenames))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
                 data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    return data_size_tr

class SequenceGraphDataset(Dataset):
    def __init__(self, sequences, labels, seq_lengths, max_len, features):
        assert len(sequences) == len(labels) == len(
            seq_lengths), "Sequences, labels, and seq_lengths must have the same length."
        self.sequences = sequences
        self.labels = labels
        self.seq_lengths = seq_lengths
        self.max_len = max_len
        self.features = features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        seq_length = self.seq_lengths[idx]

        seq_tensor = torch.LongTensor(seq)
        label_tensor = torch.FloatTensor(label)
        seq_length_tensor = torch.tensor(seq_length, dtype=torch.long)  # 转换为张量

        # 获取特征
        aai = torch.FloatTensor(self.features['aai'][idx])
        paac = torch.FloatTensor(self.features['paac'][idx])
        pc6 = torch.FloatTensor(self.features['pc6'][idx])
        blosum62 = torch.FloatTensor(self.features['blosum62'][idx])
        aac = torch.FloatTensor(self.features['aac'][idx])

        # 构建邻接矩阵
        edge_index = []
        for i in range(self.max_len - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]

        # 标签作为全局特征
        label_global = torch.FloatTensor(label)

        # 构建图数据
        data = GeometricData(x=torch.randn(self.max_len, 256),  # 假设 GNN 输入特征维度为 256
                             edge_index=edge_index,
                             y=label_tensor,
                             label_global=label_global)

        return seq_tensor, label_tensor, seq_length_tensor, aai, paac, pc6, blosum62, aac, data

def collate_fn(batch):
    seq_tensors, label_tensors, seq_lengths, aai, paac, pc6, blosum62, aac, graphs = zip(*batch)
    seq_tensors = torch.stack(seq_tensors)
    label_tensors = torch.stack(label_tensors)
    seq_lengths = torch.stack(seq_lengths)
    aai = torch.stack(aai)
    paac = torch.stack(paac)
    pc6 = torch.stack(pc6)
    blosum62 = torch.stack(blosum62)
    aac = torch.stack(aac)
    graphs = torch_geometric.data.Batch.from_data_list(graphs)
    return seq_tensors, label_tensors, seq_lengths, aai, paac, pc6, blosum62, aac, graphs.edge_index, graphs.x, graphs.batch


def main(num, data):


    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_path = os.path.join('result', 'output_log.txt')
    log_file = open(log_path, 'a')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print("=" * 80)
    print(f"Start Training Model {num + 1} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    first_dir = 'dataset'
    max_length = 50

    # 获取训练和测试数据
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

    # 将标签转换为numpy数组
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)

    # 使用 PadEncode 进行序列和标签的处理
    x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
    x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)

    # 打印训练和测试集的大小（调试用）
    print(f"After PadEncode - Training set size: {x_train.shape[0]}, Training labels size: {y_train.shape[0]}")
    print(f"After PadEncode - Test set size: {x_test.shape[0]}, Test labels size: {y_test.shape[0]}")

    # 确保训练集和测试集中的序列和标签数量相同
    assert x_train.shape[0] == y_train.shape[0], "Training data and labels size mismatch!"
    assert x_test.shape[0] == y_test.shape[0], "Test data and labels size mismatch!"

    # 提取特征
    train_features_aai = AAI_embedding(train_sequence_data, max_len=max_length)
    test_features_aai = AAI_embedding(test_sequence_data, max_len=max_length)
    train_features_paac = PAAC_embedding(train_sequence_data, max_len=max_length)
    test_features_paac = PAAC_embedding(test_sequence_data, max_len=max_length)
    train_features_pc6 = PC6_embedding(train_sequence_data, max_len=max_length)
    test_features_pc6 = PC6_embedding(test_sequence_data, max_len=max_length)
    train_features_blosum62 = BLOSUM62_embedding(train_sequence_data, max_len=max_length)
    test_features_blosum62 = BLOSUM62_embedding(test_sequence_data, max_len=max_length)
    train_features_aac = AAC_embedding(train_sequence_data)
    test_features_aac = AAC_embedding(test_sequence_data)


    train_features = {
        'aai': train_features_aai,
        'paac': train_features_paac,
        'pc6': train_features_pc6,
        'blosum62': train_features_blosum62,
        'aac': train_features_aac
    }
    test_features = {
        'aai': test_features_aai,
        'paac': test_features_paac,
        'pc6': test_features_pc6,
        'blosum62': test_features_blosum62,
        'aac': test_features_aac
    }

    # 创建数据集，传递 seq_length
    train_dataset = SequenceGraphDataset(x_train, y_train, train_length, max_length, train_features)
    test_dataset = SequenceGraphDataset(x_test, y_test, test_length, max_length, test_features)

    # 打印数据集的长度（调试用）
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

    # 定义 DataLoader
    dataset_train = DataLoader(
        train_dataset,
        batch_size=data['batch_size'],
        shuffle=False,  # 暂时关闭 shuffle 以便调试
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False  # 确保每个批次有完整的数据
    )
    # dataset_test = DataLoader(
    #     test_dataset,
    #     batch_size=data['batch_size'],
    #     shuffle=False,  # 关闭 shuffle
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    #     drop_last=False
    # )



    val_size = int(0.2 * len(test_dataset))  # 用 20% 作为验证集
    test_size = len(test_dataset) - val_size

    val_dataset, dataset_test = random_split(test_dataset, [val_size, test_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=para['batch_size'],
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    dataset_test = DataLoader(
        dataset_test,
        batch_size=para['batch_size'],
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )


    # 打印 DataLoader 生成的批次数量（调试用）
    print(f"Number of batches in train DataLoader: {len(dataset_train)}")
    print(f"Number of batches in test DataLoader: {len(dataset_test)}")

    # 初始化模型
    vocab_size = 50
    output_size = 21

    model = MFFTPC(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'],
                data['num_heads'])
    model.to(DEVICE)
    # best_params = optimize_cnn_params(model, val_loader)
    # print(f"[最佳CNN参数] {best_params}")
    # model.conv1 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv1_kernel']).to(model.device)
    # model.conv2 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv2_kernel']).to(model.device)
    # model.conv3 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv3_kernel']).to(model.device)
    # model.conv4 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv4_kernel']).to(model.device)
    # model.MaxPool1d = nn.MaxPool1d(best_params['max_pool']).to(model.device)

    # 只调用一次
    best_params = optimize_model_hyperparams(model, val_loader)
    print(f"[最佳参数] {best_params}")

    # # 然后应用这些参数
    # model.conv1 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv1_kernel']).to(
    #     model.device)
    # model.conv2 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv2_kernel']).to(
    #     model.device)
    # model.conv3 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv3_kernel']).to(
    #     model.device)
    # model.conv4 = nn.Conv1d(model.embedding_size, best_params['out_channels'], best_params['conv4_kernel']).to(
    #     model.device)
    # model.MaxPool1d = nn.MaxPool1d(best_params['max_pool']).to(model.device)

    # 应用 LSTM 参数（如果你扩展了参数空间）
    model.bilstm = nn.LSTM(
        input_size=model.embedding_size,
        hidden_size=best_params['lstm_hidden'],
        num_layers=best_params['lstm_layers'],
        batch_first=True,
        bidirectional=True
    ).to(model.device)

    lstm_out_dim = best_params['lstm_hidden'] * 2
    model.conv1 = nn.Conv1d(lstm_out_dim, best_params['out_channels'], best_params['conv1_kernel']).to(model.device)
    model.conv2 = nn.Conv1d(lstm_out_dim, best_params['out_channels'], best_params['conv2_kernel']).to(model.device)
    model.conv3 = nn.Conv1d(lstm_out_dim, best_params['out_channels'], best_params['conv3_kernel']).to(model.device)
    model.conv4 = nn.Conv1d(lstm_out_dim, best_params['out_channels'], best_params['conv4_kernel']).to(model.device)
    model.MaxPool1d = nn.MaxPool1d(best_params['max_pool']).to(model.device)


    # 应用 GNN 参数（如果你扩展了参数空间）
    model.gnn = GNN(
        input_dim=256,
        hidden_dim=best_params['gnn_hidden'],
        output_dim=256,
        num_heads=best_params['gnn_heads']
    ).to(model.device)

    # 用 dummy 数据推断新的 TextCNN 输出维度
    # dummy_input = torch.randn(1, model.embedding_size, 50).to(model.device)
    # cnn_out_dim = model.TextCNN(dummy_input).shape[1]

    dummy_input = torch.randn(1, lstm_out_dim, 50).to(model.device)
    cnn_out_dim = model.TextCNN(dummy_input).shape[1]

    # 替换 fan 和全连接层维度
    model.fan = FAN_encode(model.dropout_value, cnn_out_dim).to(model.device)
    model.full3 = nn.Linear(cnn_out_dim, 1024).to(model.device)

    rate_learning = data['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
    criterion = MarginalFocalDiceLoss()
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = BCEFocalLoss(gamma=10)

    #criterion = LDAM_loss(max_m=0.5, class_weight="balanced")
    #criterion = FocalDiceLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight'])
    #criterion = AsymmetricLoss()

    # 开始训练
    Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

    a = time.time()
    Train.train_step(dataset_train, epochs=data['epochs'], plot_picture=True)
    b = time.time()
    test_score = evaluate(model, dataset_test, device=DEVICE)
    runtime = b - a

    # 保存模型
    PATH = os.getcwd()
    print(PATH)
    each_model = os.path.join(PATH, 'result', 'Model',  f'model{num+1}.h5')

    torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)

    # 打印结果
    print(f"runtime:{runtime:.3f}s")
    print("测试集：")
    print(f'aiming: {test_score["aiming"]:.3f}')
    print(f'coverage: {test_score["coverage"]:.3f}')
    print(f'accuracy: {test_score["accuracy"]:.3f}')
    print(f'absolute_true: {test_score["absolute_true"]:.3f}')
    print(f'absolute_false: {test_score["absolute_false"]:.3f}')

    # 保存结果到 CSV
    title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
    model_name = f'model{num+1}'
    # now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = [[model_name, '%.3f' % test_score["aiming"],
                '%.3f' % test_score["coverage"],
                '%.3f' % test_score["accuracy"],
                '%.3f' % test_score["absolute_true"],
                '%.3f' % test_score["absolute_false"],
                '%.3f' % runtime,
                now]]

    path = os.path.join('result', 'model_result.csv')

    if os.path.exists(path):
        data1 = pd.read_csv(path, header=None)
        one_line = list(data1.iloc[0])
        if one_line == title:
            with open(path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
        else:
            with open(path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)
                writer.writerows(content)
    else:
        with open(path, 'w', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)

if __name__ == '__main__':


    clip_pos = 0.7
    clip_neg = 0.5
    pos_weight = 0.7

    batch_size = 256
    epochs = 256

    learning_rate = 0.0018

    embedding_size = 256

    dropout = 0.6
    fan_epochs = 1
    num_heads = 8

    para = {
        'clip_pos': clip_pos,
        'clip_neg': clip_neg,
        'pos_weight': pos_weight,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'embedding_size': embedding_size,
        'dropout': dropout,
        'fan_epochs': fan_epochs,
        'num_heads': num_heads
    }
    for i in range(100):
        print(f"Starting training iteration {i + 1}/100")
        main(i, para)








