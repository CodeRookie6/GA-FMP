# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: predictor.py
# @Software: PyCharm

import os
from ETFC.model import *

# 设置环境变量，指定可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from pathlib import Path
import argparse
import torch
import pytest

# 定义函数 ArgsGet，用于获取命令行参数
def ArgsGet():
    parse = argparse.ArgumentParser(description='ETFC')
    # 添加一个参数--file，指定fasta文件，默认为'test.fasta'
    parse.add_argument('--file', type=str, default='test.fasta', help='fasta file')
    # 添加一个参数--out_path，指定输出路径，默认为'result'
    parse.add_argument('--out_path', type=str, default='result', help='output path')
    # 解析命令行参数
    args = parse.parse_args()
    return args


# 定义函数 get_data，用于从文件中获取数据并进行编码处理
# 读取文件并处理编码
def get_data(file):
    # getting file and encoding
    # 用于存储序列数据
    seqs = []
    # 用于存储序列名称
    names = []
    # 用于存储序列长度
    seq_length = []
    with open(file) as f:
        for each in f:
            if each == '\n':
                continue
            elif each[0] == '>':
                # 将序列名称添加到 names 列表中
                names.append(each)
            else:
                # 将序列数据添加到 seqs 列表中，并移除末尾的空白字符
                seqs.append(each.rstrip())

    # 编码处理
    # encoding
    # 氨基酸序列
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    # 最大序列长度
    max_len = 50
    # 用于存储编码后的序列数据
    data_e = []
    for i in range(len(seqs)):
        length = len(seqs[i])
        # 将序列长度添加到 seq_length 列表中
        seq_length.append(length)
        #seqs[i]代表着序列数据中的第i个序列，elemt将用于存储该序列的氨基酸编码，而st则用于暂时存储当前处理的序列数据
        elemt, st = [], seqs[i]
        for j in st:
            # 获取氨基酸的索引
            index = amino_acids.index(j)
            # 将氨基酸索引添加到 elemt 列表中
            elemt.append(index)
        if length <= max_len:
            # 将序列长度补齐为最大长度，并添加到 elemt 列表中
            elemt += [0] * (max_len - length)
            # 将处理后的序列数据添加到 data_e 列表中
            data_e.append(elemt)

    return np.array(data_e), names, np.array(seq_length)


#定义函数 predict，用于进行模型预测
def predict(test, seq_length, h5_model):
    # 模型路径
    dir = 'dataset/Model/teacher/tea_model0.h5'
    print('predicting...')

    # 1. 加载模型
    # 1.loading model
    # 实例化 ETFC 模型对象
    model = ETFC(50, 256, 21, 0.6, 1, 8)
    # 加载模型参数
    model.load_state_dict(torch.load(dir))

    # 2. 预测
    # 2.predict
    # 设置模型为评估模式
    model.eval()
    # 对测试数据进行预测
    score_label = model(test, seq_length)

    # 对预测结果进行处理
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5:
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    return score_label


def Test_my(test, seq_length, output_path, names):
    # models
    # 用于存储模型文件名
    h5_model = []
    # 模型数量
    model_num = 10
    # 添加模型文件名到 h5_model 列表中
    for i in range(1, model_num + 1):
        h5_model.append('model{}.h5'.format(str(i)))

    # prediction
    # 调用 predict 函数进行预测
    result = predict(test, seq_length, h5_model)

    # label
    # 标签
    peptides = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
                'AVP',
                'BBP', 'BIP',
                'CPP', 'DPPIP',
                'QSP', 'SBP', 'THP']
    # 用于存储预测结果
    functions = []
    for e in result:
        temp = ''
        for i in range(len(e)):
            if e[i] == 1:
                temp = temp + peptides[i] + ','# 根据预测结果添加标签到 temp 中
            else:
                continue
        if temp == '':
            # 如果没有预测结果，则设为'none'
            temp = 'none'
        if temp[-1] == ',':
            # 移除末尾的逗号
            temp = temp.rstrip(',')
        # 将处理后的预测结果添加到 functions 列表中
        functions.append(temp)

    # 输出文件路径
    output_file = os.path.join(output_path, 'result.txt')
    with open(output_file, 'w') as f:
        for i in range(len(names)):
            # 写入序列名称
            f.write(names[i])
            # 写入预测结果
            f.write('functions:' + functions[i] + '\n')


if __name__ == '__main__':
    # 获取命令行参数
    args = ArgsGet()
    file = args.file  # fasta file
    output_path = args.out_path  # output path

    # building output path directory
    Path(output_path).mkdir(exist_ok=True)

    # reading file and encoding
    data, names, seq_length = get_data(file)
    data = torch.LongTensor(data)
    seq_length = torch.LongTensor(seq_length)

    # prediction
    Test_my(data, seq_length, output_path, names)
