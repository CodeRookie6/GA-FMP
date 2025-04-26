#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/8/3 21:54
# @Author : fhh
# @FileName: util.py
# @Software: PyCharm

import math
import torch
from torch import nn
from attention import *


#AddNorm 类是一个残差连接后进行层归一化的模块。在初始化中，它定义了一个 dropout 层和一个层归一化层。在前向传播中，它将输入 X 与 y 相加，并经过
# dropout 和层归一化处理。
class AddNorm(nn.Module):
    """残差连接后进行层归一化"""

    def __init__(self, normalized, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized)

    def forward(self, X, y):
        return self.ln(self.dropout(y) + X)




#PositionWiseFFN 类是一个基于位置的前馈网络。它包含两个全连接层和一个 ReLU 激活函数。在前向传播中，输入 X 首先经过第一个全连接层和 ReLU 激活函数，
# 然后再经过第二个全连接层
class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


#PositionalEncoding 类是一个位置编码模块。在初始化中，它首先定义了一个 dropout 层，并创建了一个足够长的位置编码矩阵 P。位置编码是通过公式生成的，
# 其中包括正弦和余弦函数。在前向传播中，输入 X 与位置编码 P 相加，并经过 dropout 处理。
class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)





# 定义注意力编码器模型类
class AttentionEncode(nn.Module):

    # 初始化函数，传入参数包括dropout、嵌入大小和头数
    def __init__(self, dropout, embedding_size, num_heads):
        super(AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        # 创建多头注意力层，指定嵌入维度、头数和dropout概率
        self.at1 = nn.MultiheadAttention(embed_dim=self.embedding_size,
                                         num_heads=num_heads,
                                         dropout=0.6
                                         )
        # self.at1 = MultiHeadAttention(key_size=self.embedding_size,
        #                               query_size=self.embedding_size,
        #                               value_size=self.embedding_size,
        #                               num_hiddens=self.embedding_size,
        #                               num_heads=self.num_heads,
        #                               dropout=self.dropout)
        # 创建一个添加归一化层
        self.addNorm1 = AddNorm(normalized=[50, self.embedding_size], dropout=self.dropout)
        # 创建一个位置编码前馈神经网络
        self.FFN = PositionWiseFFN(ffn_num_input=64, ffn_num_hiddens=256, ffn_num_outputs=64)


    # 定义前向传播函数，传入输入张量x和可选的张量y
    def forward(self, x, y=None):
        # 使用多头注意力层处理输入张量x，得到多头输出和注意力权重
        Multi, _ = self.at1(x, x, x)

        # 将多头输出与原始输入进行加归一化处理
        Multi_encode = self.addNorm1(x, Multi)

        # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

        return Multi_encode



# 定义FAN编码器模型类
class FAN_encode(nn.Module):

    # 初始化函数，传入参数包括dropout和形状
    def __init__(self, dropout, shape):
        super(FAN_encode, self).__init__()
        self.dropout = dropout
        # 创建一个添加归一化层
        self.addNorm = AddNorm(normalized=[1, shape], dropout=self.dropout)
        # 创建一个位置编码前馈神经网络
        self.FFN = PositionWiseFFN(ffn_num_input=shape, ffn_num_hiddens=(2*shape), ffn_num_outputs=shape)

    def forward(self, x):
        # 将输入张量经过前馈神经网络后的输出与原始输入进行加归一化处理
        encode_output = self.addNorm(x, self.FFN(x))

        return encode_output







# 定义序列掩码函数，用于屏蔽不相关的项
def sequence_mask(X, valid_len, value=0.):
    """在序列中屏蔽不相关的项"""
    # 将有效长度转换为浮点数类型
    valid_len = valid_len.float()
    # 获取输入张量X的最大长度
    MaxLen = X.size(1)
    # 生成一个掩码张量，将小于有效长度的位置置为True，大于等于有效长度的位置置为False
    mask = torch.arange(MaxLen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None].to(X.device)
    # 将X中掩码为True的位置的值替换为指定值
    X[~mask] = value
    return X


# 定义掩码softmax函数，用于通过掩蔽无效位置来执行softmax操作
def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    # 如果有效长度为None，则直接使用PyTorch中的softmax函数
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        # 获取输入张量X的形状
        shape = X.shape
    if valid_lens.dim() == 1:
        # 如果有效长度是一维张量，则将其重复扩展到与输入张量X相同的长度
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        # 否则，将其重塑为一维张量
        valid_lens = valid_lens.reshape(-1)  # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
    # 使用序列掩码函数对输入张量X进行掩码
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    # 返回softmax函数应用后的结果
    return nn.functional.softmax(X.reshape(shape), dim=-1)


# class AdditiveAttention(nn.Module):
#     """加性注意⼒"""
#
#     def __init__(self, key_size, query_size, num_hiddens, dropout):
#         super(AdditiveAttention, self).__init__()
#         self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
#         self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
#         self.w_v = nn.Linear(num_hiddens, 1, bias=False)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, queries, keys, values, valid_lens):
#         queries, keys = self.W_q(queries), self.W_k(keys)
#         # 在维度扩展后，
#         # queries的形状：(batch_size，查询的个数，1，num_hidden)
#         # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
#         # 使⽤⼴播⽅式进⾏求和
#         features = queries.unsqueeze(2) + keys.unsqueeze(1)
#         features = torch.tanh(features)
#         # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
#         # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
#         scores = self.w_v(features).squeeze(-1)
#         attention_weights = masked_softmax(scores, valid_lens)
#         # values的形状：(batch_size，“键－值”对的个数，值的维度)
#         return torch.bmm(self.dropout(attention_weights), values)


#加性注意力，在加性注意力中，通过对查询和键进行线性变换，然后计算注意力分数，再对值进行加权求和
class AdditiveAttention(nn.Module):
    """注意⼒机制"""

    def __init__(self, input_size, value_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        # 线性变换，用于将输入转换到指定维度
        self.W_k = nn.Linear(input_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(input_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(input_size, num_hiddens, bias=False)
        # 输出层的线性变换
        self.w_o = nn.Linear(50, value_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # 对查询和键进行线性变换
        queries, keys = self.W_q(queries), self.W_k(keys)
        d = queries.shape[-1]
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使⽤⼴播⽅式进⾏求和
        # features = queries + keys
        # features = torch.tanh(features)
        # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)

        # 计算注意力分数，使用点积注意力
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 对注意力分数进行线性变换和softmax操作
        scores = self.w_o(scores).permute(0, 2, 1)
        attention_weights = masked_softmax(scores, valid_lens)

        # attention_weights = nn.Softmax(dim=1)(scores)
        # 使用注意力权重对值进行加权求和
        values = self.w_v(values)
        # values = torch.transpose(values, 1, 2)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(attention_weights), values), attention_weights


#多头注意力，多头注意力将注意力机制拆分为多个头，每个头分别进行注意力计算，最后将多个头的输出进行合并并进行线性变换。
class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        # 单个头的注意力机制
        self.attention = DotProductAttention(dropout)
        # 多头注意力的线性变换
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        # 将查询、键、值进行线性变换并分割成多个头
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            # 在轴0上重复valid_lens多次，以匹配多头的数量
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # 进行多头注意力计算
        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # 合并多个头的输出并进行线性变换
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


#这两个函数用于变换张量的形状，以适应多头注意力机制的并行计算。
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    # 将输入X的形状重塑为(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    # 将维度顺序变换为(batch_size，num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    # 将X重塑为(batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)并返回
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    # 逆转transpose_qkv函数的操作，将X的形状重塑为(batch_size，num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


#这是一个缩放点积注意力机制的类实现，用于计算注意力权重并应用到给定的值上。
class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        # 计算缩放点积注意力分数
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        # 注意力加权平均values并返回
        return torch.bmm(self.dropout(attention_weights), values)


#这个类是一个基于多头注意力机制的编码器层，通过多头注意力机制处理输入并加上残差连接和层归一化。
class MASK_AttentionEncode(nn.Module):

    def __init__(self, dropout, embedding_size, num_heads):
        super(MASK_AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # 定义多头注意力层和AddNorm层
        self.at1 = MultiHeadAttention(key_size=self.embedding_size,
                                      query_size=self.embedding_size,
                                      value_size=self.embedding_size,
                                      num_hiddens=self.embedding_size,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout)
        self.addNorm = AddNorm(normalized=[50, self.embedding_size], dropout=self.dropout)

        # 定义前馈神经网络层
        self.FFN = PositionWiseFFN(ffn_num_input=64, ffn_num_hiddens=256, ffn_num_outputs=64)

    def forward(self, x, y=None):
        # Multi, _ = self.at1(x, x, x)
        # 调用多头注意力层并进行AddNorm处理
        Multi = self.at1(x, x, x, y)
        Multi_encode = self.addNorm(x, Multi)

        # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

        return Multi_encode


#这个类是一个Transformer编码器层，通过多头注意力机制和前馈神经网络来对输入进行编码。
class transformer_encode(nn.Module):

    def __init__(self, dropout, embedding, num_heads):
        super(transformer_encode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding
        self.num_heads = num_heads

        #创建一个多头注意力（Multihead Attention）的模块
        self.attention = nn.MultiheadAttention(embed_dim=256,
                                               num_heads=8,
                                               dropout=0.6
                                               )

        # 定义多头注意力层和AddNorm层
        self.at1 = MultiHeadAttention(key_size=self.embedding_size,
                                      query_size=self.embedding_size,
                                      value_size=self.embedding_size,
                                      num_hiddens=self.embedding_size,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout)

        self.addNorm = AddNorm(normalized=[50, self.embedding_size], dropout=self.dropout)

        # 定义前馈神经网络层
        self.ffn = PositionWiseFFN(ffn_num_input=self.embedding_size, ffn_num_hiddens=2*self.embedding_size,
                                   ffn_num_outputs=self.embedding_size)

    def forward(self, x, valid=None):
        # Multi, _ = self.attention(x, x, x)
        # 调用多头注意力层并进行AddNorm处理
        Multi = self.at1(x, x, x, valid)
        Multi_encode = self.addNorm(x, Multi)

        # 经过前馈神经网络层处理并进行AddNorm处理
        encode_output = self.addNorm(Multi_encode, self.ffn(Multi_encode))

        return encode_output
