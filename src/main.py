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








