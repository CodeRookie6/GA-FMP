import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_features):
        super(FeatureFusion, self).__init__()
        self.num_features = num_features
        self.weights = nn.Parameter(torch.ones(num_features, input_dim))
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        # features should be a list of [batch_size, sequence_length, feature_dim] tensors
        weighted_features = [features[i] * self.weights[i] for i in range(self.num_features)]
        fused_feature = sum(weighted_features) / self.num_features
        return fused_feature


class GateFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_features,dropout_value):
        super(GateFeatureFusion, self).__init__()
        self.num_features = num_features
        self.gates = nn.Parameter(torch.rand(num_features, input_dim))
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        # features 是一个 [batch_size, sequence_length, feature_dim] 的列表
        gated_features = [features[i] * torch.sigmoid(self.gates[i]) for i in range(self.num_features)]
        fused_feature = sum(gated_features) / self.num_features
        return fused_feature






