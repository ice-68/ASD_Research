"""
PCADNNClassifier — DNN with PCA preprocessing
原: 01_Classification_Vision_1 / dnn_pca 文件夹
"""
import torch.nn as nn


class PCADNNClassifier(nn.Module):
    """输出 1 个神经元 + Sigmoid → BCELoss (标签为 float 0/1)"""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
