import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    多层感知机分类器，用于 ABIDE 脑功能连接数据。

    输出 2 个 logits → CrossEntropyLoss (标签为 0/1 的 torch.long)
    """

    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output: 2 类 logits (无激活函数, 配合 CrossEntropyLoss)
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)
