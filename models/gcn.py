"""
BrainGCN — 图卷积网络 (简单 train/test 划分)
原: 01_Classification_Vision_3 / gcn 文件夹
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class BrainGCN(torch.nn.Module):
    """输出 2 个 logits → CrossEntropyLoss"""

    def __init__(self, num_node_features, hidden_channels=32, num_classes=2):
        super().__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        return self.lin2(x)
