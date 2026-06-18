"""
BrainGCNPro — GCN + 跳跃连接 (K-Fold)
原: 01_Classification_Vision_4 / gcn_kfold 文件夹
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv


class BrainGCNPro(torch.nn.Module):
    """输出 2 个 logits → CrossEntropyLoss"""

    def __init__(self, num_node_features, hidden_channels=64, num_classes=2):
        super().__init__()
        torch.manual_seed(32)

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(116 * hidden_channels, 64)
        self.lin2 = Linear(64, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = self.conv1(x, edge_index, edge_weight=edge_attr)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv2(x1, edge_index, edge_weight=edge_attr)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)

        x_out = x1 + x2  # 跳跃连接
        B = int(batch.max()) + 1
        x_flat = x_out.view(B, -1)

        out = F.dropout(x_flat, p=0.5, training=self.training)
        out = F.relu(self.lin1(out))
        out = F.dropout(out, p=0.2, training=self.training)
        return self.lin2(out)
