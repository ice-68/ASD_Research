"""
模型注册表 — 所有模型在这里登记
新增模型只需在这里加一条记录 + 在 train.py 里写对应数据管线
"""
from .MLP import MLPClassifier
from .dnn_pca import PCADNNClassifier
from .transformer import ImprovedBrainTransformer
from .gcn import BrainGCN
from .gcn_pro import BrainGCNPro

MODEL_REGISTRY = {
    "mlp": {
        "class": MLPClassifier,
        "desc": "多层感知机 (FC矩阵→平坦化)",
        "loss": "ce",        # CrossEntropyLoss
        "family": "simple",  # 简单 train/test 划分
    },
    "dnn_pca": {
        "class": PCADNNClassifier,
        "desc": "DNN + PCA 降维",
        "loss": "bce",       # BCELoss
        "family": "simple",
    },
    "transformer": {
        "class": ImprovedBrainTransformer,
        "desc": "Transformer + CLS Token + K-Fold",
        "loss": "ce",
        "family": "kfold",
    },
    "gcn": {
        "class": BrainGCN,
        "desc": "图卷积网络 (GCNConv + 混合池化)",
        "loss": "ce",
        "family": "gcn_simple",
    },
    "gcn_pro": {
        "class": BrainGCNPro,
        "desc": "增强 GCN (跳跃连接 + 展平MLP + K-Fold)",
        "loss": "ce",
        "family": "gcn_kfold",
    },
}


def list_models():
    """打印所有可用模型"""
    print("可用模型:")
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:12s}  {info['desc']}")
    return list(MODEL_REGISTRY.keys())
