"""
统一评估脚本 — 加载已训练模型在测试集上跑指标
用法:
    python evaluate.py --model mlp
    python evaluate.py --model gcn --threshold 0.3
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

from models import MODEL_REGISTRY, list_models
from src.dataset import load_raw_data
from src.config import DEVICE

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _load_results(model_name):
    """加载训练结果和模型"""
    pkl_path = os.path.join(RESULTS_DIR, f"{model_name}_results.pkl")
    if not os.path.exists(pkl_path):
        print(f"❌ 未找到训练结果: {pkl_path}")
        print(f"   请先运行: python train.py --model {model_name}")
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _prepare_test_data(model_name, args=None):
    """准备测试集数据"""
    ts_data, labels = load_raw_data()

    if model_name == "dnn_pca" or model_name == "dnn_pca_model":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = np.array([np.corrcoef(ts.T) for ts in ts_data])
        y = np.array(labels, dtype=np.float32)
        triu = np.triu_indices(116, k=1)
        X_feat = np.array([m[triu] for m in X])
        pca = PCA(n_components=100)
        X_pca = pca.fit_transform(X_feat)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pca)
        _, X_te, _, y_te = train_test_split(X_scaled, y, test_size=0.2,
                                             stratify=y, random_state=42)
        loader = DataLoader(TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.float32).view(-1, 1)), batch_size=32)
        return loader

    elif model_name == "transformer":
        X = np.array([np.corrcoef(ts.T) for ts in ts_data])
        y = np.array(labels)
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.2,
                                             stratify=y, random_state=42)
        # 简单标准化用训练集统计量
        mean, std = X_te.mean(), X_te.std() + 1e-8
        X_te = (X_te - mean) / std
        loader = DataLoader(TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.long)), batch_size=32)
        return loader

    elif model_name in ("gcn", "gcn_pro"):
        from torch_geometric.loader import DataLoader as GCNLoader
        from torch_geometric.data import Data
        threshold = getattr(args, "threshold", 0.3) if args else 0.3
        data_list = []
        y_all = []
        for ts, lbl in zip(ts_data, labels):
            noise = np.random.normal(0, 1e-5, size=ts.shape)
            m = np.corrcoef(ts + noise, rowvar=False)
            m = np.nan_to_num(m)
            x = torch.tensor(m, dtype=torch.float)
            adj = np.where(np.abs(m) > threshold, m, 0)
            np.fill_diagonal(adj, 0)
            rows, cols = np.nonzero(adj)
            if len(rows) == 0:
                continue
            edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
            edge_attr = torch.tensor(np.abs(adj[rows, cols]), dtype=torch.float)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                  y=torch.tensor([lbl], dtype=torch.long)))
            y_all.append(lbl)
        test_data = data_list[:int(len(data_list) * 0.2)]
        return GCNLoader(test_data, batch_size=32)

    else:  # mlp / default
        X = np.array([np.corrcoef(ts.T) for ts in ts_data])
        y = np.array(labels)
        triu = np.triu_indices(116, k=1)
        X_feat = np.array([m[triu] for m in X], dtype=np.float32)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_feat = scaler.fit_transform(X_feat)
        _, X_te, _, y_te = train_test_split(X_feat, y, test_size=0.2,
                                             stratify=y, random_state=42)
        loader = DataLoader(TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.long)), batch_size=32)
        return loader


def main():
    import argparse
    parser = argparse.ArgumentParser(description="统一评估")
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model not in MODEL_REGISTRY:
        print(f"❌ 未知模型 '{args.model}'")
        list_models()
        return

    # 加载训练结果
    results = _load_results(args.model)
    if results is None:
        return

    info = MODEL_REGISTRY[args.model]
    print(f"\n{'='*50}")
    print(f"  评估模型: {args.model} — {info['desc']}")
    print(f"{'='*50}")

    # 准备测试数据
    test_loader = _prepare_test_data(args.model, args)

    # 重建模型
    device = torch.device(DEVICE)
    if info["family"] in ("gcn_simple", "gcn_kfold"):
        sample = next(iter(test_loader))
        model = info["class"](num_node_features=sample.x.shape[1]).to(device)
    else:
        sample = next(iter(test_loader))
        if info["family"] == "simple" and args.model != "dnn_pca":
            model = info["class"](sample[0].shape[1]).to(device)
        elif args.model == "dnn_pca":
            model = info["class"](100).to(device)
        else:
            model = info["class"]().to(device)

    # 推理
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            if info["family"].startswith("gcn"):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = out.argmax(dim=-1).cpu().tolist()
                true = batch.y.cpu().tolist()
            else:
                xb, yb = batch[0].to(device), batch[1].to(device)
                out = model(xb)
                if info["loss"] == "bce":
                    pred = (out > 0.5).float().cpu().tolist()
                else:
                    pred = out.argmax(dim=1).cpu().tolist()
                true = yb.cpu().tolist()
            all_preds.extend(pred)
            all_trues.extend(true)

    # 指标
    all_preds = np.array(all_preds).flatten()
    all_trues = np.array(all_trues).flatten()

    # 对二分类 BCE 输出做阈值处理
    if info["loss"] == "bce":
        all_preds = (all_preds > 0.5).astype(int)

    print(f"\n  分类报告:")
    print(classification_report(all_trues, all_preds,
                                target_names=["Control", "ASD"]))

    cm = confusion_matrix(all_trues, all_preds)
    print(f"  混淆矩阵:")
    print(f"              Pred Control  Pred ASD")
    print(f"  True Control     {cm[0,0]:>3d}          {cm[0,1]:>3d}")
    print(f"  True ASD         {cm[1,0]:>3d}          {cm[1,1]:>3d}")

    try:
        auc = roc_auc_score(all_trues, all_preds)
        print(f"\n  AUC: {auc:.4f}")
    except Exception:
        pass

    print(f"\n  ✅ 评估完成")


if __name__ == "__main__":
    main()
