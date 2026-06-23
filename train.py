"""
统一训练入口 — 改 --model 参数即可切换模型
用法:
    python train.py --model mlp
    python train.py --model dnn_pca --pca-components 100
    python train.py --model transformer --epochs 60
    python train.py --model gcn --threshold 0.3
    python train.py --model gcn_pro
    python train.py --list                          # 列出所有模型
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from models import MODEL_REGISTRY, list_models
from src.dataset import load_raw_data
from src.config import DEVICE

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── 数据管线 ─────────────────────────────────────────────────

def _fc_matrices(ts_data, labels_raw):
    """时间序列 → FC 矩阵 (公用)"""
    matrices = []
    for ts in ts_data:
        noise = np.random.normal(0, 1e-5, size=ts.shape)
        m = np.corrcoef(ts + noise, rowvar=False)
        m = np.nan_to_num(m)
        matrices.append(m)
    return np.array(matrices, dtype=np.float32), np.array(labels_raw, dtype=np.int64)


def _build_pyg_data(ts_data, labels_raw, threshold=0.3):
    """FC 矩阵 → PyG Data 列表"""
    from torch_geometric.data import Data
    data_list = []
    for ts, lbl in zip(ts_data, labels_raw):
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
    return data_list


def _prepare_mlp():
    ts_data, labels = load_raw_data()
    X = np.array([np.corrcoef(ts.T) for ts in ts_data])
    y = np.array(labels)
    triu = np.triu_indices(116, k=1)
    X_feat = np.array([m[triu] for m in X], dtype=np.float32)
    # 标准化
    scaler = StandardScaler()
    X_feat = scaler.fit_transform(X_feat)
    X_t, X_v, y_t, y_v = train_test_split(X_feat, y, test_size=0.2,
                                           stratify=y, random_state=42)
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_t, dtype=torch.float32),
        torch.tensor(y_t, dtype=torch.long)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_v, dtype=torch.float32),
        torch.tensor(y_v, dtype=torch.long)), batch_size=32, shuffle=False)
    return train_loader, val_loader, X_feat.shape[1]


def _prepare_dnn_pca(args):
    ts_data, labels = load_raw_data()
    X = np.array([np.corrcoef(ts.T) for ts in ts_data])
    y = np.array(labels, dtype=np.float32)
    triu = np.triu_indices(116, k=1)
    X_feat = np.array([m[triu] for m in X])

    pca = PCA(n_components=args.pca_components)
    X_pca = pca.fit_transform(X_feat)
    print(f"  PCA 累计方差比: {pca.explained_variance_ratio_.sum():.3f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    X_t, X_v, y_t, y_v = train_test_split(X_scaled, y, test_size=0.2,
                                           stratify=y, random_state=42)
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_t, dtype=torch.float32),
        torch.tensor(y_t, dtype=torch.float32).view(-1, 1)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_v, dtype=torch.float32),
        torch.tensor(y_v, dtype=torch.float32).view(-1, 1)), batch_size=32, shuffle=False)
    return train_loader, val_loader, args.pca_components


def _prepare_transformer():
    ts_data, labels = load_raw_data()
    X, y = _fc_matrices(ts_data, labels)
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=42)
    # 标准化
    mean, std = X_t.mean(), X_t.std() + 1e-8
    X_t = (X_t - mean) / std
    X_v = (X_v - mean) / std
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_t, dtype=torch.float32),
        torch.tensor(y_t, dtype=torch.long)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_v, dtype=torch.float32),
        torch.tensor(y_v, dtype=torch.long)), batch_size=32, shuffle=False)
    return train_loader, val_loader, 116  # input_dim for transformer


def _prepare_gcn(args):
    ts_data, labels = load_raw_data()
    data_list = _build_pyg_data(ts_data, labels, args.threshold)
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    return train_loader, val_loader, data_list[0].x.shape[1]


# ─── 训练循环 ─────────────────────────────────────────────────

def _train_one_epoch(model, loader, criterion, optimizer, device, family):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if family.startswith("gcn"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
            correct += int((out.argmax(dim=-1) == batch.y).sum())
        else:
            xb, yb = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total += yb.size(0)
            if criterion.__class__.__name__ == "BCELoss":
                correct += ((out > 0.5).float() == yb).sum().item()
            else:
                correct += (out.argmax(dim=1) == yb).sum().item()
    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0


def _evaluate(model, loader, criterion, device, family):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, trues_all = [], []
    with torch.no_grad():
        for batch in loader:
            if family.startswith("gcn"):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                total_loss += loss.item() * batch.num_graphs
                total += batch.num_graphs
                pred = out.argmax(dim=-1)
                correct += int((pred == batch.y).sum())
                preds_all.extend(pred.cpu().tolist())
                trues_all.extend(batch.y.cpu().tolist())
            else:
                xb, yb = batch[0].to(device), batch[1].to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.size(0)
                total += yb.size(0)
                if criterion.__class__.__name__ == "BCELoss":
                    pred = (out > 0.5).float()
                    correct += (pred == yb).sum().item()
                    preds_all.extend(pred.cpu().tolist())
                else:
                    pred = out.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    preds_all.extend(pred.cpu().tolist())
                trues_all.extend(yb.cpu().tolist())
    avg_loss = total_loss / total if total > 0 else 0
    acc = correct / total if total > 0 else 0
    return avg_loss, acc, preds_all, trues_all


def _train_simple(model_cls, input_dim, train_loader, val_loader, args):
    """简单划分训练"""
    device = torch.device(DEVICE)
    model = model_cls(input_dim).to(device)

    if args.loss == "bce":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tl, ta = _train_one_epoch(model, train_loader, criterion, optimizer, device, args.family)
        vl, va, _, _ = _evaluate(model, val_loader, criterion, device, args.family)
        scheduler.step(vl)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        if va > best_val_acc:
            best_val_acc = va
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")

        if patience_counter >= args.patience:
            print(f"  ⏹ 触发早停 (Epoch {epoch})")
            break

    print(f"  ✔ 最优 Val Acc: {best_val_acc:.4f}")
    return history, model


def _train_kfold(model_cls, input_dim, args):
    """K-Fold 训练 (transformer, gcn_pro)"""
    from sklearn.metrics import confusion_matrix
    device = torch.device(DEVICE)

    ts_data, labels = load_raw_data()


    if args.family == "gcn_kfold":
        from torch_geometric.loader import DataLoader as PyGDataLoader
        data_list = _build_pyg_data(ts_data, labels, args.threshold)
        all_labels = [d.y.item() for d in data_list]
    else:
        X, y = _fc_matrices(ts_data, labels)
        data_list = X
        all_labels = y.tolist()

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_accs, all_histories = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n  ── Fold {fold + 1}/{args.k_folds} ──")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.family == "gcn_kfold":
            train_subset = [data_list[i] for i in tr_idx]
            val_subset = [data_list[i] for i in te_idx]
            train_loader = PyGDataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = PyGDataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

            model = model_cls(num_node_features=data_list[0].x.shape[1],
                              hidden_channels=args.hidden).to(device)
        else:
            X_tr, X_te = data_list[tr_idx], data_list[te_idx]
            y_tr, y_te = np.array(all_labels)[tr_idx], np.array(all_labels)[te_idx]
            mean, std = X_tr.mean(), X_tr.std() + 1e-8
            X_tr = (X_tr - mean) / std
            X_te = (X_te - mean) / std
            train_loader = DataLoader(TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long)), batch_size=32, shuffle=True)
            val_loader = DataLoader(TensorDataset(
                torch.tensor(X_te, dtype=torch.float32),
                torch.tensor(y_te, dtype=torch.long)), batch_size=32, shuffle=False)

            model = model_cls().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc, patience_counter = 0.0, 0

        for epoch in range(1, args.epochs + 1):
            tl, ta = _train_one_epoch(model, train_loader, criterion, optimizer, device, args.family)
            vl, va, _, _ = _evaluate(model, val_loader, criterion, device, args.family)
            scheduler.step(vl)

            history["train_loss"].append(tl)
            history["train_acc"].append(ta)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)

            if va > best_val_acc:
                best_val_acc = va
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:03d} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")

            if patience_counter >= args.patience:
                break

        print(f"  Fold {fold + 1} 最优 Val Acc: {best_val_acc:.4f}")
        fold_accs.append(best_val_acc)
        all_histories.append(history)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n  ✔ {args.k_folds}-Fold 平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")

    return all_histories, fold_accs


# ─── 主入口 ─────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="统一训练入口")
    parser.add_argument("--model", type=str, default="mlp",
                        help="模型名称 (--list 查看所有)")
    parser.add_argument("--list", action="store_true", help="列出所有可用模型")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--save", action="store_true", default=True,
                        help="保存结果到 runs/")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_models()
        return

    if args.model not in MODEL_REGISTRY:
        print(f"❌ 未知模型 '{args.model}'")
        list_models()
        return

    info = MODEL_REGISTRY[args.model]
    args.loss = info["loss"]
    args.family = info["family"]
    args.desc = info["desc"]
    print(f"\n{'='*50}")
    print(f"  模型: {args.model} — {info['desc']}")
    print(f"  family: {info['family']} | loss: {info['loss']}")
    print(f"{'='*50}\n")

    # 数据准备 → 训练
    history = None
    fold_accs = None

    if args.family == "simple":
        if args.model == "mlp":
            train_loader, val_loader, input_dim = _prepare_mlp()
        elif args.model == "dnn_pca":
            train_loader, val_loader, input_dim = _prepare_dnn_pca(args)
        elif args.model == "gcn":
            train_loader, val_loader, input_dim = _prepare_gcn(args)
        else:
            train_loader, val_loader, input_dim = _prepare_mlp()
        history, model = _train_simple(info["class"], input_dim, train_loader, val_loader, args)

    elif args.family == "kfold":
        history, fold_accs = _train_kfold(info["class"], None, args)

    elif args.family == "gcn_simple":
        train_loader, val_loader, input_dim = _prepare_gcn(args)
        history, model = _train_simple(info["class"], input_dim, train_loader, val_loader, args)

    elif args.family == "gcn_kfold":
        history, fold_accs = _train_kfold(info["class"], None, args)

    else:
        print(f"❌ 未知 family: {args.family}")
        return

    # 保存结果
    if args.save:
        save = {
            "model": args.model,
            "desc": args.desc,
            "family": args.family,
            "history": history,
            "fold_accs": fold_accs,
            "args": vars(args),
        }
        path = os.path.join(RESULTS_DIR, f"{args.model}_results.json")
        # 用 json 保存可序列化的部分
        json_safe = {
            "model": args.model,
            "desc": args.desc,
            "family": args.family,
            "fold_accs": fold_accs,
            "args": vars(args),
        }
        with open(path, "w") as f:
            json.dump(json_safe, f, indent=2, default=str)
        # 完整历史存 pickle (包含 numpy 数组)
        import pickle
        pkl_path = os.path.join(RESULTS_DIR, f"{args.model}_results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(save, f)
        print(f"\n  💾 结果已保存: {pkl_path}")

    print(f"\n  ✅ {args.model} 训练完成！运行以下命令查看可视化:")
    print(f"     python visualize.py --model {args.model}")


if __name__ == "__main__":
    main()
