"""
统一可视化脚本 — 加载训练历史，生成标准图表
用法:
    python visualize.py --model mlp
    python visualize.py --model transformer
    python visualize.py --list                          # 列出模型
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models import MODEL_REGISTRY, list_models

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def _load(model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_results.pkl")
    if not os.path.exists(path):
        print(f"❌ 未找到结果: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _plot_simple(history, model_name, desc):
    """简单 train/val 的 Loss + Acc 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{desc} — Loss 曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", lw=2, label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r--", lw=2, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{desc} — Accuracy 曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{model_name}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  💾 已保存: {path}")
    plt.close()


def _plot_kfold(history, fold_accs, model_name, desc):
    """K-Fold: 逐折细线 + 平均粗线 + 折叠准确率柱状图"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # ─── 曲线图 (Loss + Acc) ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    num_folds = len(history)
    max_epochs = max(len(h["train_loss"]) for h in history)

    all_train_loss = np.full((num_folds, max_epochs), np.nan)
    all_val_loss = np.full((num_folds, max_epochs), np.nan)
    all_train_acc = np.full((num_folds, max_epochs), np.nan)
    all_val_acc = np.full((num_folds, max_epochs), np.nan)

    for i, h in enumerate(history):
        cur = len(h["train_loss"])
        all_train_loss[i, :cur] = h["train_loss"]
        all_val_loss[i, :cur] = h["val_loss"]
        all_train_acc[i, :cur] = h["train_acc"]
        all_val_acc[i, :cur] = h["val_acc"]

        ax1.plot(range(1, cur + 1), h["train_loss"], color="blue", alpha=0.12, ls="--")
        ax1.plot(range(1, cur + 1), h["val_loss"], color="red", alpha=0.12, ls="--")
        ax2.plot(range(1, cur + 1), h["train_acc"], color="blue", alpha=0.12, ls="--")
        ax2.plot(range(1, cur + 1), h["val_acc"], color="red", alpha=0.12, ls="--")

    x = np.arange(1, max_epochs + 1)
    ax1.plot(x, np.nanmean(all_train_loss, axis=0), "b-", lw=2, label="Mean Train Loss")
    ax1.plot(x, np.nanmean(all_val_loss, axis=0), "r-", lw=2, label="Mean Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{desc} — K-Fold Loss 曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, np.nanmean(all_train_acc, axis=0), "b-", lw=2, label="Mean Train Acc")
    ax2.plot(x, np.nanmean(all_val_acc, axis=0), "r-", lw=2, label="Mean Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{desc} — K-Fold Accuracy 曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(FIGURE_DIR, f"{model_name}_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    print(f"  💾 已保存: {curves_path}")
    plt.close()

    # ─── 折叠准确率柱状图 ───
    if fold_accs:
        plt.figure(figsize=(8, 5))
        folds = [f"Fold {i+1}" for i in range(len(fold_accs))]
        colors = sns.color_palette("viridis", len(fold_accs))
        bars = plt.bar(folds, fold_accs, color=colors)
        mean_acc = np.mean(fold_accs)
        plt.axhline(y=mean_acc, color="r", ls="--", lw=2, label=f"Mean: {mean_acc:.4f}")

        for bar, val in zip(bars, fold_accs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        plt.ylim(0, 1.0)
        plt.title(f"{desc} — 各折准确率")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        bars_path = os.path.join(FIGURE_DIR, f"{model_name}_fold_bars.png")
        plt.savefig(bars_path, dpi=150, bbox_inches="tight")
        print(f"  💾 已保存: {bars_path}")
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="统一可视化")
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model not in MODEL_REGISTRY:
        print(f"❌ 未知模型 '{args.model}'")
        list_models()
        return

    data = _load(args.model)
    if data is None:
        return

    info = MODEL_REGISTRY[args.model]
    desc = info["desc"]
    history = data["history"]
    fold_accs = data.get("fold_accs")

    print(f"\n{'='*50}")
    print(f"  可视化: {args.model} — {desc}")
    print(f"{'='*50}\n")

    if info["family"] in ("kfold", "gcn_kfold"):
        _plot_kfold(history, fold_accs, args.model, desc)
    else:
        _plot_simple(history, args.model, desc)

    print(f"\n  ✅ 可视化完成! 图片保存到: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
