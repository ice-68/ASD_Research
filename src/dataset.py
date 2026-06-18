import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import CSV_PATH, TS_FOLDER_PATH, BATCH_SIZE, DATA_TYPE
import pandas as pd


def load_raw_data():
    """
    加载 ABIDE 数据，将 .1D 时间序列文件与表型 CSV 匹配。

    Returns:
        matched_all_time_series: list of np.ndarray, 每个元素形状 (T, 116)
        matched_labels: list of int, 0=Control, 1=ASD
    """
    phenotype_df = pd.read_csv(CSV_PATH)
    file_list = glob.glob(os.path.join(TS_FOLDER_PATH, '*.1D'))

    matched_all_time_series = []
    matched_labels = []

    for file_path in file_list:
        filename = os.path.basename(file_path)

        # 从文件名中提取受试者 ID (5-7位数字)
        match = re.search(r'\d{5,7}', filename)
        if not match:
            continue
        sub_id = int(match.group())

        subject_row = phenotype_df[phenotype_df['SUB_ID'] == sub_id]
        if subject_row.empty:
            continue

        time_series = np.loadtxt(file_path)
        raw_label = subject_row['DX_GROUP'].values[0]

        # 标签映射: DX_GROUP 中原值 1=ASD → 1, 2=Control → 0
        if raw_label == 1:
            label = 1
        elif raw_label == 2:
            label = 0
        else:
            continue  # 过滤异常标签

        matched_all_time_series.append(time_series)
        matched_labels.append(label)

    return matched_all_time_series, matched_labels


class ABIDEDataset(Dataset):
    """ABIDE 脑功能连接数据集，支持 pearson / raw 两种模式。"""

    def __init__(self, time_series_list, labels, data_type='pearson'):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.data = []

        for ts in time_series_list:
            if data_type == 'pearson':
                # 计算皮尔逊相关矩阵 (116 x 116) → 压平
                matrix = np.corrcoef(ts.T)
                # 处理恒定脑区导致的 NaN
                matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
                feat = matrix.flatten()

            elif data_type == 'raw':
                # 原始时间序列：固定截取前 150 个时间点
                min_time_len = 150
                feat = ts[:min_time_len, :].flatten()
            else:
                raise ValueError("Unknown data_type! Choose 'raw' or 'pearson'")

            self.data.append(torch.tensor(feat, dtype=torch.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloader():
    """
    工厂函数，返回 DataLoader 和输入特征维度。

    Returns:
        dataloader: torch DataLoader
        input_dim: int, 每条样本的特征维度
    """
    ts_data, labels = load_raw_data()
    dataset = ABIDEDataset(ts_data, labels, data_type=DATA_TYPE)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), dataset[0][0].shape[0]
