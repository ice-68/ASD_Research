import os
import torch

# 自动定位项目根目录（基于 src/config.py 的位置）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "abide_data", "ABIDE_pcp")

CSV_PATH = os.path.join(DATA_DIR, "Phenotypic_V1_0b_preprocessed1.csv")
TS_FOLDER_PATH = os.path.join(DATA_DIR, "cpac", "nofilt_noglobal")

# 数据类型控制：'raw' 或 'pearson'
DATA_TYPE = 'pearson'

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# 自动检测设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
