# ================================
#  training_code.py
# ================================

import os
import re
import glob
import copy
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler

# 从 predict_utils.py 导入相关函数和模型
from predict_utils import (
    wavelet_denoise,
    extract_features,
    MultiOutputModel,
    TARGET_LENGTH,
    NUM_CLASS_INTENSITY,
    NUM_CLASS_ANGLE,
    NUM_CLASS_FREQ,
    frequency_loss,
    compute_frequency_weights  # 自动统计并计算频率权重的函数
)

mpl.rcParams["font.size"] = 12
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["figure.autolayout"] = True
sns.set_style("whitegrid")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== 全局配置 ==========
DATA_ROOT = r"D:\石阳\博士工作\科研\周报\65-shiyang-20250308\gammry-20250313\night-1st test\全部条件切分周期-auto-two stages"
MODEL_PATH = "pytorch_ecg_model.pth"
TEST_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0
FIX_SEED = True

LOSS_WEIGHT_INTENSITY = 1.0
LOSS_WEIGHT_ANGLE     = 1.0
LOSS_WEIGHT_FREQ      = 1.0

# ========== Dataset & collate_fn ==========
class ECGDataset(Dataset):
    """
    扫描形如 'peaks_auto_(光强)%_(角度)_(频率)Hz' 的子文件夹，按8:2拆分数据。
    """
    def __init__(self, root, test_ratio=0.2, is_train=True):
        super().__init__()
        pattern = r"peaks_auto_(\d+)%_(\d+)_(\d+)Hz"
        subfolders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        all_data = []
        for folder_name in subfolders:
            folder_path = os.path.join(root, folder_name)
            match = re.match(pattern, folder_name)
            if not match:
                continue

            intensity_val = int(match.group(1))
            angle_val     = int(match.group(2))
            freq_val      = int(match.group(3))

            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            random.shuffle(csv_files)
            split_idx = int(len(csv_files) * 0.8)
            if is_train:
                csv_files = csv_files[:split_idx]
            else:
                csv_files = csv_files[split_idx:]

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if "Current_A" not in df.columns:
                    continue
                signal = df["Current_A"].values

                # 小波去噪 + 特征提取（直接调用 predict_utils 中的函数）
                denoised = wavelet_denoise(signal)
                feats = extract_features(denoised, TARGET_LENGTH)

                all_data.append((feats, (intensity_val, angle_val, freq_val)))

        self.samples = all_data
        print(f"{'Train' if is_train else 'Test'} dataset size={len(all_data)}")

        # 构建标签映射
        self.intensity_list = sorted(set(x[1][0] for x in self.samples))
        self.angle_list     = sorted(set(x[1][1] for x in self.samples))
        self.freq_list      = sorted(set(x[1][2] for x in self.samples))

        self.i2idx = {val: idx for idx, val in enumerate(self.intensity_list)}
        self.a2idx = {val: idx for idx, val in enumerate(self.angle_list)}
        self.f2idx = {val: idx for idx, val in enumerate(self.freq_list)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, (i_val, a_val, f_val) = self.samples[idx]
        feats = feats[:TARGET_LENGTH + 2].reshape(-1, 1)  # shape: [102, 1]
        i_idx = self.i2idx[i_val]
        a_idx = self.a2idx[a_val]
        f_idx = self.f2idx[f_val]
        return feats, (i_idx, a_idx, f_idx)

def collate_fn(batch):
    xs = []
    ys = []
    for feats, label_tuple in batch:
        xs.append(torch.tensor(feats, dtype=torch.float32))
        ys.append(torch.tensor(label_tuple, dtype=torch.long))
    x_tensor = torch.stack(xs, dim=0)
    y_tensor = torch.stack(ys, dim=0)
    return x_tensor, y_tensor

# ========== 训练 & 测试辅助函数 ==========
def eval_one_epoch(model, loader, criterion, device, freq_weights):
    model.eval()
    total_loss = 0
    correct_i = correct_a = correct_f = 0
    total_samples = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            yi = Y[:, 0]
            ya = Y[:, 1]
            yf = Y[:, 2]

            out_i, out_a, out_f = model(X)
            prob_i = F.softmax(out_i, dim=1)
            prob_a = F.softmax(out_a, dim=1)
            prob_f = F.softmax(out_f, dim=1)
            pi = prob_i.argmax(dim=1)
            pa = prob_a.argmax(dim=1)
            pf = prob_f.argmax(dim=1)

            li = criterion(out_i, yi)
            la = criterion(out_a, ya)
            lf = frequency_loss(out_f, yf, freq_weights)
            loss = (LOSS_WEIGHT_INTENSITY * li +
                    LOSS_WEIGHT_ANGLE     * la +
                    LOSS_WEIGHT_FREQ      * lf)

            bs = X.size(0)
            total_loss += loss.item() * bs
            correct_i += (pi == yi).sum().item()
            correct_a += (pa == ya).sum().item()
            correct_f += (pf == yf).sum().item()
            total_samples += bs

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        i_acc = correct_i / total_samples
        a_acc = correct_a / total_samples
        f_acc = correct_f / total_samples
    else:
        avg_loss = i_acc = a_acc = f_acc = 0

    return avg_loss, i_acc, a_acc, f_acc

def plot_training_curves(loss_list, i_list, a_list, f_list):
    epochs = len(loss_list)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, epochs + 1), loss_list, label='Train Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()

    axes[1].plot(range(1, epochs + 1), i_list, label='Train_Int')
    axes[1].plot(range(1, epochs + 1), a_list, label='Train_Ang')
    axes[1].plot(range(1, epochs + 1), f_list, label='Train_Frq')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()

    # 保存图像到脚本所在目录
    plt.tight_layout()
    plt.savefig("train_curves.png", dpi=400)
    print("训练曲线已保存为 train_curves.png")
    plt.close()

def plot_testing_curves(test_i_list, test_a_list, test_f_list):
    epochs = len(test_i_list)
    x_axis = range(1, epochs + 1)
    plt.figure(figsize=(6, 5))
    plt.plot(x_axis, test_i_list, label='Test_Int')
    plt.plot(x_axis, test_a_list, label='Test_Ang')
    plt.plot(x_axis, test_f_list, label='Test_Frq')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy (after training finishes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_curves.png", dpi=400)
    print("测试曲线已保存为 test_curves.png")
    plt.close()

def plot_confusion_matrices(model, loader, device,
                            intensity_list, angle_list, freq_list):
    model.eval()
    all_i_true, all_a_true, all_f_true = [], [], []
    all_i_pred, all_a_pred, all_f_pred = [], [], []

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            yi = Y[:, 0]
            ya = Y[:, 1]
            yf = Y[:, 2]

            out_i, out_a, out_f = model(X)
            prob_i = F.softmax(out_i, dim=1)
            prob_a = F.softmax(out_a, dim=1)
            prob_f = F.softmax(out_f, dim=1)
            pi = prob_i.argmax(dim=1).cpu().numpy()
            pa = prob_a.argmax(dim=1).cpu().numpy()
            pf = prob_f.argmax(dim=1).cpu().numpy()

            all_i_true.append(yi.cpu().numpy())
            all_a_true.append(ya.cpu().numpy())
            all_f_true.append(yf.cpu().numpy())
            all_i_pred.append(pi)
            all_a_pred.append(pa)
            all_f_pred.append(pf)

    all_i_true = np.concatenate(all_i_true)
    all_a_true = np.concatenate(all_a_true)
    all_f_true = np.concatenate(all_f_true)
    all_i_pred = np.concatenate(all_i_pred)
    all_a_pred = np.concatenate(all_a_pred)
    all_f_pred = np.concatenate(all_f_pred)

    # 对混淆矩阵进行归一化处理，按行归一化（即每个类别的正确率）
    for title, y_true, y_pred, label_list in [
        ("Intensity",  all_i_true, all_i_pred, intensity_list),
        ("Angle",      all_a_true, all_a_pred, angle_list),
        ("Frequency",  all_f_true, all_f_pred, freq_list),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(label_list)))
        # 若行和为0，则保持为0，否则归一化到 [0,1]
        cm_normalized = np.zeros_like(cm, dtype=np.float32)
        for i in range(cm.shape[0]):
            row_sum = np.sum(cm[i, :])
            if row_sum > 0:
                cm_normalized[i, :] = cm[i, :] / row_sum

        # 保存归一化后的混淆矩阵为 CSV
        norm_df = pd.DataFrame(cm_normalized, index=label_list, columns=label_list)
        norm_df.to_csv(f"confusion_{title}_normalized.csv")
        print(f"混淆矩阵归一化数据[{title}]已保存为 confusion_{title}_normalized.csv")

        plt.figure(figsize=(5, 4))
        # 使用归一化后的数据进行绘制，并显示百分比（2位小数）
        ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_list, yticklabels=label_list)
        plt.title(f"Confusion Matrix - {title}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        filename = f"confusion_{title}.png"
        plt.savefig(filename, dpi=400)
        print(f"混淆矩阵[{title}]已保存为 {filename}")
        plt.close()

# ========== 主函数：train_model ==========
def train_model():
    if FIX_SEED:
        set_seed(42)

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    train_ds = ECGDataset(DATA_ROOT, test_ratio=TEST_RATIO, is_train=True)
    test_ds  = ECGDataset(DATA_ROOT, test_ratio=TEST_RATIO, is_train=False)

    BOOST = {5: 8.0, 40: 4.0} 

    sample_weights = []
    for _, (_, _, f_val) in train_ds.samples:
        sample_weights.append(BOOST.get(f_val, 1.0))
    
    sampler = WeightedRandomSampler(sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler,         # ← 用 sampler
                          collate_fn=collate_fn)   # 不能再传 shuffle
    
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] Using device:", device)

    model = MultiOutputModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 自动统计频率样本数量并计算对应的loss权重
    freq_weights = compute_frequency_weights(train_ds)
    # 为了降低训练轮次增加了5Hz的损失
    idx5 = train_ds.f2idx[5] 
    freq_weights[idx5] *= 10 
    freq_weights = freq_weights / freq_weights.sum() * len(freq_weights)

    train_loss_list = []
    train_i_list = []
    train_a_list = []
    train_f_list = []

    snapshots = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct_i = correct_a = correct_f = 0
        total_samples = 0

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            yi = Y[:, 0]
            ya = Y[:, 1]
            yf = Y[:, 2]

            out_i, out_a, out_f = model(X)
            li = criterion(out_i, yi)
            la = criterion(out_a, ya)
            lf = frequency_loss(out_f, yf, freq_weights)
            loss = (LOSS_WEIGHT_INTENSITY * li +
                    LOSS_WEIGHT_ANGLE     * la +
                    LOSS_WEIGHT_FREQ      * lf)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = X.size(0)
            total_loss += loss.item() * bs

            prob_i = F.softmax(out_i, dim=1)
            prob_a = F.softmax(out_a, dim=1)
            prob_f = F.softmax(out_f, dim=1)
            pi = prob_i.argmax(dim=1)
            pa = prob_a.argmax(dim=1)
            pf = prob_f.argmax(dim=1)

            correct_i += (pi == yi).sum().item()
            correct_a += (pa == ya).sum().item()
            correct_f += (pf == yf).sum().item()
            total_samples += bs

        avg_loss = total_loss / total_samples
        i_acc = correct_i / total_samples
        a_acc = correct_a / total_samples
        f_acc = correct_f / total_samples

        train_loss_list.append(avg_loss)
        train_i_list.append(i_acc)
        train_a_list.append(a_acc)
        train_f_list.append(f_acc)

        print(f"[Epoch {epoch}/{EPOCHS}] Train Loss={avg_loss:.4f}  iAcc={i_acc:.3f}, aAcc={a_acc:.3f}, fAcc={f_acc:.3f}")
        snapshots.append(copy.deepcopy(model.state_dict()))

    # 保存最终模型权重
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[Info] final model saved => {MODEL_PATH}")

    # 绘制训练曲线并保存
    plot_training_curves(train_loss_list, train_i_list, train_a_list, train_f_list)

    # 每个 epoch 的测试
    test_loss_list = []
    test_i_list = []
    test_a_list = []
    test_f_list = []

    temp_model = MultiOutputModel().to(device)
    for epoch_idx, state_dict in enumerate(snapshots, start=1):
        temp_model.load_state_dict(state_dict)
        test_loss, i_acc, a_acc, f_acc = eval_one_epoch(temp_model, test_loader, criterion, device, freq_weights)
        test_loss_list.append(test_loss)
        test_i_list.append(i_acc)
        test_a_list.append(a_acc)
        test_f_list.append(f_acc)
        print(f"[Eval after epoch {epoch_idx}] Test Loss={test_loss:.4f}, iAcc={i_acc:.3f}, aAcc={a_acc:.3f}, fAcc={f_acc:.3f}")

    # 绘制测试曲线并保存
    plot_testing_curves(test_i_list, test_a_list, test_f_list)

    # 用最终 epoch 的状态画归一化混淆矩阵并保存图像
    model.load_state_dict(snapshots[-1])
    plot_confusion_matrices(model, test_loader, device,
                            train_ds.intensity_list,
                            train_ds.angle_list,
                            train_ds.freq_list)
    
      # 保存训练曲线数据为 CSV
    train_metrics = pd.DataFrame({
        "Epoch": list(range(1, EPOCHS + 1)),
        "Train_Loss": train_loss_list,
        "Train_Intensity_Acc": train_i_list,
        "Train_Angle_Acc": train_a_list,
        "Train_Freq_Acc": train_f_list
    })
    train_metrics.to_csv("train_metrics.csv", index=False)
    print("训练过程数据已保存为 train_metrics.csv")

    test_metrics = pd.DataFrame({
    "Epoch": list(range(1, EPOCHS + 1)),
    "Test_Intensity_Acc": test_i_list,
    "Test_Angle_Acc": test_a_list,
    "Test_Freq_Acc": test_f_list
    })
    test_metrics.to_csv("test_metrics.csv", index=False)
    print("测试过程数据已保存为 test_metrics.csv")


if __name__ == "__main__":
    train_model()
