import os 
import re
import glob
import random
import csv        # 导出CSV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# ========= 从 predict_utils 导入所需函数/模型 =========
from predict_utils import (
    wavelet_denoise,  # 小波去噪
    extract_features, # 特征提取
    MultiOutputModel, # 模型结构
    TARGET_LENGTH     # 与训练时保持一致的特征长度
)

# 新的测试集文件夹
TEST_DATA_ROOT = r"D:\石阳\博士工作\科研\周报\65-shiyang-20250308\gammry-20250313\night-2nd test\全部条件切分周期-auto-two stages"
MODEL_PATH = "pytorch_ecg_model.pth"
BATCH_SIZE = 64

class ECGDataset(Dataset):
    """
    扫描形如 'peaks_auto_(光强)%_(角度)_(频率)Hz' 的子文件夹，
    本例中用于测试整个文件夹的数据，不进行拆分。
    (已去掉原先“if len(denoised) > 2000: denoised = denoised[1000:-1000]”逻辑)
    """
    def __init__(self, root):
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
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if "Current_A" not in df.columns:
                    continue

                signal = df["Current_A"].values
                # 1) 小波去噪
                denoised = wavelet_denoise(signal)
                # 2) 这里不再截取 [1000:-1000]
                # 3) 特征提取
                feats = extract_features(denoised, TARGET_LENGTH)

                # 记录 CSV 文件路径，以便输出预测错误时使用
                all_data.append((feats, (intensity_val, angle_val, freq_val), csv_file))

        self.samples = all_data
        print(f"Test dataset size={len(all_data)}")

        # 收集标签并构建映射
        self.intensity_list = sorted(set(x[1][0] for x in self.samples))
        self.angle_list     = sorted(set(x[1][1] for x in self.samples))
        self.freq_list      = sorted(set(x[1][2] for x in self.samples))

        self.i2idx = {val: idx for idx, val in enumerate(self.intensity_list)}
        self.a2idx = {val: idx for idx, val in enumerate(self.angle_list)}
        self.f2idx = {val: idx for idx, val in enumerate(self.freq_list)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回: (feats, (i_idx, a_idx, f_idx), csv_path)
        """
        feats, (i_val, a_val, f_val), csv_path = self.samples[idx]
        feats = feats[:TARGET_LENGTH + 2].reshape(-1, 1)
        
        i_idx = self.i2idx[i_val]
        a_idx = self.a2idx[a_val]
        f_idx = self.f2idx[f_val]
        
        return feats, (i_idx, a_idx, f_idx), csv_path

def collate_fn(batch):
    xs = []
    ys = []
    paths = []
    for (feats, label_tuple, csv_path) in batch:
        xs.append(torch.tensor(feats, dtype=torch.float32))
        ys.append(torch.tensor(label_tuple, dtype=torch.long))
        paths.append(csv_path)
    x_tensor = torch.stack(xs, dim=0)
    y_tensor = torch.stack(ys, dim=0)
    return x_tensor, y_tensor, paths

def eval_model(model, loader, device):
    model.eval()
    total_loss = 0
    correct_i = correct_a = correct_f = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, Y, _ in loader:
            X = X.to(device)
            Y = Y.to(device)
            yi = Y[:, 0]
            ya = Y[:, 1]
            yf = Y[:, 2]

            out_i, out_a, out_f = model(X)

            # 三分支简单相加做总loss(可改成加权)
            li = criterion(out_i, yi)
            la = criterion(out_a, ya)
            lf = criterion(out_f, yf)
            loss = li + la + lf

            bs = X.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            pi = out_i.argmax(dim=1)
            pa = out_a.argmax(dim=1)
            pf = out_f.argmax(dim=1)
            correct_i += (pi == yi).sum().item()
            correct_a += (pa == ya).sum().item()
            correct_f += (pf == yf).sum().item()

    avg_loss = total_loss / total_samples if total_samples else 0
    acc_intensity = correct_i / total_samples if total_samples else 0
    acc_angle = correct_a / total_samples if total_samples else 0
    acc_freq = correct_f / total_samples if total_samples else 0

    return avg_loss, acc_intensity, acc_angle, acc_freq

def plot_confusion_matrix(y_true, y_pred, labels, title):
    """
    根据真实标签和预测标签绘制归一化后的混淆矩阵（按行归一化，每行总和为1），
    并保存图像到当前脚本目录中。
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    # 归一化：每一行除以该行和 (若行和为0则保持0)
    cm_normalized = np.zeros_like(cm, dtype=np.float32)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_normalized[i] = cm[i] / row_sum
    
    # 保存归一化后的混淆矩阵为 CSV
    csv_filename = title.lower().replace(" ", "_") + "_matrix.csv"
    df_cm = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    df_cm.to_csv(csv_filename, encoding="utf-8-sig")
    print(f"[Info] 混淆矩阵数据已保存为 {csv_filename}")

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=400)
    print(f"[Info] 混淆矩阵已保存为 {filename}")
    plt.close()

if __name__ == "__main__":
    # 读取测试集数据
    test_dataset = ECGDataset(TEST_DATA_ROOT)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] Using device:", device)

    model = MultiOutputModel().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[Info] Model loaded from", MODEL_PATH)
    else:
        print("[Error] Model file not found!")
        exit(1)

    # 评估测试集
    avg_loss, acc_i, acc_a, acc_f = eval_model(model, test_loader, device)
    print("==== Test Accuracy ====")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Intensity Accuracy: {acc_i:.3f}")
    print(f"Angle Accuracy: {acc_a:.3f}")
    print(f"Frequency Accuracy: {acc_f:.3f}")

    # ========== 生成混淆矩阵并保存图像 ========== 
    all_i_true, all_a_true, all_f_true = [], [], []
    all_i_pred, all_a_pred, all_f_pred = [], [], []
    all_paths = []

    model.eval()
    with torch.no_grad():
        for X, Y, paths in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            yi = Y[:, 0]
            ya = Y[:, 1]
            yf = Y[:, 2]

            out_i, out_a, out_f = model(X)
            pi = out_i.argmax(dim=1)
            pa = out_a.argmax(dim=1)
            pf = out_f.argmax(dim=1)

            all_i_true.append(yi.cpu().numpy())
            all_a_true.append(ya.cpu().numpy())
            all_f_true.append(yf.cpu().numpy())
            all_i_pred.append(pi.cpu().numpy())
            all_a_pred.append(pa.cpu().numpy())
            all_f_pred.append(pf.cpu().numpy())

            all_paths.extend(paths)

    all_i_true = np.concatenate(all_i_true)
    all_a_true = np.concatenate(all_a_true)
    all_f_true = np.concatenate(all_f_true)
    all_i_pred = np.concatenate(all_i_pred)
    all_a_pred = np.concatenate(all_a_pred)
    all_f_pred = np.concatenate(all_f_pred)

    # 分别绘制三个分支的混淆矩阵（归一化后显示正确率）
    plot_confusion_matrix(all_i_true, all_i_pred,
                          test_dataset.intensity_list,
                          "Confusion Matrix - Intensity")
    plot_confusion_matrix(all_a_true, all_a_pred,
                          test_dataset.angle_list,
                          "Confusion Matrix - Angle")
    plot_confusion_matrix(all_f_true, all_f_pred,
                          test_dataset.freq_list,
                          "Confusion Matrix - Frequency")

    # 新增功能：打印预测错误的样本到 CSV (UTF-8-BOM)
    wrong_list = []
    intensity_list = test_dataset.intensity_list
    angle_list     = test_dataset.angle_list
    freq_list      = test_dataset.freq_list

    for idx in range(len(all_i_true)):
        if (all_i_true[idx] != all_i_pred[idx]) \
           or (all_a_true[idx] != all_a_pred[idx]) \
           or (all_f_true[idx] != all_f_pred[idx]):
            real_i_val = intensity_list[ all_i_true[idx] ]
            real_a_val = angle_list[ all_a_true[idx] ]
            real_f_val = freq_list[ all_f_true[idx] ]
            pred_i_val = intensity_list[ all_i_pred[idx] ]
            pred_a_val = angle_list[ all_a_pred[idx] ]
            pred_f_val = freq_list[ all_f_pred[idx] ]

            sample_path = all_paths[idx]
            wrong_list.append([
                sample_path,
                real_i_val, pred_i_val,
                real_a_val, pred_a_val,
                real_f_val, pred_f_val
            ])

    if wrong_list:
        csv_name = "wrong_predictions.csv"
        # 使用 utf-8-sig, 确保含中文路径时 Excel 不会乱码
        with open(csv_name, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_path",
                "real_intensity", "pred_intensity",
                "real_angle",     "pred_angle",
                "real_freq",      "pred_freq"
            ])
            writer.writerows(wrong_list)
        print(f"[Info] 预测错误的样本已输出到 {csv_name} (UTF-8 BOM)，共 {len(wrong_list)} 条。")
    else:
        print("[Info] 所有样本预测正确，无错误样本可导出。")
