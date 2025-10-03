# ================================
#  predict_utils.py
# ================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

##########################
# 配置（与训练脚本保持一致）
##########################
TARGET_LENGTH = 100
AUX_DIM = 2  # 增加的 mean/std 特征数

# 类别数（与训练脚本一致）
NUM_CLASS_INTENSITY = 3
NUM_CLASS_ANGLE     = 7
NUM_CLASS_FREQ      = 5

#############################################
# 自动统计频率样本数量及计算对应loss权重
#############################################
def compute_frequency_weights(dataset):
    """
    根据传入的dataset（例如ECGDataset）自动统计各频率类别样本数量，
    并计算归一化后的权重向量。权重计算方法：
       weight = 1 / (count)
       然后归一化，使得所有权重之和等于类别数（NUM_CLASS_FREQ）
    Args:
        dataset: 包含samples属性的数据集对象，每个样本格式：(features, (i_val, a_val, f_val))
    Returns:
        weights: torch.Tensor, shape=(NUM_CLASS_FREQ,)
    """
    freq_counts = {}
    for sample in dataset.samples:
        # sample[1] 为标签tuple，第三个元素为原始频率值
        f_val = sample[1][2]
        freq_counts[f_val] = freq_counts.get(f_val, 0) + 1

    # 根据ECGDataset中已有的频率标签顺序（已排序）
    freq_list = sorted(set(sample[1][2] for sample in dataset.samples))
    counts = [freq_counts[f] for f in freq_list]
    
    # 计算倒数并归一化：
    raw_weights = [1.0 / count for count in counts]
    total = sum(raw_weights)
    normalized_weights = [w / total * len(raw_weights) for w in raw_weights]
    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)
    
    print("自动统计频率样本数量：", freq_counts)
    print("计算得到的频率loss权重：", weights_tensor)
    
    return weights_tensor

#############################################
# 定义针对频率分支的加权交叉熵损失函数
#############################################
def frequency_loss(freq_logits, freq_target, freq_weights):
    """
    使用根据训练集中频率数据分布计算得来的权重进行加权交叉熵损失计算
    Args:
        freq_logits: 模型频率输出的logits (batch, NUM_CLASS_FREQ)
        freq_target: 真实标签 (batch,)
        freq_weights: 权重向量，由 compute_frequency_weights 得到
    Returns:
        加权后的交叉熵损失
    """
    return F.cross_entropy(freq_logits, freq_target, weight=freq_weights.to(freq_logits.device))

##########################
# 1) 小波去噪 + 特征提取
##########################
def wavelet_denoise(signal):
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet('db5').dec_len)
    level = min(9, max_level)
    coeffs = pywt.wavedec(data=signal, wavelet='db5', level=level)
    cA = coeffs[0]
    cDs = coeffs[1:]
    threshold = (np.median(np.abs(cDs[-1])) / 0.6745) * (np.sqrt(2 * np.log(len(cDs[-1]))))
    for i in range(len(cDs)):
        cDs[i] = pywt.threshold(cDs[i], threshold, mode="soft")
    rdata = pywt.waverec([cA] + cDs, "db5")
    return rdata

def downsample(data, n_points):
    if len(data) > n_points:
        original_indices = np.arange(len(data))
        interp_indices = np.linspace(0, len(data) - 1, n_points)
        sampled = np.interp(interp_indices, original_indices, data)
        return np.arange(n_points), sampled
    else:
        return np.arange(len(data)), data

def extract_features(signal, target_len=100):
    segment = signal
    if len(segment) >= target_len:
        _, sampled = downsample(segment, target_len)
        feat = np.array(sampled, dtype=np.float32)
    else:
        pad_len = target_len - len(segment)
        feat = np.pad(segment, (0, pad_len), mode='constant').astype(np.float32)

    mean = feat.mean()
    std = feat.std()
    if std > 1e-8:
        normed_feat = (feat - mean) / std
    else:
        normed_feat = feat - mean

    combined_feat = np.concatenate([normed_feat, [mean, std]]).astype(np.float32)
    return combined_feat

##########################
# 2) 模型定义（与训练脚本一致）
##########################
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        score = torch.tanh(self.W(x))
        attn_weights = self.v(score)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (attn_weights * x).sum(dim=1)
        return context

class MultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(4, 16, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(input_size=64, hidden_size=64,
                            batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim=128)

        self.fc1 = nn.Linear(128 + AUX_DIM, 128)  # 重点：增加mean/std通道
        self.dropout = nn.Dropout(0.2)

        self.out_int = nn.Linear(128, NUM_CLASS_INTENSITY)
        self.out_ang = nn.Linear(128, NUM_CLASS_ANGLE)
        self.out_fre = nn.Linear(128, NUM_CLASS_FREQ)

    def forward(self, x):
        aux = x[:, -2:, 0]  # 提取 mean/std 信息（batch, 2）
        x = x[:, :-2, :]    # 剩余部分送入卷积

        x = x.transpose(1, 2)
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = self.pool3(x)
        x = torch.relu(self.conv4(x))
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)

        # 拼接额外的统计信息
        context = torch.cat([context, aux], dim=1)

        fc = torch.relu(self.fc1(context))
        fc = self.dropout(fc)

        out_i = self.out_int(fc)
        out_a = self.out_ang(fc)
        out_f = self.out_fre(fc)
        return out_i, out_a, out_f

##########################
# 3) 统一预测类
##########################
class Predictor:
    def __init__(self, model_path="pytorch_ecg_model.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiOutputModel().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print(f"[Warn] 模型文件 {model_path} 不存在，无法加载。")

    def predict_signal(self, raw_signal):
        denoised = wavelet_denoise(raw_signal)
        feats = extract_features(denoised, TARGET_LENGTH)
        input_tensor = torch.from_numpy(feats).view(1, TARGET_LENGTH + 2, 1).to(self.device)
        with torch.no_grad():
            out_i, out_a, out_f = self.model(input_tensor)
            prob_i = F.softmax(out_i, dim=1)[0]
            prob_a = F.softmax(out_a, dim=1)[0]
            prob_f = F.softmax(out_f, dim=1)[0]

            pred_int = prob_i.argmax().item()
            pred_ang = prob_a.argmax().item()
            pred_freq = prob_f.argmax().item()

        return pred_int, pred_ang, pred_freq, \
               prob_i.cpu().numpy(), prob_a.cpu().numpy(), prob_f.cpu().numpy()
