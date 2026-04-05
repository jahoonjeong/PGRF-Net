import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- Loss Function ---

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > 1:
            mse_per_sample = inputs.mean(dim=1)
        else:
            mse_per_sample = inputs

        if mse_per_sample.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)

        min_mse, max_mse = torch.min(mse_per_sample), torch.max(mse_per_sample)
        if (max_mse - min_mse).abs() < 1e-8:
            pt = torch.ones_like(mse_per_sample)
        else:
            pt = 1 - (mse_per_sample - min_mse) / (max_mse - min_mse)
        pt = pt.clamp(min=1e-8, max=1 - 1e-8)

        focal_term = (1 - pt)**self.gamma
        f_loss = focal_term * mse_per_sample

        at = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        f_loss = at * f_loss

        if self.reduction == 'mean':
            return torch.mean(f_loss)
        elif self.reduction == 'sum':
            return torch.sum(f_loss)
        else:
            return f_loss

# --- Data Processing ---

def create_windows(series: np.ndarray, labels: np.ndarray, window_size: int):
    X, Y, L = [], [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        Y.append(series[i + window_size])
        L.append(labels[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(Y), dtype=torch.float32), \
           torch.tensor(np.array(L), dtype=torch.float32)

def create_windows_for_inference(series: np.ndarray, window_size: int):
    X, Y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        Y.append(series[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(Y), dtype=torch.float32)

def apply_first_order_differencing(data: np.ndarray) -> np.ndarray:
    return np.diff(data, axis=0, prepend=data[0:1, :])

# --- Evaluation Metrics ---

def calc_point2point(predict: np.ndarray, actual: np.ndarray):
    """Calculates point-wise F1, precision, and recall."""
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return f1, precision, recall, TP, TN, FP, FN

def adjust_predicts(score: np.ndarray, label: np.ndarray, threshold: float, pred=None, calc_latency: bool = False):
    score, label = np.asarray(score), np.asarray(label)
    predict = score > threshold if pred is None else pred
    actual = label > 0.1
    anomaly_state, anomaly_count, latency = False, 0, 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]: break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return (predict, latency / (anomaly_count + 1e-4)) if calc_latency else predict

def calc_seq(score: np.ndarray, label: np.ndarray, threshold: float, calc_latency: bool = False):
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=False)
        return calc_point2point(predict, label)

def bf_search(score: np.ndarray, label: np.ndarray, start=None, end=None, step_num: int = 100, verbose: bool = False):
    start = np.min(score) if start is None else start
    end = np.max(score) if end is None else end
    search_step = (end - start) / step_num
    if search_step == 0:
        return calc_seq(score, label, start, calc_latency=True), start
        
    thresholds = np.arange(start, end, search_step) if search_step > 0 else [start]
    m = (-1.,) * 8
    m_t = 0.0
    
    pbar = tqdm(thresholds, desc="BF Search Threshold", disable=not verbose, leave=False)
    for threshold in pbar:
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
    return m, m_t

# --- Model Helper Function ---

def h_func(W: torch.Tensor) -> torch.Tensor:
    d = W.shape[0]
    W_prime = W * (1 - torch.eye(d, device=W.device))
    h = torch.trace(torch.linalg.matrix_exp(W_prime)) - d
    return h
