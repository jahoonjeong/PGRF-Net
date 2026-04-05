import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import FocalLoss, h_func

class EarlyStopping:
    def __init__(self, patience: int = 10, verbose: bool = False, path: str = 'checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model_stage1(model, X_train, Y_train, L_train, **train_params):
    focal_loss_fn = FocalLoss(gamma=train_params.get('focal_gamma', 2.0), alpha=train_params.get('focal_alpha', 0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.get('lr', 1e-3))
    
    num_train_samples = int(len(X_train) * 0.8)
    X_train_split, Y_train_split, L_train_split = X_train[:num_train_samples], Y_train[:num_train_samples], L_train[:num_train_samples]
    X_val_split, Y_val_split, L_val_split = X_train[num_train_samples:], Y_train[num_train_samples:], L_train[num_train_samples:]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_split, Y_train_split, L_train_split),
        batch_size=train_params.get('batch_size', 256), shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_split, Y_val_split, L_val_split),
        batch_size=train_params.get('batch_size', 256), shuffle=False
    )

    early_stopping = EarlyStopping(patience=train_params.get('patience_stage1', 10), verbose=False, path='checkpoint_stage1.pt')
    model.is_base_mask_set.fill_(False)

    print("--- Starting Stage 1 Training: Multi-faceted Evidence Extractor ---")
    for epoch in tqdm(range(train_params.get('epochs_stage1', 50)), desc="Stage 1 Epochs", leave=False):
        model.train()
        for batch_X, batch_Y, batch_L in train_loader:
            if next(model.parameters()).is_cuda:
                batch_X, batch_Y, batch_L = batch_X.cuda(), batch_Y.cuda(), batch_L.cuda()

            pred, dyn_mask, structural_mask, ctx_score, spike_score, _, _ = model(batch_X)

            # --- Loss Calculation ---
            mse_for_focal = F.mse_loss(pred, batch_Y, reduction='none')
            weights_mse = torch.ones_like(batch_L).float()
            weights_mse[batch_L == 1] = train_params.get('anomaly_weight', 10.0)
            
            loss_pred = (focal_loss_fn(mse_for_focal, batch_L) * weights_mse).mean()
            loss_mask_reg = dyn_mask.abs().mean()
            loss_mask_diff = torch.linalg.norm(structural_mask - model.base_structural_mask, ord='fro').mean() if model.is_base_mask_set.item() else 0.0
            loss_acyclic = h_func(structural_mask) 
            loss_l1 = torch.linalg.norm(structural_mask, ord=1) 
            loss_sparsity = torch.norm(model.proto_bank.prototype_masks, p=1)
            loss_ctx = ctx_score.mean()
            loss_spike = spike_score.mean()
            loss = (loss_pred +
                    train_params.get('mask_reg_weight', 0.01) * loss_mask_reg +
                    train_params.get('mask_diff_weight', 0.001) * loss_mask_diff +
                    train_params.get('acyclic_penalty_weight', 0.0001) * loss_acyclic +
                    train_params.get('lambda1', 0.001) * loss_l1 +
                    train_params.get('sparsity_lambda', 0.001) * loss_sparsity +
                    train_params.get('context_loss_weight_stage1', 0.01) * loss_ctx +
                    train_params.get('spike_loss_weight_stage1', 0.01) * loss_spike)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_Y_val, _ in val_loader:
                if next(model.parameters()).is_cuda:
                    batch_X_val, batch_Y_val = batch_X_val.cuda(), batch_Y_val.cuda()
                pred_val, *_ = model(batch_X_val)
                total_val_loss += F.mse_loss(pred_val, batch_Y_val).item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered for Stage 1.")
            break

    model.load_state_dict(torch.load(early_stopping.path))
    model.eval()

def train_model_stage2(model, X_train, Y_train, L_train, **train_params):
    for param in model.parameters(): param.requires_grad = False
    for param in model.gate_controller.parameters(): param.requires_grad = True
    for param in model.context_proto_bank.parameters(): param.requires_grad = True
    for param in model.spike_proto_bank.parameters(): param.requires_grad = True

    gating_params = list(model.gate_controller.parameters()) + \
                    list(model.context_proto_bank.parameters()) + \
                    list(model.spike_proto_bank.parameters())
    
    optimizer_stage2 = torch.optim.Adam(gating_params, lr=train_params.get('lr_stage2', 1e-4))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train, L_train),
        batch_size=train_params.get('batch_size', 256), shuffle=True
    )
    
    early_stopping_stage2 = EarlyStopping(patience=train_params.get('patience_stage2', 5), verbose=False, path='checkpoint_stage2.pt')

    print("--- Starting Stage 2 Training: Gated Evidence Fusion Network ---")
    for epoch in tqdm(range(train_params.get('epochs_stage2', 20)), desc="Stage 2 Epochs", leave=False):
        model.train()
        total_epoch_loss = 0
        for batch_X, batch_Y, _ in train_loader:
            if next(model.parameters()).is_cuda:
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()

            pred, _, _, ctx_score, spike_score, _, gate_weights = model(batch_X)
            
            # --- Loss Calculation for Gating ---
            mse_for_pseudo = F.mse_loss(pred, batch_Y, reduction='none').mean(dim=1)
            
            # Identify pseudo-normal samples (lowest reconstruction error)
            num_pseudo_normal = int(batch_X.shape[0] * train_params.get('pseudo_normal_percent', 0.2))
            _, sorted_indices = torch.sort(mse_for_pseudo)
            pseudo_normal_indices = sorted_indices[:num_pseudo_normal]

            loss_gate_normal_suppress = (gate_weights[pseudo_normal_indices, 1:].sum(dim=1)).mean() if len(pseudo_normal_indices) > 0 else 0.0
            loss_gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
            loss_ctx = ctx_score.mean()
            loss_spike = spike_score.mean()

            loss_stage2 = (
                train_params.get('context_loss_weight_stage2', 0.1) * loss_ctx + 
                train_params.get('spike_loss_weight_stage2', 0.1) * loss_spike +
                train_params.get('gate_normal_suppress_weight', 0.1) * loss_gate_normal_suppress +
                train_params.get('gate_entropy_weight', 0.001) * loss_gate_entropy
            )
            
            optimizer_stage2.zero_grad()
            loss_stage2.backward()
            optimizer_stage2.step()
            total_epoch_loss += loss_stage2.item()

        # Use training loss for early stopping in this stage as it's unsupervised
        avg_epoch_loss = total_epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        early_stopping_stage2(avg_epoch_loss, model)
        if early_stopping_stage2.early_stop:
            print("Early stopping triggered for Stage 2.")
            break

    model.load_state_dict(torch.load(early_stopping_stage2.path))
    model.eval()