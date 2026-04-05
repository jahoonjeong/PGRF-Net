import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from utils import create_windows_for_inference

def infer_scores(model, series, window_size, batch_size=256):
    model.eval()
    raw_scores = {'predictive':[], 'structural':[], 'contextual':[], 'spike':[]}
    
    X_inf, Y_inf = create_windows_for_inference(series, window_size)
    if X_inf.shape[0] == 0:
        return {} 

    inf_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_inf, Y_inf),
        batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_X, batch_Y in tqdm(inf_loader, desc="Inferring Scores", leave=False):
            if next(model.parameters()).is_cuda:
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()

            pred, dyn_mask, _, ctx_score, spike_score, _, _ = model(batch_X)

            # 1. Predictive Evidence Score (Reconstruction Error)
            raw_scores['predictive'].extend(((pred - batch_Y) ** 2).mean(dim=1).cpu().numpy())
            
            # 2. Structural Evidence Score (Change from base structural graph) 
            if model.is_base_mask_set.item():
                mask_diff = torch.linalg.norm(dyn_mask - model.base_structural_mask.unsqueeze(0), ord='fro', dim=(1, 2))
                raw_scores['structural'].extend(mask_diff.cpu().numpy()) 
            else: # Fallback if base mask wasn't set
                raw_scores['structural'].extend(np.zeros(batch_X.shape[0])) 

            # 3. Contextual Evidence Score
            raw_scores['contextual'].extend(ctx_score.cpu().numpy())
            
            # 4. Spike Evidence Score
            raw_scores['spike'].extend(spike_score.cpu().numpy())
    
    # Normalize and pad scores to match original series length
    final_scores = {}
    for score_type, scores in raw_scores.items():
        new_key = f"{score_type}_scores"
        if not scores:
            final_scores[new_key] = np.zeros(len(series))
            continue
            
        scaler = MinMaxScaler()
        scores_array = np.array(scores)
        
        # Handle potential inf/nan values before scaling
        scores_array[np.isinf(scores_array)] = np.nan
        if np.all(np.isnan(scores_array)):
            scaled_scores = np.zeros_like(scores_array)
        else:
            scores_array[np.isnan(scores_array)] = np.nanmean(scores_array)
            scaled_scores = scaler.fit_transform(scores_array.reshape(-1, 1)).flatten()
        
        padded = np.zeros(len(series))
        padded[window_size : window_size + len(scaled_scores)] = scaled_scores
        final_scores[new_key] = padded

    return final_scores