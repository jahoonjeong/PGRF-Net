import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# --- Import from project files ---
from model import PGRFNet
from training import train_model_stage1, train_model_stage2
from inference import infer_scores
from utils import create_windows, bf_search
from data_loader import load_dataset 

if __name__ == "__main__":
    # --- 1. Experiment Configuration ---
    DATASET_TO_RUN = 'SMAP' # 'MSL', 'SMAP', 'PSM', 'SMD', 'SWAT' 

    window_size = 60
    FIXED_ALPHA = 0.1 
    num_protos_all = 10

    train_params = {
        'epochs_stage1': 50, 'lr': 1e-4, 'batch_size': 128, 'patience_stage1': 10,
        'epochs_stage2': 20, 'lr_stage2': 1e-4, 'patience_stage2': 5, 
        'focal_gamma': 2.0, 'focal_alpha': 0.5, 'anomaly_weight': 10.0,
        'mask_reg_weight': 0.01, 'mask_diff_weight': 0.001, 'acyclic_penalty_weight': 1e-4,
        'lambda1': 1e-3, 'sparsity_lambda': 1e-3, 'context_loss_weight_stage1': 0.01,
        'spike_loss_weight_stage1': 0.01,
        'pseudo_normal_percent': 0.25, 'gate_normal_suppress_weight': 0.1,
        'gate_entropy_weight': 0.001, 'context_loss_weight_stage2': 0.2, 
        'spike_loss_weight_stage2': 0.2,
    }

    # --- 2. Data Path Configuration ---
    DATASET_PATHS = {
        'SMAP': "/root/Data/SMAP",
        'MSL': "/root/Data/MSL",
        'SMD': "/root/Data/SMD/ServerMachineDataset",
        'PSM': "/root/Data/PSM",
        'SWAT': "/root/Data/SWaT"
    }

    # --- 3. Load Data using Unified Loader ---
    try:
        processed_data_list = load_dataset(DATASET_TO_RUN, DATASET_PATHS)
        if not processed_data_list:
            print(f"No data loaded for {DATASET_TO_RUN}. Please check paths and data files. Exiting.")
            exit()
    except ValueError as e:
        print(e)
        exit()

    all_results_list = []

    # --- 4. Per-Channel/Dataset Training and Evaluation Loop ---
    for data_unit in processed_data_list:
        data_id = data_unit['data_id']
        scaled_train_data = data_unit['scaled_train_data']
        scaled_test_data = data_unit['scaled_test_data']
        test_labels = data_unit['test_labels']
        
        print(f"\n{'='*80}\n--- Processing: {data_id} (from {DATASET_TO_RUN} dataset) ---\n{'='*80}")
        
        # --- 4a. Create Training Windows ---
        train_labels = np.zeros(len(scaled_train_data))
        X_train, Y_train, L_train = create_windows(scaled_train_data, train_labels, window_size)
        
        if X_train.shape[0] == 0:
            print(f"Not enough training data to create windows for {data_id}. Skipping.")
            continue
        
        N_VARS = X_train.shape[2]

        # --- 4b. Initialize and Train Model ---
        print(f"[{data_id}] Initializing and training a new model...")
        model = PGRFNet(
            num_vars=N_VARS, 
            seq_len=window_size,
            num_protos=num_protos_all,
            num_context_protos=num_protos_all,
            num_spike_protos=num_protos_all
        )
        if torch.cuda.is_available(): model.cuda()

        train_model_stage1(model, X_train, Y_train, L_train, **train_params)
        train_model_stage2(model, X_train, Y_train, L_train, **train_params)
        
        # --- 4c. Infer Anomaly Scores ---
        print(f"[{data_id}] Inferring anomaly scores...")
        scores_dict = infer_scores(model, scaled_test_data, window_size)
        if not scores_dict:
            print(f"[{data_id}] Inference resulted in empty scores. Skipping.")
            continue

        s_pred = scores_dict['predictive_scores']
        s_structural = scores_dict['structural_scores']
        s_ctx = scores_dict['contextual_scores']
        s_spike = scores_dict['spike_scores']
        
        expl_score_components = np.stack([s_structural, s_ctx, s_spike], axis=-1)
        expl_score_sum = expl_score_components.sum(axis=1, keepdims=True) + 1e-8
        expl_score_norm = expl_score_components / expl_score_sum
        expl_score = np.sum(expl_score_norm * expl_score_components, axis=1)

        # --- 4d. Evaluate with Fixed Alpha ---
        print(f"[{data_id}] Evaluating with fixed alpha = {FIXED_ALPHA}...")
        combined_scores = (1 - FIXED_ALPHA) * s_pred + FIXED_ALPHA * expl_score
        
        if np.sum(test_labels) > 0:
            best_metrics, _ = bf_search(combined_scores, test_labels, verbose=False)
            result = {'F1': best_metrics[0], 'P': best_metrics[1], 'R': best_metrics[2]}
            if len(np.unique(test_labels)) > 1:
                result['AUROC'] = roc_auc_score(test_labels, combined_scores)
                p, r, _ = precision_recall_curve(test_labels, combined_scores)
                result['AUCPR'] = auc(r, p)
            else:
                result['AUROC'], result['AUCPR'] = np.nan, np.nan
        else:
            result = {'F1': np.nan, 'P': np.nan, 'R': np.nan, 'AUROC': np.nan, 'AUCPR': np.nan}
        
        result['data_id'] = data_id
        all_results_list.append(result)

    # --- 5. Final Result Aggregation and Reporting ---
    if all_results_list:
        results_df = pd.DataFrame(all_results_list)
        final_performance_metrics = results_df.mean(numeric_only=True)
        print(f"\n\n{'='*80}\n🏆 Final Report: Overall Performance for {DATASET_TO_RUN}\n{'='*80}")
        print(f"Using Fixed Alpha: {FIXED_ALPHA}")
        print("\n--- Average Performance Metrics ---")
        print(final_performance_metrics.to_string())
    else:
        print("No data was processed to produce final results.")
