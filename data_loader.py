import os
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from utils import apply_first_order_differencing

def _preprocess_and_scale(train_data, test_data):
    # 1. Concatenate for consistent processing
    full_data = np.concatenate((train_data, test_data), axis=0)
    
    # 2. Apply first-order differencing
    full_data_diff = apply_first_order_differencing(full_data)
    
    # 3. Scale the data
    scaler = StandardScaler()
    scaled_full_data = scaler.fit_transform(full_data_diff)
    
    # 4. Split back into train and test sets
    train_end_idx = len(train_data)
    scaled_train_data = scaled_full_data[:train_end_idx]
    scaled_test_data = scaled_full_data[train_end_idx:]
    
    return scaled_train_data, scaled_test_data

def _load_smap_msl(base_path, dataset_name):
    print(f"Loading {dataset_name} dataset from: {base_path}")
    data_list = []
    
    anomalies_table_path = os.path.join(base_path, "labeled_anomalies.csv")
    anomalies_df = pd.read_csv(anomalies_table_path)
    data_ids = anomalies_df[anomalies_df['spacecraft'] == dataset_name]['chan_id'].tolist()
    
    for data_id in data_ids:
        train_file = os.path.join(base_path, 'train', f'{data_id}.npy')
        test_file = os.path.join(base_path, 'test', f'{data_id}.npy')
        
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Data files for {data_id} not found. Skipping.")
            continue
            
        train_data = np.load(train_file).astype(np.float32)
        test_data = np.load(test_file).astype(np.float32)
        if train_data.ndim == 1: train_data = train_data.reshape(-1, 1)
        if test_data.ndim == 1: test_data = test_data.reshape(-1, 1)

        test_labels = np.zeros(len(test_data), dtype=int)
        anomaly_sequences_str = anomalies_df[anomalies_df['chan_id'] == data_id]['anomaly_sequences'].values[0]
        if anomaly_sequences_str != '[]':
            anomaly_sequences = ast.literal_eval(anomaly_sequences_str)
            for start, end in anomaly_sequences:
                test_labels[start:end] = 1
        
        data_list.append({
            'data_id': data_id,
            'train_data': train_data,
            'test_data': test_data,
            'test_labels': test_labels
        })
    return data_list

def _load_smd(base_path):
    print(f"Loading SMD dataset from: {base_path}")
    data_list = []
    train_dir = os.path.join(base_path, 'train')
    machine_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.txt')])

    for filename in machine_files:
        machine_id = filename.replace('.txt', '')
        train_path = os.path.join(base_path, 'train', filename)
        test_path = os.path.join(base_path, 'test', filename)
        label_path = os.path.join(base_path, 'test_label', filename)

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            continue

        train_data = pd.read_csv(train_path, header=None, delimiter=',').interpolate(method='linear', limit_direction='both').values.astype(np.float32)
        test_data = pd.read_csv(test_path, header=None, delimiter=',').values.astype(np.float32)
        test_labels = pd.read_csv(label_path, header=None, delimiter=',').values.flatten().astype(np.int32)
        
        data_list.append({
            'data_id': machine_id,
            'train_data': train_data,
            'test_data': test_data,
            'test_labels': test_labels
        })
    return data_list

def _load_psm(base_path):
    print(f"Loading PSM dataset from: {base_path}")
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    test_label_df = pd.read_csv(os.path.join(base_path, 'test_label.csv'))

    train_data = train_df.drop(columns=['timestamp_(min)']).interpolate(method='linear', limit_direction='both').values.astype(np.float32)
    test_data = test_df.drop(columns=['timestamp_(min)']).values.astype(np.float32)
    test_labels = test_label_df['label'].values.astype(np.int32)
    
    return [{
        'data_id': 'psm',
        'train_data': train_data,
        'test_data': test_data,
        'test_labels': test_labels
    }]

def _load_swat(base_path):
    print(f"Loading SWaT dataset from: {base_path}")
    normal_df = pd.read_excel(os.path.join(base_path, 'SWaT_Dataset_Normal_v0.xlsx'), header=1)
    attack_df = pd.read_excel(os.path.join(base_path, 'SWaT_Dataset_Attack_v0.xlsx'), header=1)
    
    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()
    
    data_columns = normal_df.columns.drop(['Timestamp', 'Normal/Attack'], errors='ignore')
    
    train_data = normal_df[data_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
    test_data = attack_df[data_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
    test_labels = (attack_df['Normal/Attack'] == 'Attack').astype(int).values
    
    return [{
        'data_id': 'swat',
        'train_data': train_data,
        'test_data': test_data,
        'test_labels': test_labels
    }]

def load_dataset(dataset_name, base_paths):
    dataset_name = dataset_name.upper()
    if dataset_name not in base_paths:
        raise ValueError(f"Dataset '{dataset_name}' is not supported or its path is not defined.")

    base_path = base_paths[dataset_name]
    
    # Step 1: Load raw data based on dataset type
    if dataset_name in ['SMAP', 'MSL']:
        raw_data_list = _load_smap_msl(base_path, dataset_name)
    elif dataset_name == 'SMD':
        raw_data_list = _load_smd(base_path)
    elif dataset_name == 'PSM':
        raw_data_list = _load_psm(base_path)
    elif dataset_name == 'SWAT':
        raw_data_list = _load_swat(base_path)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    # Step 2: Apply common preprocessing to each data chunk
    processed_list = []
    for data_unit in raw_data_list:
        scaled_train, scaled_test = _preprocess_and_scale(data_unit['train_data'], data_unit['test_data'])
        processed_list.append({
            'data_id': data_unit['data_id'],
            'scaled_train_data': scaled_train,
            'scaled_test_data': scaled_test,
            'test_labels': data_unit['test_labels']
        })
        
    return processed_list
