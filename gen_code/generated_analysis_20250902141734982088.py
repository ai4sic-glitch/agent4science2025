import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def compute_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, metric='haversine')
    distances, indices = tree.query(coords, k=k+1)
    
    spatial_lag = []
    for i in range(len(df)):
        neighbor_indices = indices[i][1:]
        spatial_lag.append(df[target_col].iloc[neighbor_indices].mean())
    
    return spatial_lag

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 'latitude',
        'EVI_2023.07.30', 'CanopyArea_2023.07.30', 'EVI_2022.08.04', 'CanopyArea_2022.08.04',
        'vineSpace', 'block_variety'
    ]
    
    target = 'redvine_count_2024'
    
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    df['EVI_2024.07.10_CanopyArea_2024.07.10_interaction'] = df['EVI_2024.07.10'] * df['CanopyArea_2024.07.10']
    df['EVI_2023.07.30_CanopyArea_2023.07.30_interaction'] = df['EVI_2023.07.30'] * df['CanopyArea_2023.07.30']
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023')
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022')
    
    le = LabelEncoder()
    df['block_variety_encoded'] = le.fit_transform(df['block_variety'].astype(str))
    
    all_features = features + [
        'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024',
        'EVI_2024.07.10_CanopyArea_2024.07.10_interaction',
        'EVI_2023.07.30_CanopyArea_2023.07.30_interaction',
        'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022',
        'block_variety_encoded'
    ]
    
    df = df.dropna(subset=all_features + [target])
    
    X = df[all_features].copy()
    y = (df[target] > 0).astype(int)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        n_jobs=-1
    )
    
    automl.fit(X, y)
    
    y_pred = automl.predict(X)
    
    print("Features used:", all_features)
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()