import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def compute_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40, metric='haversine')
    spatial_lag = np.zeros(len(df))
    
    for i, (lat, lon) in enumerate(coords):
        dist, idx = tree.query([[lat, lon]], k=k+1)
        neighbor_indices = idx[0][1:]  # Exclude self
        spatial_lag[i] = df.iloc[neighbor_indices][target_col].mean()
    
    return spatial_lag

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023')
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022')
    
    df['EVI_2024.07.10_CanopyArea_2024.07.10_interaction'] = df['EVI_2024.07.10'] * df['CanopyArea_2024.07.10']
    df['EVI_2023.07.30_CanopyArea_2023.07.30_interaction'] = df['EVI_2023.07.30'] * df['CanopyArea_2023.07.30']
    
    features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 'latitude',
        'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022',
        'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024',
        'block_variety', 'vineSpace',
        'EVI_2024.07.10_CanopyArea_2024.07.10_interaction',
        'EVI_2023.07.30_CanopyArea_2023.07.30_interaction'
    ]
    
    target = 'redvine_count_2024'
    df[target] = (df[target] > 0).astype(int)
    
    le = LabelEncoder()
    df['block_variety_encoded'] = le.fit_transform(df['block_variety'].astype(str))
    features.append('block_variety_encoded')
    features.remove('block_variety')
    
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features]
    y = df_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8}
    )
    
    automl.fit(X_train, y_train, dataset_name='redvine_classification')
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nAuto-sklearn leaderboard:")
    print(automl.leaderboard())

if __name__ == "__main__":
    main()