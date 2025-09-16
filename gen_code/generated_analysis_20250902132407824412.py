import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def compute_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40)
    spatial_lag = np.zeros(len(df))
    
    for i, (lat, lon) in enumerate(coords):
        dist, idx = tree.query([[lat, lon]], k=k+1)
        neighbor_indices = idx[0][1:]  # Exclude self
        spatial_lag[i] = df.iloc[neighbor_indices][target_col].mean()
    
    return spatial_lag

def main():
    try:
        df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    except FileNotFoundError:
        print("Dataset file not found. Please check the file path.")
        return
    
    required_features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'Presence_2024.06.04',
        'Presence_2024.07.10', 'longitude', 'latitude', 'EVI_2023.07.30',
        'CanopyArea_2023.07.30', 'Presence_2023.07.30', 'EVI_2022.08.04',
        'CanopyArea_2022.08.04', 'Presence_2022.08.04'
    ]
    
    for col in required_features:
        if col not in df.columns:
            print(f"Required column {col} not found in dataset")
            return
    
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    df['EVI_delta_2022_to_2023'] = df['EVI_2023.07.30'] - df['EVI_2022.08.04']
    df['CanopyArea_delta_2022_to_2023'] = df['CanopyArea_2023.07.30'] - df['CanopyArea_2022.08.04']
    
    df['weighted_sum_prior_counts'] = (
        0.1 * df['redvine_count_2021'] + 
        0.3 * df['redvine_count_2022'] + 
        0.6 * df['redvine_count_2023']
    )
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023')
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022')
    df['spatial_lag_weighted_sum'] = compute_spatial_lag(df, 'weighted_sum_prior_counts')
    
    final_features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 'latitude',
        'EVI_2023.07.30', 'CanopyArea_2023.07.30', 'EVI_2022.08.04',
        'CanopyArea_2022.08.04', 'EVI_delta_2023_to_2024',
        'CanopyArea_delta_2023_to_2024', 'EVI_delta_2022_to_2023',
        'CanopyArea_delta_2022_to_2023', 'weighted_sum_prior_counts',
        'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022',
        'spatial_lag_weighted_sum'
    ]
    
    X = df[final_features].copy()
    y = (df['redvine_count_2024'] > 0).astype(int)
    
    for col in X.select_dtypes(include=[object]).columns:
        X[col] = pd.to_numeric(X[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    
    X = X.fillna(X.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        n_jobs=-1,
        ensemble_size=1,
        initial_configurations_via_metalearning=0
    )
    
    automl.fit(X_train_scaled, y_train)
    
    y_pred = automl.predict(X_test_scaled)
    
    print("Features used:", final_features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nAuto-sklearn leaderboard:")
    print(automl.leaderboard())

if __name__ == "__main__":
    main()