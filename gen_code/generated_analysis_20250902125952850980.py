import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def create_spatial_lag_features(df, k_neighbors=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40, metric='haversine')
    
    spatial_lag_features = {}
    
    for year in [2022, 2023]:
        col_name = f'redvine_count_{year}'
        if col_name in df.columns:
            distances, indices = tree.query(coords, k=k_neighbors+1)
            neighbor_counts = df[col_name].values[indices[:, 1:]]
            spatial_lag = np.mean(neighbor_counts, axis=1)
            spatial_lag_features[f'spatial_lag_redvine_{year}'] = spatial_lag
    
    return spatial_lag_features

def preprocess_data(df):
    df = df.copy()
    
    spatial_lags = create_spatial_lag_features(df)
    for col_name, values in spatial_lags.items():
        df[col_name] = values
    
    suggested_features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10',
        'Presence_2024.06.04', 'Presence_2024.07.10',
        'longitude', 'latitude', 'EVI_2023.07.30',
        'CanopyArea_2023.07.30', 'Presence_2023.07.30',
        'EVI_2022.08.04', 'CanopyArea_2022.08.04',
        'Presence_2022.08.04', 'EVI_delta_2023_to_2024',
        'CanopyArea_delta_2023_to_2024', 'EVI_delta_2022_to_2023',
        'CanopyArea_delta_2022_to_2023', 'spatial_lag_redvine_2023',
        'spatial_lag_redvine_2022'
    ]
    
    available_features = [f for f in suggested_features if f in df.columns]
    
    if 'EVI_delta_2023_to_2024' not in df.columns and 'EVI_2023.07.30' in df.columns and 'EVI_2024.06.04' in df.columns:
        df['EVI_delta_2023_to_2024'] = df['EVI_2024.06.04'] - df['EVI_2023.07.30']
        available_features.append('EVI_delta_2023_to_2024')
    
    if 'CanopyArea_delta_2023_to_2024' not in df.columns and 'CanopyArea_2023.07.30' in df.columns and 'CanopyArea_2024.06.04' in df.columns:
        df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.06.04'] - df['CanopyArea_2023.07.30']
        available_features.append('CanopyArea_delta_2023_to_2024')
    
    if 'EVI_delta_2022_to_2023' not in df.columns and 'EVI_2022.08.04' in df.columns and 'EVI_2023.07.30' in df.columns:
        df['EVI_delta_2022_to_2023'] = df['EVI_2023.07.30'] - df['EVI_2022.08.04']
        available_features.append('EVI_delta_2022_to_2023')
    
    if 'CanopyArea_delta_2022_to_2023' not in df.columns and 'CanopyArea_2022.08.04' in df.columns and 'CanopyArea_2023.07.30' in df.columns:
        df['CanopyArea_delta_2022_to_2023'] = df['CanopyArea_2023.07.30'] - df['CanopyArea_2022.08.04']
        available_features.append('CanopyArea_delta_2022_to_2023')
    
    interaction_terms = []
    for spatial_col in ['spatial_lag_redvine_2022', 'spatial_lag_redvine_2023']:
        for delta_col in ['EVI_delta_2022_to_2023', 'EVI_delta_2023_to_2024', 
                         'CanopyArea_delta_2022_to_2023', 'CanopyArea_delta_2023_to_2024']:
            if spatial_col in df.columns and delta_col in df.columns:
                interaction_name = f'{spatial_col}_{delta_col}_interaction'
                df[interaction_name] = df[spatial_col] * df[delta_col]
                interaction_terms.append(interaction_name)
    
    available_features.extend(interaction_terms)
    
    X = df[available_features].copy()
    y = (df['redvine_count_2024'] > 0).astype(int)
    
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
            except:
                X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, available_features

def main():
    try:
        df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return
    
    X, y, features_used = preprocess_data(df)
    
    print(f"Features used: {features_used}")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    class_weights = {0: 1.0, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}
    print(f"Class weights: {class_weights}")
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        metric=autosklearn.metrics.balanced_accuracy,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )
    
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Auto-sklearn Leaderboard ===")
    print(automl.leaderboard())
    
    print("\n=== Best Model ===")
    print(automl.show_models())

if __name__ == "__main__":
    main()