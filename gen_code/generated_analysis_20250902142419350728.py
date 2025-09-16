import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def compute_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40, metric='euclidean')
    spatial_lag = np.zeros(len(df))
    
    for i, (lat, lon) in enumerate(coords):
        dist, idx = tree.query([[lat, lon]], k=k+1)
        neighbor_indices = idx[0][1:]
        spatial_lag[i] = df.iloc[neighbor_indices][target_col].mean()
    
    return spatial_lag

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 'latitude',
        'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024',
        'block_variety', 'vineSpace', 'EVI_2024.07.10_CanopyArea_2024.07.10_interaction',
        'EVI_2023.07.30_CanopyArea_2023.07.30_interaction', 'EVI_2023.07.30',
        'CanopyArea_2023.07.30', 'EVI_2022.08.04', 'CanopyArea_2022.08.04'
    ]
    
    target = 'redvine_count_2024'
    
    df = df.dropna(subset=features + [target])
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023', k=5)
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022', k=5)
    
    df['target_class'] = (df[target] > 0).astype(int)
    
    final_features = features + ['spatial_lag_redvine_2023', 'spatial_lag_redvine_2022']
    
    X = df[final_features]
    y = df['target_class']
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=180,
        memory_limit=4096,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8},
        ensemble_size=1,
        initial_configurations_via_metalearning=0
    )
    
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", final_features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAuto-sklearn models:", automl.show_models())

if __name__ == "__main__":
    main()