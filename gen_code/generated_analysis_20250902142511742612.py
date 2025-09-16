import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import autosklearn.classification
from sklearn.neighbors import BallTree

def compute_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, metric='euclidean')
    distances, indices = tree.query(coords, k=k+1)
    
    spatial_lag = []
    for i in range(len(df)):
        neighbor_indices = indices[i][1:]
        spatial_lag.append(df[target_col].iloc[neighbor_indices].mean())
    
    return spatial_lag

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    target = 'redvine_count_2024'
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'vineSpace', 'block_variety', 'EVI_2023.07.30', 
                'CanopyArea_2023.07.30', 'EVI_2022.08.04', 'CanopyArea_2022.08.04']
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023', k=5)
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022', k=5)
    
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    
    df['EVI_2024.07.10_CanopyArea_2024.07.10_interaction'] = df['EVI_2024.07.10'] * df['CanopyArea_2024.07.10']
    df['EVI_2023.07.30_CanopyArea_2023.07.30_interaction'] = df['EVI_2023.07.30'] * df['CanopyArea_2023.07.30']
    
    features.extend(['spatial_lag_redvine_2023', 'spatial_lag_redvine_2022', 
                    'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024',
                    'EVI_2024.07.10_CanopyArea_2024.07.10_interaction', 
                    'EVI_2023.07.30_CanopyArea_2023.07.30_interaction'])
    
    df = df.dropna(subset=features + [target])
    
    df['block_variety'] = df['block_variety'].astype('category').cat.codes
    
    X = df[features]
    y = (df[target] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=180,
        memory_limit=4096,
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8}
    )
    
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAuto-sklearn models:", automl.show_models())

if __name__ == "__main__":
    main()