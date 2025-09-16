import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def calculate_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40)
    spatial_lags = []
    
    for i in range(len(coords)):
        dist, idx = tree.query([coords[i]], k=k+1)
        neighbor_indices = idx[0][1:]
        spatial_lag = df.iloc[neighbor_indices][target_col].mean()
        spatial_lags.append(spatial_lag)
    
    return spatial_lags

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'block_variety']
    
    target = 'redvine_count_2024'
    
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.06.04'] - df['EVI_2023.06.01']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.06.04'] - df['CanopyArea_2023.06.01']
    
    df['spatial_lag_redvine_2023'] = calculate_spatial_lag(df, 'redvine_count_2023')
    df['spatial_lag_redvine_2022'] = calculate_spatial_lag(df, 'redvine_count_2022')
    
    features.extend(['EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024', 
                    'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022'])
    
    df = df.dropna(subset=features + [target])
    
    le = LabelEncoder()
    df['block_variety_encoded'] = le.fit_transform(df['block_variety'])
    features.remove('block_variety')
    features.append('block_variety_encoded')
    
    X = df[features].copy()
    y = (df[target] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        n_jobs=-1,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
        metric=autosklearn.metrics.balanced_accuracy
    )
    
    automl.fit(X_train, y_train, dataset_name='redvine_classification')
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nAuto-sklearn models:")
    print(automl.show_models())
    print("\nAuto-sklearn leaderboard:")
    print(automl.leaderboard())

if __name__ == "__main__":
    main()