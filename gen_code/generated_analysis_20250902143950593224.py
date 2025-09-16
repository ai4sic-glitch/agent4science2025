import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'EVI_2023.07.30', 'CanopyArea_2023.07.30', 
                'block_variety', 'vineSpace']
    
    target = 'redvine_count_2024'
    
    df['target_binary'] = (df[target] > 0).astype(int)
    
    le = LabelEncoder()
    df['block_variety_encoded'] = le.fit_transform(df['block_variety'].astype(str))
    features.append('block_variety_encoded')
    
    if 'spatial_lag_redvine_2023' not in df.columns:
        coords = df[['longitude', 'latitude']].values
        tree = BallTree(coords, leaf_size=40, metric='haversine')
        distances, indices = tree.query(coords, k=6)
        
        df['spatial_lag_redvine_2023'] = 0.0
        df['spatial_lag_redvine_2022'] = 0.0
        
        for i in range(len(df)):
            neighbor_indices = indices[i][1:]
            df.loc[df.index[i], 'spatial_lag_redvine_2023'] = df.iloc[neighbor_indices]['redvine_count_2023'].mean()
            df.loc[df.index[i], 'spatial_lag_redvine_2022'] = df.iloc[neighbor_indices]['redvine_count_2022'].mean()
    
    features.extend(['spatial_lag_redvine_2023', 'spatial_lag_redvine_2022'])
    
    if 'EVI_delta_2023_to_2024' not in df.columns:
        df['EVI_delta_2023_to_2024'] = df['EVI_2024.06.04'] - df['EVI_2023.07.30']
    
    if 'CanopyArea_delta_2023_to_2024' not in df.columns:
        df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.06.04'] - df['CanopyArea_2023.07.30']
    
    features.extend(['EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024'])
    
    available_features = [f for f in features if f in df.columns]
    df_clean = df.dropna(subset=available_features + ['target_binary'])
    
    X = df_clean[available_features]
    y = df_clean['target_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
        n_jobs=-1
    )
    
    automl.fit(X_train, y_train, dataset_name='redvine_prediction')
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", available_features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAuto-sklearn models:")
    print(automl.show_models())

if __name__ == "__main__":
    main()