import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def compute_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, metric='euclidean')
    spatial_lag = np.zeros(len(df))
    
    for i in range(len(df)):
        dist, idx = tree.query([coords[i]], k=k+1)
        neighbor_indices = idx[0][1:]
        spatial_lag[i] = df.iloc[neighbor_indices][target_col].mean()
    
    return spatial_lag

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024', 
                'block_variety', 'vineSpace', 
                'EVI_2024.07.10_CanopyArea_2024.07.10_interaction', 
                'EVI_2023.07.30_CanopyArea_2023.07.30_interaction', 
                'EVI_2023.07.30', 'CanopyArea_2023.07.30', 
                'EVI_2022.08.04', 'CanopyArea_2022.08.04']
    
    target = 'redvine_count_2024'
    
    for col in ['vineSpace']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.extract('(\d+\.?\d*)').astype(float)
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023', k=5)
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022', k=5)
    
    le = LabelEncoder()
    df['block_variety_encoded'] = le.fit_transform(df['block_variety'])
    
    X = df[features + ['spatial_lag_redvine_2023', 'spatial_lag_redvine_2022', 'block_variety_encoded']].copy()
    y = (df[target] > 0).astype(int)
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8}
    )
    
    automl.fit(X_train_scaled, y_train)
    
    y_pred = automl.predict(X_test_scaled)
    
    print("Features used:", list(X.columns))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nAuto-sklearn models:", automl.show_models())

if __name__ == "__main__":
    main()