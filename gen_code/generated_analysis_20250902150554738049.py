import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree
from imblearn.over_sampling import SMOTE
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
    
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023', k=5)
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022', k=5)
    
    features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude',
        'latitude', 'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022',
        'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024',
        'block_variety', 'vineSpace'
    ]
    
    target = 'redvine_count_2024'
    
    X = df[features].copy()
    y = (df[target] > 0).astype(int)
    
    le = LabelEncoder()
    X['block_variety'] = le.fit_transform(X['block_variety'].astype(str))
    
    for col in X.select_dtypes(include=[object]).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        n_jobs=-1,
        random_state=42
    )
    
    automl.fit(X_train_res, y_train_res)
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nAuto-sklearn ensemble:")
    print(automl.show_models())

if __name__ == "__main__":
    main()