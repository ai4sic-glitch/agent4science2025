import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def compute_spatial_lag(df, target_col, k=5):
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40, metric='euclidean')
    spatial_lag = np.zeros(len(df))
    
    for i in range(len(df)):
        dist, idx = tree.query([coords[i]], k=k+1)
        neighbors_idx = idx[0][1:]
        spatial_lag[i] = df.iloc[neighbors_idx][target_col].mean()
    
    return spatial_lag

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude',
        'latitude', 'EVI_2023.07.30', 'CanopyArea_2023.07.30',
        'EVI_2022.08.04', 'CanopyArea_2022.08.04', 'block_variety', 'vineSpace'
    ]
    
    target = 'redvine_count_2024'
    
    df = df[features + [target]].copy()
    
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    df['EVI_delta_2022_to_2023'] = df['EVI_2023.07.30'] - df['EVI_2022.08.04']
    df['CanopyArea_delta_2022_to_2023'] = df['CanopyArea_2023.07.30'] - df['CanopyArea_2022.08.04']
    
    df['weighted_sum_prior_counts'] = (
        0.5 * df['redvine_count_2021'] + 
        0.7 * df['redvine_count_2022'] + 
        0.9 * df['redvine_count_2023']
    )
    
    df['spatial_lag_redvine_2023'] = compute_spatial_lag(df, 'redvine_count_2023', k=5)
    df['spatial_lag_redvine_2022'] = compute_spatial_lag(df, 'redvine_count_2022', k=5)
    df['spatial_lag_weighted_sum'] = compute_spatial_lag(df, 'weighted_sum_prior_counts', k=5)
    
    le = LabelEncoder()
    df['block_variety'] = le.fit_transform(df['block_variety'].astype(str))
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].str.extract('(\d+\.?\d*)')[0], errors='coerce')
            except:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    X = df.drop(columns=[target])
    y = (df[target] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        n_jobs=-1,
        ensemble_size=1,
        initial_configurations_via_metalearning=0
    )
    
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", list(X.columns))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAuto-sklearn models:")
    print(automl.show_models())

if __name__ == "__main__":
    main()