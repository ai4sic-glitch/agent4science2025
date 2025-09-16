import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
import autosklearn.regression
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    except FileNotFoundError:
        print("Dataset file not found. Please check the path.")
        return
    
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'EVI_2023.07.30', 'CanopyArea_2023.07.30', 
                'EVI_2022.08.04', 'CanopyArea_2022.08.04', 'block_variety', 'vineSpace']
    target = 'redvine_count_2024'
    
    for col in features + [target]:
        if col not in df.columns:
            print(f"Column {col} not found in dataset")
            return
    
    df = df[features + [target]].copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=features + [target])
    
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40, metric='haversine')
    
    k = 5
    distances, indices = tree.query(coords, k=k+1)
    
    spatial_lag_features = []
    for i, target_col in enumerate(['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023']):
        if target_col in df.columns:
            lag_values = []
            for idx in indices:
                neighbor_indices = idx[1:k+1]
                valid_neighbors = df[target_col].iloc[neighbor_indices].dropna()
                if len(valid_neighbors) > 0:
                    lag_values.append(valid_neighbors.mean())
                else:
                    lag_values.append(0)
            new_col = f'{target_col}_spatial_lag_k{k}'
            df[new_col] = lag_values
            spatial_lag_features.append(new_col)
    
    if 'EVI_2024.07.10' in df.columns and 'EVI_2023.07.30' in df.columns:
        df['EVI_delta'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    
    if 'CanopyArea_2024.07.10' in df.columns and 'CanopyArea_2023.07.30' in df.columns:
        df['CanopyArea_delta'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    
    final_features = features + spatial_lag_features
    if 'EVI_delta' in df.columns:
        final_features.append('EVI_delta')
    if 'CanopyArea_delta' in df.columns:
        final_features.append('CanopyArea_delta')
    
    X = df[final_features]
    y = df[target]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=300,
        memory_limit=4096,
        n_jobs=-1,
        metric=autosklearn.metrics.mean_absolute_error
    )
    
    automl.fit(X_train_scaled, y_train)
    
    y_pred = automl.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Features used:", list(X.columns))
    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("\nAuto-sklearn leaderboard:")
    print(automl.leaderboard())
    print("\nBest model description:")
    print(automl.show_models())

if __name__ == "__main__":
    main()