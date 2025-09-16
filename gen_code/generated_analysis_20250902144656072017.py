import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import autosklearn.regression
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return
    
    required_features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude',
        'latitude', 'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022',
        'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024',
        'block_variety', 'vineSpace', 'EVI_2024.07.10_CanopyArea_2024.07.10_interaction'
    ]
    
    target = 'redvine_count_2024'
    
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        required_features = [f for f in required_features if f in df.columns]
    
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found in dataset.")
        return
    
    df_clean = df[required_features + [target]].copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
            except:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean = df_clean.dropna(subset=required_features + [target])
    
    if df_clean.empty:
        print("Error: No valid data remaining after cleaning.")
        return
    
    X = df_clean[required_features]
    y = df_clean[target]
    
    categorical_features = ['block_variety'] if 'block_variety' in X.columns else []
    for col in categorical_features:
        X[col] = X[col].astype('category')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=180,
        memory_limit=4096,
        n_jobs=-1,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8}
    )
    
    automl.fit(X_train, y_train, dataset_name='redvine_prediction')
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", list(X.columns))
    print("\nModel Evaluation:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    print("\nAuto-sklearn leaderboard:")
    print(automl.leaderboard())
    
    print("\nBest model:")
    print(automl.show_models())

if __name__ == "__main__":
    main()