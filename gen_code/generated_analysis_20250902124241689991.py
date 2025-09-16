import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import autosklearn.regression
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load the dataset
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    # Select relevant features and target
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 
                'Presence_2024.06.04', 'Presence_2024.07.10', 
                'longitude', 'latitude']
    target = 'redvine_count_2024'
    
    # Check if all required columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return
    
    # Create feature matrix and target vector
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle missing values - drop rows with NaN in target or features
    missing_mask = y.isna() | X.isna().any(axis=1)
    if missing_mask.any():
        print(f"Dropping {missing_mask.sum()} rows with missing values")
        X = X[~missing_mask]
        y = y[~missing_mask]
    
    # Convert any object columns to numeric if possible
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                pass
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Print features being used
    print("Features used for modeling:")
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")
    print(f"\nTarget: {target}")
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize and train AutoSklearn regressor
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        memory_limit=4096,
        n_jobs=-1,
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        seed=42
    )
    
    automl.fit(X_train, y_train)
    
    # Make predictions
    y_pred = automl.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n=== Evaluation Results ===")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Show model leaderboard
    print("\n=== Model Leaderboard ===")
    print(automl.leaderboard())
    
    # Show the best model
    print("\n=== Best Model ===")
    print(automl.show_models())

if __name__ == "__main__":
    main()