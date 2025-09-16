import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load dataset
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    # Define features and target
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022', 
                'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024', 
                'EVI_delta_2022_to_2023', 'CanopyArea_delta_2022_to_2023', 
                'weighted_sum_prior_counts', 'spatial_lag_weighted_sum', 
                'EVI_2024.07.10_CanopyArea_2024.07.10_interaction', 
                'EVI_2023.07.30_CanopyArea_2023.07.30_interaction', 
                'block_variety', 'vineSpace']
    
    target = 'redvine_count_2024'
    
    # Check for missing features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        features = [f for f in features if f in df.columns]
    
    # Create binary classification target
    df['target_binary'] = (df[target] > 0).astype(int)
    
    # Handle categorical features
    if 'block_variety' in df.columns:
        le = LabelEncoder()
        df['block_variety_encoded'] = le.fit_transform(df['block_variety'].astype(str))
        if 'block_variety' in features:
            features.remove('block_variety')
            features.append('block_variety_encoded')
    
    # Prepare data
    X = df[features].copy()
    y = df['target_binary']
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print class distribution
    print("Class distribution:")
    print(y.value_counts())
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize AutoSklearn classifier
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8},
        n_jobs=1
    )
    
    # Train model
    automl.fit(X_train, y_train)
    
    # Make predictions
    y_pred = automl.predict(X_test)
    
    # Print results
    print("\n=== Model Evaluation ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Model Details ===")
    print(automl.sprint_statistics())
    print("\nBest model:")
    print(automl.show_models())
    
    print("\nFeatures used:")
    for i, feature in enumerate(features):
        print(f"{i+1}. {feature}")

if __name__ == "__main__":
    main()