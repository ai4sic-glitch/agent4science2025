import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load dataset
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    # Define features and target
    features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
        'latitude', 'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022', 
        'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024', 
        'EVI_delta_2022_to_2023', 'CanopyArea_delta_2022_to_2023', 
        'weighted_sum_prior_counts', 'spatial_lag_weighted_sum', 
        'EVI_2024.07.10_CanopyArea_2024.07.10_interaction', 
        'EVI_2023.07.30_CanopyArea_2023.07.30_interaction'
    ]
    
    target = 'redvine_count_2024'
    
    # Check for missing features and create interaction terms if needed
    available_features = []
    for feature in features:
        if feature in df.columns:
            available_features.append(feature)
        elif feature == 'EVI_2024.07.10_CanopyArea_2024.07.10_interaction':
            if 'EVI_2024.07.10' in df.columns and 'CanopyArea_2024.07.10' in df.columns:
                df[feature] = df['EVI_2024.07.10'] * df['CanopyArea_2024.07.10']
                available_features.append(feature)
        elif feature == 'EVI_2023.07.30_CanopyArea_2023.07.30_interaction':
            if 'EVI_2023.07.30' in df.columns and 'CanopyArea_2023.07.30' in df.columns:
                df[feature] = df['EVI_2023.07.30'] * df['CanopyArea_2023.07.30']
                available_features.append(feature)
    
    # Create binary classification target
    y = (df[target] > 0).astype(int)
    X = df[available_features].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Print features used
    print("Features used:", available_features)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # Train auto-sklearn classifier
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        n_jobs=-1,
        metric=autosklearn.metrics.f1_macro,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )
    
    automl.fit(X_train, y_train, dataset_name='redvine_classification')
    
    # Make predictions
    y_pred = automl.predict(X_test)
    
    # Print evaluation report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Show final model ensemble
    print("\nFinal ensemble:")
    print(automl.show_models())

if __name__ == "__main__":
    main()