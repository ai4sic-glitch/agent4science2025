import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
                'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024', 'block_variety']
    
    target = 'redvine_count_2024'
    
    # Check if all required columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        features = [f for f in features if f in df.columns]
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    
    # Create binary classification target
    y = (df[target] > 0).astype(int)
    
    # Prepare features
    X = df[features].copy()
    
    # Handle categorical variable
    if 'block_variety' in X.columns:
        X = pd.get_dummies(X, columns=['block_variety'], drop_first=True)
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Print class distribution
    print("Class distribution in training set:", np.bincount(y_train))
    print("Class distribution in test set:", np.bincount(y_test))
    
    # Initialize AutoSklearn classifier
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        n_jobs=-1,
        metric=autosklearn.metrics.f1
    )
    
    # Train model
    automl.fit(X_train, y_train, dataset_name='redvine_prediction')
    
    # Print model details
    print(automl.sprint_statistics())
    print("\nModels found by AutoSklearn:")
    print(automl.show_models())
    
    # Make predictions
    y_pred = automl.predict(X_test)
    
    # Evaluate model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Print feature importance if available
    try:
        if hasattr(automl, 'feature_importances_'):
            print("\nFeature Importances:")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': automl.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance)
    except:
        pass

if __name__ == "__main__":
    main()