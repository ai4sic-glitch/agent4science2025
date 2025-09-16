import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load dataset
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    # Define features and target
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 
                'Presence_2024.06.04', 'Presence_2024.07.10', 'longitude', 
                'latitude', 'EVI_2023.07.30', 'CanopyArea_2023.07.30', 
                'Presence_2023.07.30', 'EVI_2022.08.04', 'CanopyArea_2022.08.04', 
                'Presence_2022.08.04', 'spatial_lag_redvine_2023', 
                'spatial_lag_redvine_2022', 'EVI_delta_2023_to_2024', 
                'CanopyArea_delta_2023_to_2024', 'EVI_delta_2022_to_2023', 
                'CanopyArea_delta_2022_to_2023']
    
    target = 'redvine_count_2024'
    
    # Create binary classification target
    df['target_binary'] = (df[target] > 0).astype(int)
    
    # Create cumulative disease pressure feature
    df['weighted_sum_prior_counts'] = (0.5 * df['redvine_count_2021'] + 
                                      0.7 * df['redvine_count_2022'] + 
                                      0.9 * df['redvine_count_2023'])
    
    # Add the new feature to the feature list
    features.append('weighted_sum_prior_counts')
    
    # Check for missing features and handle them
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    # Prepare data
    X = df[available_features].copy()
    y = df['target_binary']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train AutoML classifier
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        n_jobs=-1,
        ensemble_size=1,
        initial_configurations_via_metalearning=0
    )
    
    automl.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = automl.predict(X_test_scaled)
    
    # Print results
    print("Available features used:", available_features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Get leaderboard without showing models to avoid KeyError
    try:
        print("\nLeaderboard:")
        print(automl.leaderboard())
    except:
        print("Leaderboard not available")

if __name__ == "__main__":
    main()