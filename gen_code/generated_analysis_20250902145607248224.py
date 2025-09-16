import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree
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
                'latitude', 'block_variety', 'vineSpace']
    target = 'redvine_count_2024'
    
    # Create binary classification target
    df['target_binary'] = (df[target] > 0).astype(int)
    
    # Compute deltas
    df['EVI_delta_2023_to_2024'] = df['EVI_2024.07.10'] - df['EVI_2023.07.30']
    df['CanopyArea_delta_2023_to_2024'] = df['CanopyArea_2024.07.10'] - df['CanopyArea_2023.07.30']
    
    # Add deltas to features
    features.extend(['EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024'])
    
    # Encode categorical variable
    le = LabelEncoder()
    df['block_variety_encoded'] = le.fit_transform(df['block_variety'].astype(str))
    features.append('block_variety_encoded')
    features.remove('block_variety')
    
    # Compute spatial lag features using BallTree for efficiency
    coords = df[['latitude', 'longitude']].values
    tree = BallTree(coords, leaf_size=40, metric='euclidean')
    
    # Compute spatial lag for redvine_count_2023
    k = 5
    distances, indices = tree.query(coords, k=k+1)
    spatial_lag_2023 = []
    for i in range(len(df)):
        neighbor_indices = indices[i][1:]  # Exclude self
        spatial_lag_2023.append(df['redvine_count_2023'].iloc[neighbor_indices].mean())
    df['spatial_lag_redvine_2023'] = spatial_lag_2023
    
    # Compute spatial lag for redvine_count_2022
    spatial_lag_2022 = []
    for i in range(len(df)):
        neighbor_indices = indices[i][1:]  # Exclude self
        spatial_lag_2022.append(df['redvine_count_2022'].iloc[neighbor_indices].mean())
    df['spatial_lag_redvine_2022'] = spatial_lag_2022
    
    # Add spatial lag features
    features.extend(['spatial_lag_redvine_2023', 'spatial_lag_redvine_2022'])
    
    # Prepare final feature set and target
    X = df[features].copy()
    y = df['target_binary']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Setup and run AutoSklearn
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        n_jobs=-1,
        metric=autosklearn.metrics.balanced_accuracy
    )
    
    automl.fit(X_train, y_train, dataset_name='redvine_classification')
    
    # Predict and evaluate
    y_pred = automl.predict(X_test)
    
    # Print results
    print("Features used:", features)
    print("\nAutoSklearn model summary:")
    print(automl.sprint_statistics())
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nBest model:", automl.show_models())

if __name__ == "__main__":
    main()