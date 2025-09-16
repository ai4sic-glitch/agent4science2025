import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import autosklearn.classification
from sklearn.preprocessing import LabelEncoder

def main():
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022', 
                'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024', 'block_variety']
    
    target = 'redvine_count_2024'
    
    X = df[features].copy()
    y = (df[target] > 0).astype(int)
    
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    if 'block_variety' in X.columns:
        le = LabelEncoder()
        X['block_variety'] = le.fit_transform(X['block_variety'].astype(str))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        memory_limit=4096,
        n_jobs=-1,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8}
    )
    
    automl.fit(X_train, y_train)
    
    print("Features used:", list(X.columns))
    print("\nEvaluation on test set:")
    y_pred = automl.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No Infection', 'Infection']))
    
    return automl

if __name__ == "__main__":
    main()