import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return
    
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 
                'Presence_2024.06.04', 'Presence_2024.07.10', 'longitude', 
                'latitude', 'EVI_2023.07.30', 'CanopyArea_2023.07.30', 
                'Presence_2023.07.30', 'EVI_2022.08.04', 'CanopyArea_2022.08.04', 
                'Presence_2022.08.04']
    
    target = 'redvine_count_2024'
    
    for col in features + [target]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=features + [target])
    
    X = df[features].copy()
    y = (df[target] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,
        memory_limit=4096,
        n_jobs=-1,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.8}
    )
    
    automl.fit(X_train_scaled, y_train, dataset_name='redvine_classification')
    
    y_pred = automl.predict(X_test_scaled)
    
    print("Features used:", features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nAuto-sklearn models:")
    print(automl.show_models())

if __name__ == "__main__":
    main()