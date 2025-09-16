import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import autosklearn.classification
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return
    
    target = 'redvine_count_2024'
    
    features = [
        'redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023',
        'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10',
        'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10',
        'Presence_2024.06.04', 'Presence_2024.07.10',
        'longitude', 'latitude',
        'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022',
        'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024',
        'EVI_delta_2022_to_2023', 'CanopyArea_delta_2022_to_2023'
    ]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
            except:
                pass
    
    for col in features:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' not found in dataset")
    
    available_features = [f for f in features if f in df.columns]
    
    df['EVI_CanopyArea_interaction_2024_06'] = df['EVI_2024.06.04'] * df['CanopyArea_2024.06.04']
    df['EVI_CanopyArea_interaction_2024_07'] = df['EVI_2024.07.10'] * df['CanopyArea_2024.07.10']
    
    available_features.extend(['EVI_CanopyArea_interaction_2024_06', 'EVI_CanopyArea_interaction_2024_07'])
    
    df = df.dropna(subset=available_features + [target])
    
    if df.empty:
        print("Error: No data available after preprocessing.")
        return
    
    X = df[available_features]
    y = (df[target] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=180,
        memory_limit=4096,
        n_jobs=-1,
        metric=autosklearn.metrics.f1_macro
    )
    
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    
    print("Features used:", available_features)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAuto-sklearn models:")
    print(automl.show_models())

if __name__ == "__main__":
    main()