import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load the dataset
    df = pd.read_csv('./data/08_multiscale_features_simple_clean.csv')
    
    # Define features and target
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'girdle_presence_2023', 'EVI_2024.06.04', 'EVI_2024.07.10', 
                'CanopyArea_2024.06.04', 'CanopyArea_2024.07.10', 'longitude', 
                'latitude', 'EVI_2023.07.30', 'CanopyArea_2023.07.30', 
                'spatial_lag_redvine_2023', 'spatial_lag_redvine_2022', 
                'EVI_delta_2023_to_2024', 'CanopyArea_delta_2023_to_2024', 
                'block_variety', 'vineSpace']
    
    target = 'redvine_count_2024'
    
    # Create binary classification target
    df['target_binary'] = (df[target] > 0).astype(int)
    
    # Handle missing values
    df = df.dropna(subset=features + ['target_binary'])
    
    # Preprocess categorical features
    le = LabelEncoder()
    df['block_variety_encoded'] = le.fit_transform(df['block_variety'].astype(str))
    
    # Update features list with encoded categorical variable
    features_processed = [f for f in features if f != 'block_variety'] + ['block_variety_encoded']
    
    # Extract features and target
    X = df[features_processed]
    y = df['target_binary']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_res)
    
    # Make predictions
    y_pred = rf.predict(X_test_scaled)
    
    # Print evaluation results
    print("Features used:", features_processed)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print feature importances
    feature_importance = pd.DataFrame({
        'feature': features_processed,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importance.to_string(index=False))

if __name__ == "__main__":
    main()