import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    print(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df):
    print("Preprocessing data...")
    # Drop missing values if any
    df = df.dropna()
    
    # Categorical Encoding
    categorical_cols = ['gender', 'platform_usage', 'social_interaction_level', 'sleep_quality', 'digital_wellbeing_flag']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    # Features scaling
    numeric_cols = ['age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 
                    'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level', 
                    'addiction_level', 'mental_health_risk_score']
    scaler = StandardScaler()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
            
    return df

def save_data(df, output_path):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    raw_data_path = "../Teen_Mental_Health_raw.csv"
    processed_data_path = "Teen_Mental_Health_preprocessing.csv"
    
    df = load_data(raw_data_path)
    df_clean = preprocess_data(df)
    save_data(df_clean, processed_data_path)
