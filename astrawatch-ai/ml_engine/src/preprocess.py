# ml_engine/src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import geohash_hilbert as gh

BENGALURU_BBOX = {
    'min_lat': 12.4, 'max_lat': 13.5,
    'min_lon': 77.3, 'max_lon': 78.2
}

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Target Variable: Calculate Clearance Duration
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    
    duration = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    df['clearance_duration'] = duration.fillna(45.0)
    df['clearance_duration'] = df['clearance_duration'].clip(lower=5.0, upper=480.0)
    
    return df

def apply_target_encoding(train_df, test_df, target_col='clearance_duration', alpha=10.0):
    categorical_cols = ['event_type', 'event_cause', 'junction', 'zone']
    global_mean = train_df[target_col].mean()
    mappings = {'global_mean': global_mean}
    
    for col in categorical_cols:
        train_df[col] = train_df[col].fillna('unknown').astype(str)
        test_df[col] = test_df[col].fillna('unknown').astype(str)
        
        # M-estimate smoothing
        agg = train_df.groupby(col)[target_col].agg(['count', 'mean'])
        smoothed = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
        
        col_mapping = smoothed.to_dict()
        mappings[col] = col_mapping
        
        # Apply mapping
        train_df[f'{col}_encoded'] = train_df[col].map(col_mapping).fillna(global_mean)
        test_df[f'{col}_encoded'] = test_df[col].map(col_mapping).fillna(global_mean)
        
    return train_df, test_df, mappings

def engineer_features(df):
    # Impute and Clip Spatial Coordinates
    df['latitude'] = df['latitude'].fillna(np.nanmedian(df['latitude'])).clip(BENGALURU_BBOX['min_lat'], BENGALURU_BBOX['max_lat'])
    df['longitude'] = df['longitude'].fillna(np.nanmedian(df['longitude'])).clip(BENGALURU_BBOX['min_lon'], BENGALURU_BBOX['max_lon'])
    
    # Temporal Features
    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    # Multi-resolution Geohashing
    def compute_geohashes(row):
        try:
            return pd.Series({
                'geohash_6': gh.encode(row['longitude'], row['latitude'], precision=6),
                'geohash_5': gh.encode(row['longitude'], row['latitude'], precision=5),
                'geohash_4': gh.encode(row['longitude'], row['latitude'], precision=4)
            })
        except:
            return pd.Series({'geohash_6': "unknown", 'geohash_5': "unknown", 'geohash_4': "unknown"})
            
    geohashes = df.apply(compute_geohashes, axis=1)
    df = pd.concat([df, geohashes], axis=1)
    
    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    df['priority_score'] = df['priority'].map(priority_map).fillna(2)
    
    return df

def prepare_datasets(df):
    feature_cols = [
        'latitude', 'longitude', 'hour_sin', 'hour_cos', 
        'day_sin', 'day_cos', 'is_weekend', 'priority_score',
        'event_type_encoded', 'event_cause_encoded', 'junction_encoded', 'zone_encoded'
    ]
    
    # Train-test split before target encoding to prevent data leakage
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Apply Target Encoding
    train_df, test_df, mappings = apply_target_encoding(train_df, test_df)
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['clearance_duration']
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['clearance_duration']
    
    return X_train, X_test, y_train, y_test, train_df, mappings