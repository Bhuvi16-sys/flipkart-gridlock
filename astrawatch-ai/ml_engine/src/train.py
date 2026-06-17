# ml_engine/src/train.py
import os
import pickle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import load_and_clean_data, engineer_features, prepare_datasets

def build_lookup_matrix(df, geohash_col):
    matrix = df.groupby([geohash_col, 'hour']).agg(
        historical_incident_count=('id', 'count'),
        avg_priority=('priority_score', 'mean'),
        avg_clearance=('clearance_duration', 'mean')
    ).reset_index()
    
    max_incidents = matrix['historical_incident_count'].max() or 1
    matrix['congestion_risk_score'] = (
        (matrix['historical_incident_count'] / max_incidents) * 0.7 + 
        (matrix['avg_priority'] / 4.0) * 0.3
    ).clip(0.0, 1.0)
    
    return matrix.set_index([geohash_col, 'hour'])['congestion_risk_score'].to_dict()

def train_and_export():
    print("⚡ Loading and transforming dataset...")
    data_path = "../data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    
    df = load_and_clean_data(data_path)
    df = engineer_features(df)
    
    # Split and Target Encode safely
    X_train, X_test, y_train, y_test, train_df, categorical_mappings = prepare_datasets(df)
    
    print("⚙️ Generating Multi-Resolution Spatiotemporal Lookups...")
    lookup_dict_6 = build_lookup_matrix(train_df, 'geohash_6')
    lookup_dict_5 = build_lookup_matrix(train_df, 'geohash_5')
    lookup_dict_4 = build_lookup_matrix(train_df, 'geohash_4')
    
    spatiotemporal_lookups = {
        6: lookup_dict_6,
        5: lookup_dict_5,
        4: lookup_dict_4,
        'global_median': train_df['clearance_duration'].median()
    }
    
    print("🤖 Training Optimized XGBoost Engine...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"✅ Model Training Complete. Mean Absolute Error: {mae:.2f} minutes")
    
    os.makedirs("../models", exist_ok=True)
    
    with open("../models/clearance_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("../models/spatiotemporal_lookup.pkl", "wb") as f:
        pickle.dump(spatiotemporal_lookups, f)
    with open("../models/categorical_mappings.pkl", "wb") as f:
        pickle.dump(categorical_mappings, f)
        
    print("📦 Artifacts successfully saved to 'ml_engine/models/' folder!")

if __name__ == "__main__":
    train_and_export()