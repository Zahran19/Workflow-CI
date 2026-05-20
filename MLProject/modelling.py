import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import shutil

if __name__ == "__main__":
    print("Mulai melatih model untuk MLProject...")
    
    # Load dataset hasil preprocessing sebelumnya
    df = pd.read_csv('dataset_clean.csv')
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)
        
        # 1. Log model ke MLflow Tracking
        mlflow.sklearn.log_model(model, "model")
        
        # 2. Simpan model secara lokal ke folder khusus untuk kebutuhan build-docker
        if os.path.exists("live_model"):
            shutil.rmtree("live_model")
            
        mlflow.sklearn.save_model(model, "live_model")
        print("Model berhasil dilatih! Artefak disimpan di folder 'live_model'.")