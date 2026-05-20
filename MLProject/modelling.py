import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import shutil

if __name__ == "__main__":
    print("Mulai melatih model untuk MLProject...")
    
    # Load dataset
    df = pd.read_csv('dataset_clean.csv')
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)
        
        # Trik Senior Engineer: Paksa simpan ke root folder GitHub Actions
        workspace = os.getenv("GITHUB_WORKSPACE", ".")
        model_path = os.path.join(workspace, "live_model")
        
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        mlflow.sklearn.save_model(model, model_path)
        print(f"Model berhasil dilatih! Artefak aman tersimpan di: {model_path}")