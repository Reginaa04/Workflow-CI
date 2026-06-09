import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_baseline():
    # 1. Memuat data menggunakan path yang sudah diperbaiki (tanpa folder 'preprocessing/')
    X_train = pd.read_csv('namadataset_preprocessing/train_preprocessed.csv')
    y_train = pd.read_csv('namadataset_preprocessing/y_train.csv').squeeze('columns')
    X_test = pd.read_csv('namadataset_preprocessing/test_preprocessed.csv')
    y_test = pd.read_csv('namadataset_preprocessing/y_test.csv').squeeze('columns')
    
    # 2. Hanya aktifkan autolog (Sesuai instruksi reviewer, set_experiment DIBUANG)
    mlflow.autolog()
    
    # 3. Langsung jalankan model (Sesuai instruksi reviewer, start_run DIBUANG)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluasi dan catat metrik secara resmi ke MLflow
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Baseline Autolog Accuracy: {acc:.4f}")
    mlflow.log_metric("accuracy", acc)

if __name__ == '__main__':
    train_baseline()
