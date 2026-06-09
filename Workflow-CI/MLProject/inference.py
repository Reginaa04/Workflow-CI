import os
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score

def run_inference():
    # 1. Tentukan path dinamis lokasi data asli dan model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path data riil (bukan dummy!)
    X_test_path = os.path.join(current_dir, 'namadataset_preprocessing', 'test_preprocessed.csv')
    y_test_path = os.path.join(current_dir, 'namadataset_preprocessing', 'y_test.csv')
    
    # Path model hasil kriteria 2 (Training otomatis menghasilkan mlruns/0/)
    # Kita ambil model dari Run ID ID yang tersimpan di dalam folder eksperimen 0
    model_dir = os.path.join(current_dir, 'mlruns', '0')
    
    # Mencari folder run ID secara otomatis di dalam folder mlruns/0
    run_ids = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f)) and f != 'meta.yaml']
    if not run_ids:
        raise FileNotFoundError("Model hasil training tidak ditemukan di folder mlruns. Pastikan modelling.py berjalan sukses duluan!")
    
    # Ambil Run ID paling pertama/terbaru
    latest_run_id = run_ids[0]
    model_uri = os.path.join(model_dir, latest_run_id, 'artifacts', 'model')

    print(f"--> Melakukan LOAD MODEL asli dari: {model_uri}")
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # 2. Baca data riil asli medis diabetes
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze('columns')

    # 3. Lakukan inferensi/prediksi dengan data asli menggunakan model hasil load
    print("--> Menjalankan prediksi menggunakan data riil...")
    predictions = loaded_model.predict(X_test)

    # 4. Hitung akurasi riil dari proses serving lokal ini
    acc = accuracy_score(y_test, predictions)
    print(f"✅ SERVING INFERENCE SUKSES!")
    print(f"Hasil Evaluasi Serving Model - Akurasi: {acc:.4f}")

if __name__ == '__main__':
    run_inference()
