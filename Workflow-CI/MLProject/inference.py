import os
import pandas as pd
import requests
import json

def run_inference():
    # 1. Ambil data tes asli hasil preprocessing (DATA RIIL MEDIS DIABETES!)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    X_test_path = os.path.join(current_dir, 'namadataset_preprocessing', 'test_preprocessed.csv')
    
    if not os.path.exists(X_test_path):
        raise FileNotFoundError(f"File data tes tidak ketemu di: {X_test_path}")
        
    df_test = pd.read_csv(X_test_path)
    
    # Ambil 5 baris sampel data riil/nyata
    sample_data = df_test.head(5)
    
    # =========================================================================
    # FORMAT PALING REKOMENDASI UNTUK MLFLOW SERVING: DATAFRAME_SPLIT
    # BIAR SERVER MLFLOW LANGSUNG PAHAM STRUKTUR KOLOM DATA ASLI KAMU
    # =========================================================================
    payload = {
        "dataframe_split": sample_data.to_dict(orient='split')
    }
    
    # 3. Tembak ke endpoint lokal resmi MLflow (/invocations) di port 5002
    url = "http://127.0.0.1:5002/invocations"
    headers = {"Content-Type": "application/json"}
    
    print("Sending real data tracking to MLflow serving endpoint...")
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # 4. Cetak hasil response dari server
    if response.status_code == 200:
        print("✅ BERHASIL SERVING! Hasil Prediksi Model:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ GAGAL SERVING! Status Code: {response.status_code}")
        print(response.text)
        raise Exception("Proses inference gagal menembak model server.")

if __name__ == '__main__':
    run_inference()
