import os
import pandas as pd
import requests
import json
import sys

def run_inference():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Cari file data riil diabetes hasil preprocessing kriteria 1
    X_test_path = os.path.join(current_dir, 'namadataset_preprocessing', 'test_preprocessed.csv')
    
    if not os.path.exists(X_test_path):
        print(f"❌ File data riil tidak ditemukan di: {X_test_path}")
        sys.exit(1)
        
    df_test = pd.read_csv(X_test_path)
    
    # Ambil 3 baris sampel data klinis nyata untuk diuji ke model
    sample_data = df_test.head(3)
    
    # 2. Format payload wajib 'dataframe_split' agar diterima oleh MLflow Server
    payload = {
        "dataframe_split": sample_data.to_dict(orient='split')
    }
    
    # 3. Endpoint lokal port 5002 yang sudah dinyalakan oleh main.yml
    url = "http://127.0.0.1:5002/invocations"
    headers = {"Content-Type": "application/json"}
    
    print("--> Mengirimkan request data riil ke server MLflow (Waitress)...")
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        # 4. Tampilkan hasil prediksi model terlatih nyata untuk bukti kelulusan
        if response.status_code == 200:
            print("\n✅ EVIDENCE VALID! Hasil Prediksi Model Terlatih Nyata:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ GAGAL! Server merespons dengan status code: {response.status_code}")
            print(response.text)
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Gagal melakukan hit endpoint: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run_inference()
