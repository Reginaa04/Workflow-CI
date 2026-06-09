import os
import pandas as pd
import requests
import json

def run_inference():
    # 1. Ambil data tes asli hasil preprocessing (DATA RIIL MEDIS DIABETES)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    X_test_path = os.path.join(current_dir, 'namadataset_preprocessing', 'test_preprocessed.csv')
    
    if not os.path.exists(X_test_path):
        raise FileNotFoundError(f"File data tes tidak ditemukan di: {X_test_path}")
        
    df_test = pd.read_csv(X_test_path)
    sample_data = df_test.head(5)
    
    # 2. Payload format split sesuai standar spesifikasi MLflow Server
    payload = {
        "dataframe_split": sample_data.to_dict(orient='split')
    }
    
    # 3. Hit ke endpoint resmi
    url = "http://127.0.0.1:5002/invocations"
    headers = {"Content-Type": "application/json"}
    
    print("--> Melakukan API hit menggunakan requests.post() ke MLflow Serve...")
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            print("✅ EVIDENCE VALID! Hasil Prediksi Model Terlatih Nyata:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ GAGAL SERVING! Status Code: {response.status_code}")
            print(response.text)
            raise Exception("Inference gagal mendapatkan hasil dari model server.")
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ GAGAL: Koneksi ditolak. Server MLflow gagal di-serve. Detail: {e}")
        raise

if __name__ == '__main__':
    run_inference()
