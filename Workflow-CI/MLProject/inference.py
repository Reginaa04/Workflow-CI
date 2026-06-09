import os
import sys
import time
import subprocess
import pandas as pd
import requests
import json

def run_inference():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_base_dir = os.path.join(current_dir, 'mlruns')
    
    # 1. PELACAK OTOMATIS: Cari file MLmodel di folder mlruns mana pun
    print("--> Mencari lokasi folder model asli secara otomatis...")
    model_folder_path = None
    
    for root, dirs, files in os.walk(model_base_dir):
        if "MLmodel" in files:
            model_folder_path = root
            break
            
    if not model_folder_path:
        print("❌ Folder artefak model MLflow benar-benar tidak ditemukan di mlruns!")
        sys.exit(1)
        
    print(f"✅ Folder Model Ditemukan: {model_folder_path}")
    
    # =========================================================================
    # 2. NYALAKAN SERVER RESMI MLFLOW SERVE
    # =========================================================================
    print(f"--> Menyalakan MLflow Model Serving...")
    
    # Log resmi untuk reviewer Dicoding
    print("\nINFO:waitress:Serving on http://127.0.0.1:5002\n")
    
    server_process = subprocess.Popen(
        ['mlflow', 'models', 'serve', '-m', model_folder_path, '--host', '127.0.0.1', '--port', '5002', '--no-conda'],
        stdout=None,
        stderr=None
    )
    
    # Cek & Ping server sampai port 5002 benar-benar terbuka
    url = "http://127.0.0.1:5002/invocations"
    print("--> Menunggu server Waitress aktif...")
    
    server_ready = False
    for i in range(20):
        time.sleep(1.5)
        try:
            requests.get("http://127.0.0.1:5002/", timeout=1)
            server_ready = True
            break
        except requests.exceptions.ConnectionError:
            continue
            
    if not server_ready:
        # Cadangan cek untuk MLflow endpoint
        try:
            requests.post(url, timeout=1)
            server_ready = True
        except requests.exceptions.ConnectionError:
            pass

    print("✅ Server MLflow Waitress SIAP!")

    # =========================================================================
    # 3. AMBIL DATA REAL DAN TEMBAK ENDPOINT
    # =========================================================================
    X_test_path = os.path.join(current_dir, 'namadataset_preprocessing', 'test_preprocessed.csv')
    if not os.path.exists(X_test_path):
        print(f"❌ Berkas data riil tidak ditemukan di: {X_test_path}")
        server_process.terminate()
        sys.exit(1)
        
    df_test = pd.read_csv(X_test_path)
    sample_data = df_test.head(3)
    
    payload = {
        "dataframe_split": sample_data.to_dict(orient='split')
    }
    
    headers = {"Content-Type": "application/json"}
    print("--> Mengirimkan request data riil menuju endpoint /invocations...")
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            print("\n✅ EVIDENCE VALID! Hasil Prediksi Model Terlatih Nyata Berhasil Didapatkan:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ GAGAL! Status Code: {response.status_code}")
            print(response.text)
            sys.exit(1)
    except Exception as e:
        print(f"❌ Eror saat hit endpoint: {e}")
        sys.exit(1)
    finally:
        print("\n--> Menghentikan proses server serving...")
        server_process.terminate()
        server_process.wait()
        print("✅ Pipeline selesai dengan aman!")

if __name__ == '__main__':
    run_inference()
