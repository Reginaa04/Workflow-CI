import os
import sys
import time
import subprocess
import pandas as pd
import requests
import json

def run_inference():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_base_dir = os.path.join(current_dir, 'mlruns', '0')
    
    # 1. Cari RUN_ID hasil training
    if not os.path.exists(model_base_dir):
        print("❌ Folder mlruns/0 tidak ditemukan!")
        sys.exit(1)
        
    run_ids = [f for f in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, f)) and f != 'meta.yaml']
    if not run_ids:
        print("❌ Tidak ada Run ID hasil training!")
        sys.exit(1)
        
    latest_run_id = run_ids[0]
    artifacts_dir = os.path.join(model_base_dir, latest_run_id, 'artifacts')
    
    # Cari folder model valid
    model_folder = None
    for item in os.listdir(artifacts_dir):
        sub_path = os.path.join(artifacts_dir, item)
        if os.path.isdir(sub_path) and "MLmodel" in os.listdir(sub_path):
            model_folder = item
            break
            
    if not model_folder:
        print("❌ Folder artefak model MLflow tidak ditemukan!")
        sys.exit(1)
        
    model_uri = os.path.join(artifacts_dir, model_folder)
    
    # =========================================================================
    # 2. NYALAKAN SERVER RESMI MLFLOW SERVE
    # =========================================================================
    print(f"--> Menyalakan MLflow Model Serving untuk path: {model_uri}")
    
    server_process = subprocess.Popen(
        ['mlflow', 'models', 'serve', '-m', model_uri, '--host', '127.0.0.1', '--port', '5002', '--no-conda'],
        stdout=None,
        stderr=None
    )
    
    # KUNCI ANTI CONNECTION REFUSED: Ping server sampai benar-benar merespons!
    url = "http://127.0.0.1:5002/invocations"
    print("--> Menunggu server Waitress aktif mendengarkan port 5002...")
    
    server_ready = False
    for i in range(30): # Coba ping selama 30 detik maksimal
        time.sleep(2)
        try:
            # Kirim request kosong hanya untuk cek apakah port sudah terbuka
            requests.get("http://127.0.0.1:5002/", timeout=1)
            server_ready = True
            break
        except requests.exceptions.ConnectionError:
            continue
            
    if not server_ready:
        # Jika dalam 30 detik masih gagal, cek apakah port merespons 405 (artinya aktif tapi method salah, itu wajar untuk MLflow)
        try:
            requests.post(url, timeout=1)
            server_ready = True
        except requests.exceptions.ConnectionError:
            print("❌ Server gagal menyala dalam waktu 30 detik.")
            server_process.terminate()
            sys.exit(1)

    print("✅ Server MLflow Waitress AKTIF! Siap menerima data.")

    # =========================================================================
    # 3. AMBIL DATA REAL DAN TEMBAK
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
        print(f"❌ Eror mendadak saat hit endpoint: {e}")
        sys.exit(1)
    finally:
        print("\n--> Menghentikan proses server serving...")
        server_process.terminate()
        server_process.wait()
        print("✅ Pipeline selesai dieksekusi dengan aman!")

if __name__ == '__main__':
    run_inference()
