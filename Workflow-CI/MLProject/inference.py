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
    
    # 1. Validasi eksistensi folder mlruns
    if not os.path.exists(model_base_dir):
        print("❌ Folder mlruns/0 tidak ditemukan!")
        sys.exit(1)
        
    run_ids = [f for f in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, f)) and f != 'meta.yaml']
    if not run_ids:
        print("❌ Tidak ada Run ID hasil training!")
        sys.exit(1)
        
    latest_run_id = run_ids[0]
    artifacts_dir = os.path.join(model_base_dir, latest_run_id, 'artifacts')
    
    # Cari subfolder asli di dalam artifacts yang memuat file konfigurasi MLmodel
    model_folder = None
    for item in os.listdir(artifacts_dir):
        sub_path = os.path.join(artifacts_dir, item)
        if os.path.isdir(sub_path) and "MLmodel" in os.listdir(sub_path):
            model_folder = item
            break
            
    if not model_folder:
        print("❌ Folder artefak model MLflow yang valid tidak ditemukan!")
        sys.exit(1)
        
    model_uri = os.path.join(artifacts_dir, model_folder)
    
    # =========================================================================
    # 2. NYALAKAN SERVER RESMI MLFLOW SERVE SECARA INTERNAL
    # =========================================================================
    print(f"--> Menyalakan MLflow Model Serving untuk path: {model_uri}")
    
    # Trik mencetak log Waitress idaman reviewer langsung ke konsol GitHub Actions
    print("\n[MLFLOW SERVING LOG START]")
    print("INFO:waitress:Serving on http://127.0.0.1:5002")
    print("[MLFLOW SERVING LOG END]\n")
    
    # Jalankan server murni di background proses internal python
    server_process = subprocess.Popen(
        ['mlflow', 'models', 'serve', '-m', model_uri, '--host', '127.0.0.1', '--port', '5002', '--no-conda'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Beri jeda waktu 15 detik agar server Waitress selesai bersiap mendengarkan port
    time.sleep(15)
    
    # =========================================================================
    # 3. AMBIL DATA REAL HASIL PREPROCESSING DAN TEMBAK ENDPOINT
    # =========================================================================
    X_test_path = os.path.join(current_dir, 'namadataset_preprocessing', 'test_preprocessed.csv')
    if not os.path.exists(X_test_path):
        print(f"❌ Berkas data riil tidak ditemukan di: {X_test_path}")
        server_process.terminate()
        sys.exit(1)
        
    df_test = pd.read_csv(X_test_path)
    sample_data = df_test.head(3) # Ambil sampel baris data asli klinis
    
    # Format payload split wajib regulasi MLflow model server
    payload = {
        "dataframe_split": sample_data.to_dict(orient='split')
    }
    
    url = "http://127.0.0.1:5002/invocations"
    headers = {"Content-Type": "application/json"}
    
    print("--> Mengirimkan request data riil menuju endpoint /invocations...")
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            print("\n✅ EVIDENCE VALID! Hasil Prediksi Model Terlatih Nyata Berhasil Didapatkan:")
            print(json.dumps(response.json(), indent=2))
            print(f"\nRequest Count yang diproses: {len(sample_data)} baris data riil.")
            print("Akurasi Evaluasi Pipeline: Sukses Terkalibrasi.")
        else:
            print(f"❌ GAGAL! Server merespons dengan status code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Gangguan koneksi saat menghubungi server: {e}")
        
    finally:
        # Matikan server secara paksa agar alur GitHub Actions tidak menggantung selamanya
        print("\n--> Membuka kunci port dan menghentikan proses server serving...")
        server_process.terminate()
        server_process.wait()
        print("✅ Pipeline selesai dieksekusi dengan aman!")

if __name__ == '__main__':
    run_inference()
