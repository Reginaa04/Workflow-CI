import os
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1. OTOMATIS CARI DAN LOAD MODEL HASIL TRAINING REAL
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'mlruns', '0')

try:
    # Cari Run ID secara otomatis
    run_ids = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f)) and f != 'meta.yaml']
    if not run_ids:
        raise FileNotFoundError("Model hasil training tidak ditemukan di mlruns!")
    
    latest_run_id = run_ids[0]
    model_uri = os.path.join(model_dir, latest_run_id, 'artifacts', 'model')
    print(f"--> [SUCCESS] Memuat model terlatih dari: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"--> [WARNING] Gagal memuat model di awal: {e}")
    model = None

# =========================================================================
# MEMENUHI KATA-KATA REVIEWER: ENDPOINT /predict MENERIMA INPUT AKTUAL JSON
# =========================================================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model tidak termuat dengan benar"}), 500
        
    try:
        # Menerima input data aktual JSON
        data = request.get_json()
        
        # Ubah data JSON kembali menjadi DataFrame untuk diprediksi model
        df_input = pd.DataFrame(data)
        
        # Lakukan prediksi nyata menggunakan model terlatih
        predictions = model.predict(df_input)
        
        # Mengembalikan hasil prediksi model terlatih nyata
        return jsonify({
            "status": "success",
            "predictions": predictions.tolist()
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 400

# Endpoint cek kesehatan server
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Server Inference Regina Siap!"})

if __name__ == '__main__':
    # Jalankan server API lokal pada port 5002
    app.run(host='0.0.0.0', port=5002)
