# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def pipeline_preprocessing(input_path, output_dir):
    # 1. Load data
    print("Memuat dataset dari:", input_path)
    df = pd.read_csv(input_path)
    
    # 2. Preprocessing / Pembersihan Data Singkat
    # Memisahkan fitur (X) dan target (y) - Asumsi kolom terakhir adalah target 'Outcome'
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Mengatasi missing value dasar (jika ada) dengan nilai mean
    X = X.fillna(X.mean())
    
    # Split data menjadi Train (80%) dan Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling fitur menggunakan StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Mengembalikan ke bentuk DataFrame untuk disimpan ke CSV
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['Outcome'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['Outcome'] = y_test.values
    
    # 3. Menyimpan hasil preprocessing ke folder tujuan
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'test_preprocessed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("Sukses! File train dan test preprocessed telah disimpan di:", output_dir)

if __name__ == '__main__':
    # Menjalankan fungsi otomatisasi preprocessing
    pipeline_preprocessing(
        input_path='namadataset_raw/diabetes.csv', 
        output_dir='preprocessing/namadataset_preprocessing'
    )
