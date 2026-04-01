import pandas as pd
from datetime import datetime
import os

# Membuat folder untuk menyimpan log Excel jika belum ada
EXCEL_LOG_PATH = os.path.join(os.getcwd(), 'training_logs')
os.makedirs(EXCEL_LOG_PATH, exist_ok=True)

# Fungsi untuk mencatat data pelatihan ke dalam Excel
def log_training_data_to_excel(data, excel_filename="training_log.xlsx"):
    # Tentukan path file Excel
    excel_file_path = os.path.join(EXCEL_LOG_PATH, excel_filename)

    # Jika file Excel sudah ada, baca dan tambahkan data baru
    if os.path.exists(excel_file_path):
        # Baca file Excel yang ada
        df = pd.read_excel(excel_file_path, engine='openpyxl')
        # Tambahkan data baru
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    else:
        # Jika file Excel belum ada, buat data baru
        df = pd.DataFrame(data)
    
    # Simpan data ke Excel
    df.to_excel(excel_file_path, index=False, engine='openpyxl')

    print(f"Data pelatihan berhasil disimpan ke {excel_file_path}")