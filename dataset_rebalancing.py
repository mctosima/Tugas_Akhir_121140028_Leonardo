import os
import random
import pandas as pd

# Path ke direktori kelas
FALL_CLASS_PATH = "data-skeleton/fall"
NON_FALL_CLASS_PATH = "data-skeleton/not_fall"
EXCEL_FILE_PATH = "deleted_files.xlsx"  # Tempat menyimpan daftar file yang dihapus

RANDOM_SEED = 42

def get_files_in_directory(directory_path: str) -> list:
    """
    Mengambil semua file dalam sebuah direktori dan subdirektorinya.
    
    Args:
        directory_path (str): Path ke direktori utama.
    
    Returns:
        list: Daftar path file yang ada dalam direktori dan subdirektorinya.
    """
    file_list = []
    
    # Telusuri direktori dan subdirektori
    for root, dirs, files in os.walk(directory_path):
        # Tambahkan setiap file ke dalam list
        for file in files:
            file_list.append(os.path.join(root, file)) 
    
    return file_list

def balance_classes(fall_class_path: str, non_fall_class_path: str, excel_file_path: str) -> None:
    """Menyeimbangkan jumlah file antara dua kelas dan mendata file yang dihapus ke Excel."""
    
    random.seed(RANDOM_SEED)  

    # Ambil daftar file dari kedua kelas
    fall_files = get_files_in_directory(fall_class_path)
    non_fall_files = get_files_in_directory(non_fall_class_path)
    
    # Hitung selisih
    num_fall = len(fall_files)
    num_non_fall = len(non_fall_files)
    
    print(f"Jumlah 'fall': {num_fall}")
    print(f"Jumlah 'non_fall': {num_non_fall}")
    
    if num_fall == num_non_fall:
        print("Dataset sudah seimbang!")
        return

    # Tentukan kelas dengan jumlah lebih banyak
    if num_fall > num_non_fall:
        larger_class = fall_files
        larger_class_path = fall_class_path
        smaller_class = non_fall_files
    else:
        larger_class = non_fall_files
        larger_class_path = non_fall_class_path
        smaller_class = fall_files

    # Hitung selisih
    difference = abs(num_fall - num_non_fall)
    
    # Pilih file secara acak dari kelas yang lebih banyak
    files_to_delete = random.sample(larger_class, difference)
    
    # Hapus file dan catat di Excel
    deleted_files = []
    for file in files_to_delete:
        normalized_path = file.replace('\\', '/')
        os.remove(normalized_path)  # Hapus file
        deleted_files.append({"deleted_file": file, "class": "fall" if larger_class_path == fall_class_path else "non_fall"})
    
    # Simpan daftar file yang dihapus ke dalam Excel
    if deleted_files:
        df = pd.DataFrame(deleted_files)
        if os.path.exists(excel_file_path):
            existing_df = pd.read_excel(excel_file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_excel(excel_file_path, index=False)
        else:
            df.to_excel(excel_file_path, index=False)
    
    print(f"Total file yang dihapus: {len(files_to_delete)}")

# Panggil fungsi untuk menyeimbangkan dataset

def main() -> None:
    balance_classes(FALL_CLASS_PATH, NON_FALL_CLASS_PATH, EXCEL_FILE_PATH)
    

if __name__ == "__main__":
    main()