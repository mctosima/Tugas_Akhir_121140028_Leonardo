import cv2
import pandas as pd
import os
import numpy as np

def time_to_seconds(tstr):
    """
    Mengonversi string waktu (HH:MM:SS atau MM:SS) menjadi detik.
    """
    parts = [int(p) for p in tstr.split(':')]
    return parts[-1] + (parts[-2] * 60) + (parts[-3] * 3600 if len(parts) == 3 else 0)

def adjust_start_sec(start_sec, end_sec, percentage_base, scale_control):
    """
    Menggeser start_sec mendekati end_sec dengan pergeseran yang bergantung pada selisih waktu.
    Semakin besar selisih (end_sec - start_sec), semakin besar pergeseran start_sec.
    """
    # Hitung selisih waktu
    time_diff = end_sec - start_sec
    
    # Tentukan faktor pengali berdasarkan panjang rentang waktu (time_diff)
    # Rentang waktu kecil: pergeseran kecil. Rentang waktu besar: pergeseran besar.
    if time_diff > 2:
        scaling_factor = percentage_base * (time_diff / scale_control)
    
    # Pastikan scaling_factor tidak lebih kecil dari percentage_base (untuk rentang waktu yang sangat kecil)
    # scaling_factor = max(scaling_factor, percentage_base)
        scaling_factor = scaling_factor + percentage_base
    else:
        scaling_factor = percentage_base
    
    # Hitung seberapa besar perubahan yang perlu diterapkan
    shift = time_diff * (scaling_factor / 100)
    
    # Geser start_sec mendekati end_sec
    new_start_sec = start_sec + shift
    
    # Pastikan start_sec tidak lebih besar dari end_sec
    return min(new_start_sec, end_sec)

def extract_random_frames(video_path, start_sec, end_sec, output_dir, label, num_frames=30):
    """
    Mengekstrak frame secara acak dalam rentang waktu yang telah disesuaikan.
    Jika frame unik kurang dari 30, ambil frame yang sama secara acak hingga mencapai jumlah 30 frame.
    """
    
    if os.path.exists(output_dir):
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * video_fps)
    end_frame = min(int(end_sec * video_fps), total_frames)
    
    # Membuat daftar untuk frame yang dipilih
    selected_frames = []
    
    # Ambil frame secara acak hingga semua frame dalam rentang terambil
    while len(selected_frames) < num_frames:
        # Pilih frame acak dalam rentang waktu yang diberikan
        available_frames = list(range(start_frame, end_frame))
        
        # Pilih frame yang belum diambil
        remaining_frames = list(set(available_frames) - set(selected_frames))
        
        if remaining_frames:
            # Pilih frame acak dari yang tersisa
            random_frame = np.random.choice(remaining_frames)
            selected_frames.append(random_frame)
        else:
            # Jika semua frame unik sudah diambil, ambil secara acak lagi sampai mencapai 30 frame
            random_frame = np.random.choice(available_frames)
            selected_frames.append(random_frame)
    
    # Ekstraksi frame yang telah dipilih
    frame_count = 0
    for frame_idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        output_filename = f"{label}_{frame_count}.jpg"
        cv2.imwrite(os.path.join(output_dir, output_filename), frame)
        frame_count += 1
    
    cap.release()


def main():
    df = pd.read_excel('.\dataset\kuleuven.app.box\dataset_desc_generated.xlsx')
    base_output_dir = 'extracted_frames'
    
    for idx, row in df.iterrows():
        video_file = row['video_path']
        start_t = row['start_time']
        end_t = row['end_time']
        label = row.get('label', 'unknown_event')
        
        start_sec = time_to_seconds(str(start_t))
        end_sec = time_to_seconds(str(end_t))
        if label == 'fall':
            # Sesuaikan rentang start berdasarkan persentase (misalnya 30%)
            start_sec = adjust_start_sec(start_sec, end_sec, 25, 15)

        
        label_folder = os.path.join(base_output_dir, label)
        scenario_folder_name = f"{os.path.splitext(os.path.basename(video_file))[0]}_{label}_{idx}"
        output_dir = os.path.join(label_folder, scenario_folder_name)
        
        extract_random_frames(
            video_path=video_file,
            start_sec=start_sec,
            end_sec=end_sec,
            output_dir=output_dir,
            label=label,
            num_frames=30
        )

if __name__ == "__main__":
    main()
