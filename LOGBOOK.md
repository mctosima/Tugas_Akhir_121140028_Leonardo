# Research Logbook

## Januari 2025

### 27 - 31 jan
- Mencari dataset yang sesuai dari berbagai penelitian terdahulu (sejauh ini belum nemu yang tersedia publik)
- Mempelajari berbagai jenis dataset yang telah dipakai untuk penelitian serupa, seperti Wearable-Based, Wearable-Based, dan Multimodal. [sumber](https://www.researchgate.net/publication/332730828_UP-Fall_Detection_Dataset_A_Multimodal_Approach)


## Februari 2025

### 01 - 08
- Mengerjakan proyek Praktik Kuliah Lapangan
- Mengerjakan laporan praktik kuliah Lapangan
- Belum melanjutkan tugas akhir

### 09 - 15
- Finalisasi proyek Praktik Kuliah Lapangan
- Riset dataset yang relevan, berikut hasilnya: [link gdocs](https://docs.google.com/document/d/1vt1xzBMEAnTlnfMnwdX95cKNSS1D3mnqtPFkGD9IhH8/edit?usp=sharing)

### 16 - 22
- Diskusi dengan Pak Martin, dataset yang disepakati adalah [kuleuven.app.box](https://kuleuven.app.box.com/s/dyo66et36l2lqvl19i9i7p66761sy0s6?)
- Mengerjakan serta bimbingan BAB 1, 2, 3 Laporan akhir PKL
- Belajar dasar Pytorch

### 23 - 28
- Mengerjakan revisi serta bimbingan BAB 1, 2, 3, 4 Laporan akhir PKL
- Membuat tabel deskripsi video, nama file "dataset_desc.xlsx". Tabel ini akan digunakan dalam program ekstrak frame video.
- Membuat program ekstrak frame video untuk mengambil kejadian jatuh dengan bantuan ChatGPT. Berikut 2 fungsi utama ekstraksi frame
```
def time_to_seconds(tstr):
    """
    Mengonversi string waktu (HH:MM:SS atau MM:SS) menjadi detik.
    Contoh:
      - '00:01:35' -> 95 detik
      - '01:35' -> 95 detik
    """
    parts = tstr.split(':')
    parts = [int(p) for p in parts]
    
    if len(parts) == 3:
        # format HH:MM:SS
        h, m, s = parts
    elif len(parts) == 2:
        # format MM:SS
        h = 0
        m, s = parts
    else:
        raise ValueError("Format waktu tidak dikenali. Gunakan HH:MM:SS atau MM:SS.")
        
    return h*3600 + m*60 + s
```
Fungsi di atas untuk konversi waktu kejadian ke satuan detik.

```
def extract_frames(video_path, start_sec, end_sec, output_dir, label):
    """
    Mengekstrak frame dari video mulai detik start_sec sampai end_sec.
    Disimpan di output_dir dengan penamaan berurutan.
    """
    # Pastikan output_dir ada
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Jika fps asli video tidak terlalu jauh dari fps yang kita inginkan,
    # kita bisa mengambil semua frame. Atau kita sampling sesuai 'fps' target.
    
    frame_count = 0
    # Hitung frame start dan end berdasarkan fps video
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simpan frame
        output_filename = f"{label}_{current_frame}.jpg"
        cv2.imwrite(os.path.join(output_dir, output_filename), frame)
        
        current_frame += 1
        
    cap.release()
```
Fungsi di atas untuk mengekstrak frame dalam durasi tertentu, sesuai dengan data dari "dataset_desc.xlsx".


## Maret 2025

### 1 - 8
- mengerjakan revisi serta bimbingan BAB 1, 2, 3, 4, 5, 6 Laporan akhir PKL
- Mempelajari lebih dalam program ekstraksi video.

### 9 - 15
- mengikuti bimbingan untuk mendiskusikan program ekstraksi video yang telah dibuat. program tersebut dapat mengekstraks frame dengan durasi yang ditetapkan dalam file "dataset_desc.xlsx". program tersebut ditolak karena frame yang diambil tidak acak.
- mengerjakan perbaikan program ekstraksi frame video agar dapat mengambil frame secara random dalam durasi video yang ditetapkan dalam file "dataset_desc.xlsx".

### 16 - 22
- menyelesaikan program ekstraksi frame video dan data reader tahap awal serta menjalankan keseluruhan program hingga training model.
- mengikuti bimbingan untuk mendiskusikan program ekstraksi frame video dan data reader. hasil train Dan validation tidak masuk akal (mencapai 100% accuracy), kata Pak martin terdapat kebocoran data saat split data di data reader. Solusinya menerapkan split K fold. Misal data diterapkan 5 fold, artinya datanya bagi menjadi 5 bagian, lalu modelnya ditrain dengan 4 bagian Dan divalidasi dengan sisa 1 bagian. Lakukan sebanyak 5 kali dishuffel.
- hasil ekstrak skeleton pada dataset frame hasil ekstrak tidak bagus (total hasil ekstrak skeleton 4000an sementara hasil frame asli ada 8000an). solusinya: Tambahkan di batasan Masalah bahwa "Citra yang diproses adalah citra dengan pose manusia yang jelas".


### 23 - 31
- Tidak mengerjakan apa pun yang berhubungan dengan kuliah.

## April 2025

### 1 - 5
- Mengerjakan BAB 1 Laporan Tugas Akhir di microsoft word.

### 6 - 12
- mencoba menerapkan pembagian data 5-Fold dengan merevisi program data reader dan train model. Setelah revisi, hasil train masih tetap sama, yaitu 100% dari epoch pertama sampai akhir
- Mengecek dan menghapus manual file data skeleton yang gagal dideteksi oleh Mediapipe.
- Belajar Menyusun Laporan Tugas Akhir di Overleaf.

### 13 - 19
- Mengerjakan revisi laporan Tugas Akhir.

### 20 - 26
- Membahas perbaikan program train_gcn_kfold.py dengan pak martin.
- Daftar Seminar proposal.

### 27 - 30
- Mengerjakan evisi Laporan Tugas Akhir.

## Mei 2025

### 1 - 3
- Melakukan Bimbingan Laporan Tugas Akhir dengan Bu Leslie.

### 4 - 10
- Mengerjakan perbaikan porgram train_gcn_kfold.py dengan mengubah alur pelatihan model dimana tiap fold dataset melatih model yang berbeda. (saat ini 5-Fold, 5 model berbeda).
- Mengerjakan revisi BAB 2 laporan Tugas Akhir.