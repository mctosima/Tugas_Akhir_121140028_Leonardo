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

### 11 - 17
- Mengerjakan revisi BAB 2 laporan Tugas Akhir.
- Persiapan mengikuti test sertifikasi DCML.
- Catatan Pak Martin: [link](https://getupnote.com/share/notes/IodsClZLz1Z6O49Xs9Dbzh6mKUS2/f0899ab5-afe3-4e2e-bb59-a401a13c3cb4)

### 18 - 24
- Fokus menegerjakan test sertifikasi DCML

### 25 - 31
- Mengerjakan revisi Laporan TA Bab 1, Bab 2 dan mulai menyicil Bab 3

## Juni 2025

### 1 - 7
- Mengerjakan revisi Laporan TA Bab 1, Bab 2 dan mulai menyicil Bab 3

### 8 - 14
- Persiapan Seminar Proposal
- Pada tgl 11 Juni jam 16.00 WIB melaksanakan seminar proposal

### 15 - 21
- Belum mengerjakan apapun yang berhubungan dengan Tugas Akhir

### 22 - 30
- Mulai mengerjakan revisi Tugas Akhir

## Juli 2025
### 01 - 05
- Melakukan bimbingan revisi seminar proposal dengan Pak Martin.

### 06 - 12
- Melakukan bimbingan penulisan laporan dan revisi seminar proposal dengan Bu Leslie.

### 13 - 19
- Menyelesaikan revisi seminar proposal dari catatan pak Indro
- Awalnya diminta perbandingan jumlah parameter dan flops antara model yang dikembangkan dengan model penelitian terdahulu. Penelitian ini bertipe Post teori. Pengukuran jumlah parameter dari torch.summary, pengukuran flops dari “epikorn” punya facebook untuk menghitung berapa floping point operation. Jadi pengukuran efisiensi menggunakan perbandingan jumlah parameter dan flopping point, sementara pengukuran efektifitas menggunakan confusion matrix (akurasi). (kemarin disarankan mempelajari “FVcore flops”). 
- Ditanya pak indra:  apakah tujuan penelitian saya itu merancang kerangka kerja baru atau sekedar penerapan suatu algorithm untuk teknologi deteksi posisi jatuh saja? 
- Lalu singgung bahwa penelitian saya hanya sebatas modelling, tidak sampai deployment pada alat IOT atau semacamnya. Jika begitu, maka tidak perlu ada pengukuran efisiensi, cukup tampilkan dan bahas nilai confusion matrix nya modelmu saja, tidak perlu ada perbandingan, namun tetap perlu melakukan kajian teori / kajian Pustaka saja, kenapa saya memakai Metodi ini dibandingkan Metode yang lain, kenapa saya menggunakan teknologi itu
- Penelitian yang sebatas modelling, biasanya hanya berfokus pada metrix performance yang berkaitan dengan efektifitas. Karena nanti ada Teknik nya lagi misalkan ternyata modelnya terlalu berat sehingga perlu teknik untuk mengurangkan nya saat deployment.
- Jelaskan di bab 3 pakai bagan, batasannya sampai mana. 
- Saya perlu melakukan ”data understanding”. Harus didata secara spesifik kegiatan subjek dalam scene video, misal scene org jalan berapa kali, duduk berapa kali, berbaring di Kasur berapa kali, begitu juga dengan kegiatan jatuh, misal “jatuh dari berdiri” Ganti sebutanya kayak merunduk berapa kali, “jatuh dari kursi” harus diganti juga istilahnya berapa kali, “jatuh dari berjalan” harus diganti juga istilahnya berapa kali.
- Berbaring harus dianggap sebagai jatuh, kecuali berbaring di Kasur. Hal itu karena terdapat kemungkinan besar frame yang didapat adalah keadaan hampir atau berbaring penuh, lalu seandainya dideploy, pasti frame yang akan diproses per detik itu paling tidak 3 frame (project pak indra dengan pihak korea Selatan aja segitu). Jika 30 frame yang diproses dalam 1 detik maka alatnya pasti jebol, rusak. Untuk sekarang, silahkan deskripsikan aktivitas dan label sesuka saya, misal berbaring di lantai sebagai jatuh, berbaring di Kasur sebagai tidak jatuh. Mungkin buat statistic saja
- Perlu ditambahkan penjelasan mengenai transfer learning, “zero sub” dan fine-tuning di bab 2 karena penggunaan pre-train model dari kerangka kerja mediapipe untuk mendeteksi pose landmark sudah termasuk ke transfer learning kategori zero-sub. Tambahin dulu di laporan subbab 2.2.5 transfer learning yang juga membahas sedikit tentang “zero sub” dan fine tuning
- Melakukan bimbingan penulisan laporan dengan bu Leslie.

### 20 - 26
- Menyelesaikan revisi seminar proposal dari catatan Pak Andre
- Perbaikan minor selanjutnya hanya menambahkan penjelasan sub judul tiap bab pada subbab sistematika penulisan. 

### 27 - 31
- Belum mengerjakan apapun yang berhubungan dengan Tugas Akhir


## Agustus 2025
### 01 - 02
- Belum mengerjakan apapun yang berhubungan dengan Tugas Akhir

### 03 - 09
- Mempelajari kembali alur program akhir dengan membuat beberapa flowchart di Notion Tugas Akhir

### 10 - 16
- Menambahkan rancangan ekstraksi baru dalam file meta excel (belum selesai)
- Menambahkan fitur pendataan frame yang tidak dapat diekstrak menjadi skeleton dalam program "extract_skeleton.py"
- Menambahkan program baru "dataset_rebalancing.py" untuk menyamakan total data masing-masing kelas

### 17 - 23
- program ekstrak frame dijalankan kembali, video fall_1 dihilangkan karena terdapat 2 orang, penambahan rentang waktu baru untuk ADL.

### 24 - 31
- dalam excel meta, data untuk ekstrak ADL sudah disamakan jumlahnya (54 x 5 = 270) dengan jumlah Fall (267 karena ada 3 cam yang hilang).
- program train model sudah ditambahkan fitur variasi kombinasi hyperparameter. (saat ini batch_size, learning_rate, dropout_rate, residuals)
- pembuatan fitur pencatatan hasil pelatihan (log_training.py)
- diskusi dengan pak martin terkait permasalahan variasi fitur yang mengakibatkan pelatihan yang terlalu lama.
- lakukan ablation study: melakukan pengujian berupa pelatihan dengan variasi hyperparameter yang dibatasi (sedikit saja) menggunakan 1 fold untuk melihat pengaruh tiap hyperparameter terhadap pelatihan dan menemukan kombinasi hyperparameter terbaik.
- gunakan early stopping pada pelatihan, jangan berpatokan pada jumlah epoch
- tambahkan random_seed pada penghapusan data ADL
- tambahkan fitur variasi optimizer pada pelatihan.
  - rencara ablation study:
    - perbandingan optimizer
    - perbandingan batch_size
    - perbandingan learning_rate
    - perbandingan num_layers
- penambahan random_seed pada program dataset_rebalancing.py
- program utama improved_train_gcn_kfold_v2.py berubah nama menjadi train_gcn_kfold.py
- total data awal ADL: 54 (ada beberapa skenario dalam 1 durasi)
    - jalan: 13
    - duduk-diri: 7
    - diri-duduk: 16
    - diri-merunduk: 11
    - duduk-merunduk: 4
    - duduk (wheelchair): 3
    - ganti baju: 4
- penambahan fitur variasi optimizer
- penambahan early stopping

## September 2025
### 01 - 06
- rencana menambahkan random seed pada ekstrak frame non-fall saja karena durasi not-fall jauh lebih lama daripada fall
- mengaktifkan wandb dan running program utama untuk testing kombinasi hyperparameter optimizer, log training yang sekarang dipisah per fold, dan grafik pelatihan wandb (grafik saat ini cuma titik yang merupakan rata rata atau nilai akhir pelatihan)

### 07 - 13
- melakukan ekstrak skeleton pada data:
    - extracted_frames-with start time shift and no random seed + extract logs tahap 1 (gak jadi karena sama dengan data ke 2)
    - extracted_frames-with start time shift for fall and no random seed + extract log tahap 2
    - extracted_frames-with start time shift, random seed, extract log tahap 3
    - extracted_frames-no time shift, add random seed, fix extract logs
- telah melakukan ekstrak skeleton pada data:
    - extracted_frames-with start time shift for fall and no random seed + extract log tahap 2
    - extracted_frames-with start time shift, random seed, extract log tahap 3
    - extracted_frames-no time shift, add random seed, fix extract logs
- telah dilakukan pelatihan model terhadap ketiga dataset tersebut dan hasil nya dapat dilihat pada https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-perbandingan-3-jenis-dataset?nw=nwuserleonardosirait80
- berdasarkan perbandingan dalam grafik aggregate, pemenangnya adalah dataset pertama (extracted_frames-with start time shift for fall and no random seed + extract log tahap 2)
- saat ini saya kebingungan bagaimana cara mendukung data tersebut karena data tersebut adalah keberuntungan, tidak memakai random seed dan memakai start time shift.
- percobaan selanjutnya adalah menghapus data z dari data skeleton, apakah skeleton 2D lebih baik daripada 3D.
- pelajari Ablation study agar dapat meyakinkan bu leslie mengenai variasi hyperparameter yang di test 1 per 1, bukannya mencari kombinasi hyperparamter terbaik dari keseluruhan.
- melakukan modifikasi program agar dapat melatih model menggunakan data skeleton 2 dimensi.
    - perubahannya di program func_distance_feature.py (mayor), extract_skeleton.py (minor, baris 113), func_lm_to_graph.py (minor, baris 41-42, 76-77), train_gcn_kfold.py (minor, 732)
- melakukan pelatihan model dengan skeleton 2 dimensi, hasilnya masih sulit dibandingkan, perlu dilakukan perbandingan dalam 1 project wandb
- melakukan data rebelancing pada data skeleton 3 dimensi dan 2 dimensi untuk pelatihan ulang dan perbandingan dataset

### 14 - 20
- melakukan perbandingan pelatihan model terhadap data skeleton 3 dimensi dan 2 dimensi dapat dilihat pada https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-2-dan-3-dimensi/workspace?nw=nwuserleonardosirait80
- hasil perbandingan tersebut terlihat bahwa data 3 dimensi lebih baik daripada data 2 dimensi.
- melakukan albation study pada kelima hyperparameter
    - percobaan variasi terbaik dari masing masing hypperparameter: https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-ablation-study-iseng?nw=nwuserleonardosirait80
    - percobaan variasi dropout: https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-ablation-study-dropout?nw=nwuserleonardosirait80
    - percobaan variasi learning rate: https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-ablation-study-learning_rate?nw=nwuserleonardosirait80
    - percobaan variasi batch size: https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-ablation-study-batch_size?nw=nwuserleonardosirait80
    - percobaan variasi optimizer: https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-ablation-study-optimizer?nw=nwuserleonardosirait80
    - percobaan variasi weight decay: https://wandb.ai/leonardosirait80-itera/fall-detection-5Fold-ablation-study-weight_decays?nw=nwuserleonardosirait80
