# Simulasi Rekonstruksi Tomografi Interaktif (FBP vs ART)

Proyek ini berfokus pada perbandingan dua pendekatan utama:
- **Filtered Back Projection (FBP)** → metode analitik klasik
- **Algebraic Reconstruction Technique (ART)** → metode iteratif berbasis sistem persamaan linear

📌 Disusun sebagai **Tugas Pengganti Kuliah Minggu ke-7**
- Ghazy Abiyyu Maulana
- 1104220120
- Mata Kuliah: Teknik Tomografi

---

## 🌟 Latar Belakang

Dalam dunia medis (CT Scan) dan inspeksi industri, seringkali kita menghadapi keterbatasan data:
- Sudut pengambilan terbatas (**Few-View Projection**)
- Rotasi sensor tidak penuh
- Pembatasan radiasi pada pasien

Metode konvensional seperti **FBP**:
- Cepat, tetapi
- Menghasilkan artefak (garis-garis) jika data terbatas

Sebaliknya, **ART**:
- Menggunakan pendekatan iteratif
- Memodelkan sistem sebagai:

\[
Ax = b
\]

- Secara bertahap memperbaiki nilai piksel
- Menghasilkan citra yang lebih akurat pada kondisi data minim

🎯 EduTomo dibuat untuk **memvisualisasikan perbedaan ini secara interaktif dan intuitif**

---

## 🛠️ Fitur Utama

### 🔬 1. Micro-ART (Konsep Dasar)
- Visualisasi step-by-step
- Menunjukkan bagaimana satu sinar mempengaruhi piksel
- Cocok untuk memahami konsep matematis ART

### 🏥 2. Macro CT Simulation
- Simulasi CT Scan realistis
- Menggunakan **phantom image** (Shepp-Logan)

### 🔄 3. Forward Model (Sinogram)
- Simulasi proses pembentukan data sensor
- Visualisasi sinogram secara langsung

### ⚡ 4. Live Reconstruction
- Rekonstruksi FBP vs ART secara real-time
- Animasi iterasi ART
- Visualisasi pengurangan artefak

### 📊 5. Analisis ROI (Region of Interest)
- Evaluasi kualitas citra pada area tertentu
- Menggunakan metrik:
  - **SSIM** (Structural Similarity Index)
  - **RMSE** (Root Mean Square Error)

### 🗺️ 6. Visualisasi Error 3D
- Pemetaan distribusi error dalam bentuk topografi 3D
- Membantu memahami area kesalahan rekonstruksi

---

## 📖 Referensi Ilmiah

Proyek ini mengacu pada penelitian berikut:

> Kojima, T., & Yoshinaga, T. (2023).  
> *Iterative Image Reconstruction Algorithm with Parameter Estimation by Neural Network for Computed Tomography*  
> **Algorithms (MDPI), 16(1), 60**

---

## 🌐 Live Demo

Coba aplikasinya di sini:  
👉 [EduTomo Web App](https://simulasi-tomografi.streamlit.app/)

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Clone Repository
```
bash
git clone https://github.com/USERNAME/NAMA_REPOSITORY.git
cd NAMA_REPOSITORY
```
### 2. Install Dependencies
Pastikan Python sudah terinstall, lalu jalankan:
````
pip install -r requirements.txt
````
### 3. Jalankan Aplikasi
````
streamlit run tomo2.py
````
