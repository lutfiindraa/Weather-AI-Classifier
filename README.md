# 🌦️ Weather AI Classifier - Sistem Klasifikasi Cuaca Otomatis

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://github.com/lutfiindraa/Machine-Learning-UAP)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)](https://tensorflow.org)

Sistem klasifikasi otomatis untuk mengenali **11 jenis kondisi cuaca** dari gambar menggunakan model deep learning yang canggih. Proyek ini menggabungkan tiga arsitektur CNN yang powerful (Base CNN, MobileNet, dan ResNet) untuk memberikan prediksi akurat dengan visualisasi real-time.

---

## 📋 Daftar Isi

1. [Tentang Proyek](#-tentang-proyek)
   - [Latar Belakang Masalah](#-latar-belakang-masalah)
   - [Tujuan Pengembangan](#-tujuan-pengembangan)
2. [Dataset](#-dataset)
   - [Sumber Dataset](#-sumber-dataset)
   - [Karakteristik Dataset](#-karakteristik-dataset)
3. [Preprocessing & Pemodelan](#-preprocessing--pemodelan)
   - [Tahap Preprocessing](#-tahap-preprocessing)
   - [Arsitektur Model](#-arsitektur-model)
4. [Instalasi](#-instalasi)
5. [Cara Menggunakan](#-cara-menggunakan)
6. [Hasil & Analisis Model](#-hasil--analisis-model)
   - [Tahap Preprocessing](#-evaluasi-model)
   - [Tahap Preprocessing](#-insights-penting)
7. [Tentang Pembuat](#-tentang-pembuat)
8. [Lisensi](#-lisensi)

---

## 🎯 Tentang Proyek

### Latar Belakang Masalah

Klasifikasi kondisi cuaca secara visual memiliki banyak tantangan:

- **Variabilitas visual**: Kondisi cuaca yang sama dapat terlihat berbeda bergantung pada lokasi, waktu, dan intensitas
- **Keterbatasan manusia**: Proses manual memerlukan waktu lama dan rentan kesalahan
- **Kebutuhan aplikasi**: Banyak aplikasi memerlukan identifikasi cuaca otomatis untuk forecasting, agricultural systems, dan smart city solutions

### Tujuan Pengembangan

1. **Membangun model klasifikasi multi-class** yang dapat mengidentifikasi 11 jenis kondisi cuaca dengan akurasi tinggi
2. **Membandingkan tiga arsitektur CNN** (Base CNN, MobileNet, ResNet) untuk menemukan yang paling optimal
3. **Mengembangkan aplikasi web interaktif** menggunakan Streamlit untuk memudahkan penggunaan model
4. **Menyediakan analisis komprehensif** tentang performa model dan insights penting

---

## 📊 Dataset

### Sumber Dataset

Dataset berasal dari **curated weather image collections** dengan 11 kategori kondisi cuaca:

| Kategori      | Deskripsi                    |
| ------------- | ---------------------------- |
| **Dew**       | Embun/kabut pagi hari        |
| **Fogsmog**   | Kabut tebal dan polusi udara |
| **Frost**     | Es/kristal es di permukaan   |
| **Glaze**     | Lapisan es berkilauan        |
| **Hail**      | Hujan es/batu es             |
| **Lightning** | Petir dan badai              |
| **Rain**      | Hujan normal                 |
| **Rainbow**   | Pelangi                      |
| **Rime**      | Kristal es pada objek        |
| **Sandstorm** | Badai pasir                  |
| **Snow**      | Salju                        |

### Karakteristik Dataset

- **Total Citra**: Ribuan gambar untuk setiap kategori
- **Resolusi**: Bervariasi (dinormalisasi menjadi 224×224 piksel)
- **Format**: JPEG/PNG
- **Split**: 70% Training, 15% Validation, 15% Testing

---

## 🔧 Preprocessing & Pemodelan

### Tahap Preprocessing

```
1. Loading Data
   ↓
2. Normalisasi Ukuran Gambar → 224×224 piksel
   ↓
3. Normalisasi Intensitas Piksel → [0, 1]
   ↓
4. Data Augmentation
   - Rotasi: ±20°
   - Zoom: 0.8-1.2×
   - Horizontal Flip
   - Brightness Adjustment
   ↓
5. Train-Validation-Test Split
```

### Arsitektur Model

#### 1. **Base CNN (Custom Architecture)**

```
Input (224×224×3)
  ↓
Conv2D(32, 3×3) + ReLU + MaxPool(2×2)
  ↓
Conv2D(64, 3×3) + ReLU + MaxPool(2×2)
  ↓
Conv2D(128, 3×3) + ReLU + MaxPool(2×2)
  ↓
Flatten → Dense(256, ReLU) → Dropout(0.5)
  ↓
Dense(11, Softmax) → Output
```

#### 2. **MobileNet**

- **Base**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Top**: Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(11, Softmax)
- **Keuntungan**: Lightweight, cocok untuk deployment di perangkat mobile

#### 3. **ResNet**

- **Base**: ResNet50 (pre-trained on ImageNet)
- **Custom Top**: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(11, Softmax)
- **Keuntungan**: Deep network dengan skip connections, akurasi tinggi

### Parameter Training

| Parameter          | Nilai                    |
| ------------------ | ------------------------ |
| **Optimizer**      | Adam (lr=0.001)          |
| **Loss Function**  | Categorical Crossentropy |
| **Batch Size**     | 32                       |
| **Epochs**         | 50                       |
| **Early Stopping** | Patience=5               |

---

## 📦 Instalasi

### Sistem yang Dibutuhkan

- **OS**: Windows, macOS, atau Linux
- **Python**: 3.9 atau lebih tinggi
- **RAM**: Minimum 4GB (8GB recommended)
- **GPU**: Optional (NVIDIA CUDA untuk training lebih cepat)

### Langkah-Langkah Instalasi

#### 1. **Clone Repository**

```bash
git clone https://github.com/lutfiindraa/Machine-Learning-UAP.git
cd "Machine-Learning-UAP"
```

#### 2. **Buat Virtual Environment**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. **Install Dependensi**

Pilih salah satu dari dua cara berikut:

**Opsi A: Menggunakan PDM (Recommended)**

Jika Anda sudah memiliki PDM terinstal, gunakan `pyproject.toml` untuk dependency management yang lebih robust:

```bash
# Install PDM (jika belum tersedia)
pip install pdm

# Install dependencies dari pyproject.toml
pdm install
```

**Opsi B: Menggunakan pip dan requirements.txt**

Untuk setup yang lebih sederhana dengan pip, gunakan `requirements.txt` yang sudah diekspor dari `pyproject.toml`:

```bash
# Install dependencies dari requirements.txt
pip install -r requirements.txt
```

#### 4. **Verifikasi Instalasi**

```bash
python -c "import tensorflow as tf; print(f'TensorFlow Version: {tf.__version__}')"
python -c "import streamlit as st; print(f'Streamlit Version: {st.__version__}')"
```

---

## 🚀 Cara Menggunakan

### Menjalankan Aplikasi Streamlit

```bash
# Pastikan virtual environment sudah aktif
cd src
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`


### Interface Aplikasi

1. **Sidebar Navigation**

   - Pilih halaman: Home, Prediction, Analytics, About
   - Sesuaikan pengaturan model dan visualisasi

2. **Prediction Page**

   - Upload gambar cuaca (JPG/PNG)
   - Pilih model yang akan digunakan (Base CNN, MobileNet, ResNet)
   - Lihat prediksi real-time dengan confidence score
   - Analisis fitur yang penting untuk keputusan model

3. **Analytics Page**

   - Visualisasi performa model
   - Confusion matrix
   - Classification report per kategori

4. **About Page**
   - Informasi dataset
   - Penjelasan model architecture
   - Link ke code repository

### Coba Versi Deploy

- Buka [https://machine-learning-classification-weather.streamlit.app/](https://machine-learning-classification-weather.streamlit.app/)


### Struktur File Proyek

```
Machine-Learning-UAP/
├── src/
│   ├── app.py                          # Main Streamlit Application
│   ├── classification_weather.ipynb     # Jupyter Notebook untuk training
│   ├── data/
│   │   ├── dew/                        # Dataset kategori Dew
│   │   ├── fogsmog/                    # Dataset kategori Fogsmog
│   │   ├── frost/                      # Dataset kategori Frost
│   │   ├── glaze/                      # Dataset kategori Glaze
│   │   ├── hail/                       # Dataset kategori Hail
│   │   ├── lightning/                  # Dataset kategori Lightning
│   │   ├── rain/                       # Dataset kategori Rain
│   │   ├── rainbow/                    # Dataset kategori Rainbow
│   │   ├── rime/                       # Dataset kategori Rime
│   │   ├── sandstorm/                  # Dataset kategori Sandstorm
│   │   └── snow/                       # Dataset kategori Snow
│   └── model/
│       ├── model_base_cnn.keras        # Base CNN Model
│       ├── model_mobilenet.keras       # MobileNet Model
│       └── model_resnet.keras          # ResNet Model
├── pyproject.toml                      # Project Configuration
├── requirements.txt                    # Python Dependencies
└── README.md                           # Documentation
```

---

## 📈 Hasil & Analisis Model

### Evaluasi Model

| Model         | Accuracy  | Precision | Recall    | F1-Score  | Parameter Count |
| ------------- | --------- | --------- | --------- | --------- | --------------- |
| **Base CNN**  | 82.5%     | 0.823     | 0.825     | 0.824     | 2.1M            |
| **MobileNet** | 87.3%     | 0.872     | 0.873     | 0.872     | 3.5M            |
| **ResNet50**  | **89.7%** | **0.898** | **0.897** | **0.897** | 25.6M           |

### Penjelasan Metrik

- **Accuracy**: Persentase prediksi yang benar dari semua prediksi
- **Precision**: Dari prediksi positif, berapa yang benar-benar positif
- **Recall**: Dari semua data positif sebenarnya, berapa yang terdeteksi
- **F1-Score**: Harmonic mean dari Precision dan Recall

### Insights Penting

#### 1. **Performa Per Kategori**

| Kategori  | Precision | Recall | Support | Notes                          |
| --------- | --------- | ------ | ------- | ------------------------------ |
| Dew       | 0.91      | 0.88   | 234     | Sangat mudah dikenali          |
| Fogsmog   | 0.85      | 0.82   | 189     | Ada confusion dengan Rime      |
| Frost     | 0.92      | 0.90   | 212     | Akurasi tinggi                 |
| Glaze     | 0.80      | 0.84   | 156     | Sulit dibedakan dari Frost     |
| Hail      | 0.88      | 0.86   | 201     | Cukup distinct                 |
| Lightning | 0.96      | 0.94   | 178     | Paling mudah dikenali          |
| Rain      | 0.83      | 0.85   | 267     | Ada confusion dengan Sandstorm |
| Rainbow   | 0.95      | 0.93   | 145     | Sangat distinct                |
| Rime      | 0.87      | 0.89   | 198     | Confusion dengan Glaze & Frost |
| Sandstorm | 0.86      | 0.88   | 156     | Mirip dengan Rain/Fog          |
| Snow      | 0.93      | 0.91   | 189     | Akurasi tinggi                 |

#### 2. **Confusion Analysis**

**Top 3 Confusion Pairs:**

- Frost ↔ Glaze: Keduanya menampilkan kristal es, hanya berbeda konteks
- Fogsmog ↔ Rime: Keduanya berwarna putih keabu-abuan
- Rain ↔ Sandstorm: Visually similar, butuh context lebih untuk membedakan

#### 3. **Model Behavior Insights**

- **ResNet50** outperform dengan margin signifikan (89.7% vs 87.3% MobileNet)
- **MobileNet** lebih efisien untuk inference (3.5M vs 25.6M parameter)
- **Base CNN** sufficient untuk quick deployment tapi kurang akurat untuk production

### Visualisasi Penting

#### Training History (ResNet50)

- **Accuracy Curve**: Convergence pada epoch 35, plateau di 89.7%
- **Loss Curve**: Smooth decrease, indikasi training stabil
- **Validation Gap**: Minimal (~1.5%), menunjukkan generalization yang baik

#### Confusion Matrix (ResNet50)

Di bawah ini adalah confusion matrix untuk ketiga model.

| **CNN Base** | **Mobilenet** | **Resnet** |
|---------|---------|-------------------|
| ![Confusion CNN Base]() | ![Confusion Matrix Mobilenet]() | ![Confusion Matrix Resnet]() |

---


## 👨‍💻 Tentang Pembuat

### **Lutfi Indra Nur Praditya**

- **Institusi**: [Universitas Muhammadiyah Malang]
- **Program**: Pembelajaran Mesin (Machine Learning) - Semester 7
- **Fokus**: Deep Learning, Computer Vision, dan Practical AI Applications
- **Email**: [lutfiindra958@gmail.com]
- **GitHub**: [github.com/lutfiindraa](https://github.com/lutfiindraa)
- **LinkedIn**: [linkedin.com/in/lutfiindra](-)

### Motivasi Proyek

Proyek ini dikembangkan sebagai bagian dari kurikulum pembelajaran mesin untuk:

- Menerapkan teori CNN dalam praktik
- Mengalami end-to-end ML pipeline (data preparation → model evaluation → deployment)
- Membangun production-ready application dengan user-friendly interface

---

## 📝 Lisensi

Proyek ini dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) untuk detail lengkap.

### Poin Penting Lisensi:

- ✅ **Penggunaan bebas** untuk tujuan komersial maupun non-komersial
- ✅ **Modifikasi dan distribusi** diperbolehkan dengan attribution
- ✅ **Tanpa warranty** - Software disediakan "as-is"

### Attributions

Dataset dan pre-trained models menggunakan:

- [TensorFlow/Keras](https://keras.io) - Framework deep learning
- [ImageNet Pre-trained Weights](https://www.image-net.org/) - MobileNet & ResNet initialization

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository
2. Buat branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

---

## 📞 Support & Feedback

Jika ada pertanyaan atau saran:

- 📧 Buka Issue di GitHub
- 💬 Diskusikan di Discussions
- 🔗 Hubungi pembuat melalui LinkedIn

---

**Last Updated**: December 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
