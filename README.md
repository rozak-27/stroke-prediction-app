# 🧠 Deteksi Dini Risiko Stroke — Aplikasi Web Streamlit

Aplikasi web berbasis **Streamlit** untuk analisis pola dan prediksi stroke menggunakan Data Mining.

## 📋 Halaman Aplikasi

| Halaman | Isi |
|---------|-----|
| 🏠 **Home** | Judul, deskripsi proyek, identitas tim |
| 📊 **Dataset Overview** | Upload data, statistik, visualisasi EDA, training model |
| 🔮 **Prediction / Analysis** | Form input pasien, prediksi risiko stroke |
| 📈 **Visualization** | Grafik EDA, clustering, feature importance, ROC curve |
| ℹ️ **About** | Penjelasan metode, dataset, informasi proyek |

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

### 3. Buka browser
Aplikasi akan terbuka di `http://localhost:8501`

## 📁 Struktur File

```
stroke_app/
├── app.py              ← File utama aplikasi Streamlit
├── requirements.txt    ← Daftar library Python
└── README.md           ← Panduan ini
```

## 🔄 Cara Penggunaan

1. **Buka halaman Dataset Overview**
2. Upload file `healthcare-dataset-stroke-data.csv` (dari Kaggle)
   _atau_ klik **Gunakan Sample Data** untuk demo
3. Klik **Latih Model Sekarang** — tunggu 30–60 detik
4. Pergi ke **Prediction / Analysis** untuk prediksi pasien baru
5. Eksplorasi grafik di **Visualization**

## 📦 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| `streamlit` | Framework web app |
| `scikit-learn` | ML: RandomForest, KMeans, preprocessing, evaluasi |
| `imbalanced-learn` | SMOTE oversampling |
| `plotly` | Visualisasi interaktif |
| `pandas`, `numpy` | Manipulasi data |
| `shap` | Explainable AI |

## ⚙️ Dataset

**Stroke Prediction Dataset** — Kaggle (fedesoriano)  
Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## 📝 Catatan

- Model dilatih secara **real-time** di dalam aplikasi
- Jika dataset asli tidak tersedia, gunakan **Sample Data** untuk demo
- Disclaimer: hasil prediksi bersifat informatif, bukan diagnosis medis
