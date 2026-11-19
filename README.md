# Proyek Analisis Data: E-Commerce Public Dataset ✨

Proyek akhir kelas **Belajar Analisis Data dengan Python** (Dicoding) oleh **Arby Ali Amaludin**.
Analisis dilakukan pada data e-commerce Olist (Brasil) periode **Januari 2017 – Agustus 2018** untuk menjawab tiga pertanyaan bisnis:

1. Bagaimana tren jumlah pesanan dan pendapatan bulanan, kapan puncaknya, dan berapa pertumbuhan Jan–Ags 2018 dibanding Jan–Ags 2017?
2. Kategori produk apa yang menghasilkan pendapatan tertinggi dan terendah, dan berapa kontribusi 10 kategori teratas?
3. Seberapa besar perbedaan skor ulasan antara pesanan tepat waktu dan terlambat, dan negara bagian mana yang paling sering terlambat?

Analisis lanjutan yang digunakan: **RFM Analysis**, **Geospatial Analysis**, dan **Clustering dengan binning/manual grouping** (tanpa machine learning).

## Struktur Direktori

```
submission/
├── dashboard/
│   ├── dashboard.py        # aplikasi dashboard Streamlit
│   └── main_data.csv       # dataset bersih hasil notebook
├── data/                   # 9 berkas CSV asli E-Commerce Public Dataset
├── notebook.ipynb          # proses analisis data lengkap
├── README.md
├── requirements.txt
└── url.txt                 # tautan dashboard di Streamlit Cloud
```

## Setup Environment

### Opsi 1 – Anaconda
```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

### Opsi 2 – Shell/Terminal (venv)
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan Dashboard

Jalankan perintah berikut dari folder utama proyek (`submission/`):

```
streamlit run dashboard/dashboard.py
```

Dashboard akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

## Fitur Dashboard

- **Filter interaktif** di sidebar: rentang tanggal pembelian, negara bagian pelanggan, kategori produk, dan jumlah kategori yang ditampilkan.
- KPI utama: total pesanan, total pendapatan, rata-rata skor ulasan, dan persentase pesanan terlambat.
- Visualisasi tren bulanan, kategori produk tertinggi dan terendah, skor ulasan menurut keterlambatan, keterlambatan per negara bagian, segmentasi RFM, peta lokasi pelanggan, dan binning nilai pesanan.

## Menjalankan Notebook

Buka `notebook.ipynb` dengan Jupyter Notebook, JupyterLab, VS Code, atau Google Colab. Notebook membaca data dari folder `data/` dan menyimpan dataset bersih ke `dashboard/main_data.csv`.
