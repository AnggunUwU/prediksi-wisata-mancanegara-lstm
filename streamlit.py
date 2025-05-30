import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Konfigurasi Aplikasi
st.set_page_config(page_title="📊 Analisis Wisatawan Mancanegara", layout="wide")
st.title('📊 Analisis Data Wisatawan Mancanegara')

# ======================================
# 1. Load Data
# ======================================
@st.cache_data
def load_data():
    # Menggunakan raw URL dari GitHub
    url = "https://github.com/AnggunUwU/prediksi-wisata-mancanegara-lstm/raw/main/Hasil%20Gabung.xlsx"
    
    try:
        # Membaca semua sheet dari file Excel
        dfs = pd.read_excel(url, sheet_name=None)
        
          # Preprocessing data
        df = df.replace('-', 0)
        df = df.fillna(0)
        df = df.replace(',', '')
        df['Tahun'] = df['Tahun'].astype(int)
        
        # Memilih kolom numerik
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Menambahkan kolom 'Tahunan'
        df['Tahunan'] = df[numeric_columns].sum(axis=1)
        df['Pintu Masuk'] = df['Pintu Masuk'].str.lower().str.strip()
        
        return df
    
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ======================================
# 2. Visualisasi Total Tahunan
# ======================================
st.header("1. Tren Total Wisatawan Tahunan")

# Hitung total tahunan
if 'Tahun' in df.columns and 'Jumlah_Wisatawan' in df.columns:
    df_tahunan = df.groupby('Tahun')['Jumlah_Wisatawan'].sum().reset_index()
    df_tahunan.columns = ['Tahun', 'Total']
    
    # Tampilkan metrik utama
    col1, col2, col3 = st.columns(3)
    col1.metric("Tahun Terakhir", df_tahunan['Tahun'].max())
    col2.metric("Total Wisatawan Terakhir", f"{df_tahunan['Total'].iloc[-1]:,.0f}")
    
    if len(df_tahunan) > 1:
        growth = (df_tahunan['Total'].iloc[-1] - df_tahunan['Total'].iloc[-2]) / df_tahunan['Total'].iloc[-2] * 100
        col3.metric("Pertumbuhan (%)", f"{growth:.1f}%")
    else:
        col3.metric("Pertumbuhan (%)", "N/A")
    
    # Buat visualisasi
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(df_tahunan['Tahun'], df_tahunan['Total'], color='#1f77b4')
    ax1.plot(df_tahunan['Tahun'], df_tahunan['Total'], color='#ff7f0e', marker='o')
    
    # Formatting
    ax1.set_title('Total Wisatawan Mancanegara per Tahun', fontsize=16, pad=20)
    ax1.set_xlabel('Tahun', fontsize=12)
    ax1.set_ylabel('Jumlah Wisatawan', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Tambahkan nilai di atas setiap bar
    for index, row in df_tahunan.iterrows():
        ax1.text(row['Tahun'], row['Total'], f"{int(row['Total']):,}", 
                 ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig1)
else:
    st.warning("Kolom 'Tahun' atau 'Jumlah_Wisatawan' tidak ditemukan dalam dataset")

# ======================================
# 3. Visualisasi Top 10 Pintu Masuk
# ======================================
st.header("2. Top 10 Pintu Masuk Wisatawan")

if 'Pintu Masuk' in df.columns and 'Jumlah_Wisatawan' in df.columns:
    # Daftar bandara utama yang akan difilter
    airport_names = [
        'ngurah rai', 'soekarno-hatta', 'juanda', 'kualanamu', 'husein sastranegara',
        'adi sucipto', 'bandara int. lombok', 'sam ratulangi', 'minangkabau',
        'sultan syarif kasim ii', 'sultan iskandar muda', 'ahmad yani', 'supadio',
        'hasanuddin', 'sultan badaruddin ii', 'hang nadim', 'sepinggan', 'sultan mahmud badaruddin ii'
    ]
    
    # Filter data dan hitung total
    df_filtered = df[df['Pintu Masuk'].str.lower().isin([x.lower() for x in airport_names])]
    df_top10 = df_filtered.groupby('Pintu Masuk')['Jumlah_Wisatawan'].sum().nlargest(10).reset_index()
    df_top10.columns = ['Pintu Masuk', 'Total']
    df_top10 = df_top10.sort_values('Total', ascending=True)
    
    # Buat visualisasi
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.barh(df_top10['Pintu Masuk'], df_top10['Total'], color=plt.cm.tab10(range(10)))
    
    # Formatting
    ax2.set_title('10 Pintu Masuk Wisatawan Terbanyak (Total Historis)', fontsize=16, pad=20)
    ax2.set_xlabel('Total Wisatawan', fontsize=12)
    ax2.set_ylabel('Pintu Masuk', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Tambahkan nilai di setiap bar
    for i, v in enumerate(df_top10['Total']):
        ax2.text(v, i, f" {int(v):,}", color='black', va='center', fontsize=10)
    
    st.pyplot(fig2)
else:
    st.warning("Kolom 'Pintu Masuk' atau 'Jumlah_Wisatawan' tidak ditemukan dalam dataset")

# ======================================
# 4. Visualisasi Tren Bulanan
# ======================================
st.header("3. Tren Wisatawan Bulanan")

if 'Tahun-Bulan' in df.columns and 'Jumlah_Wisatawan' in df.columns:
    # Ekstrak bulan dari kolom Tahun-Bulan
    df['Bulan'] = df['Tahun-Bulan'].dt.month
    df['Bulan_Nama'] = df['Tahun-Bulan'].dt.month_name()
    
    # Hitung rata-rata bulanan
    monthly_avg = df.groupby(['Bulan', 'Bulan_Nama'])['Jumlah_Wisatawan'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('Bulan')
    
    # Buat visualisasi
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(monthly_avg['Bulan_Nama'], monthly_avg['Jumlah_Wisatawan'], 
            marker='o', color='#2ca02c', linewidth=2)
    
    # Formatting
    ax3.set_title('Rata-Rata Jumlah Wisatawan per Bulan (Semua Tahun)', fontsize=16, pad=20)
    ax3.set_xlabel('Bulan', fontsize=12)
    ax3.set_ylabel('Rata-Rata Wisatawan', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Tambahkan nilai di setiap titik
    for i, row in monthly_avg.iterrows():
        ax3.text(row['Bulan_Nama'], row['Jumlah_Wisatawan'], f"{int(row['Jumlah_Wisatawan']):,}", 
                 ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig3)
else:
    st.warning("Kolom 'Tahun-Bulan' atau 'Jumlah_Wisatawan' tidak ditemukan dalam dataset")

# ======================================
# 5. Informasi Data
# ======================================
st.header("📋 Informasi Dataset")

if not df.empty:
    if 'Tahun' in df.columns:
        st.write(f"**Periode Data:** {df['Tahun'].min()} - {df['Tahun'].max()}")
    
    if 'Pintu Masuk' in df.columns:
        st.write(f"**Jumlah Pintu Masuk:** {df['Pintu Masuk'].nunique()}")
    
    st.write(f"**Total Data:** {len(df):,} observasi")
    
    with st.expander("🔍 Lihat Contoh Data"):
        st.dataframe(df.sample(10, random_state=42))
else:
    st.warning("Dataset kosong atau tidak dapat dimuat")
