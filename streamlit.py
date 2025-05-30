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
    url = "https://github.com/AnggunUwU/prediksi-wisata-mancanegara-lstm/raw/main/data.xlsx"
    
    try:
        # Baca semua sheet dan gabungkan
        all_sheets = pd.read_excel(url, sheet_name=None)
        df = pd.concat(all_sheets.values(), ignore_index=True)
        
        # Bersihkan data
        df = df.dropna(subset=['Pintu Masuk'])
        df = df[df['Pintu Masuk'] != 'Pintu Masuk']  # Hapus baris header yang terduplikat
        
        # Ubah ke format long jika diperlukan
        if 'Januari' in df.columns:  # Jika masih format wide
            df = df.melt(
                id_vars=['Pintu Masuk', 'Tahun'],
                value_vars=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                           'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'],
                var_name='Bulan',
                value_name='Jumlah_Wisatawan'
            )
            bulan_mapping = {m: i+1 for i, m in enumerate([
                'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
            ])}
            df['Bulan'] = df['Bulan'].map(bulan_mapping)
            df['Tahun-Bulan'] = pd.to_datetime(
                df['Tahun'].astype(str) + '-' + df['Bulan'].astype(str) + '-01'
            )
        
        return df.sort_values(['Pintu Masuk', 'Tahun-Bulan'])
    
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ======================================
# 2. Visualisasi Total Tahunan - DIUBAH UKURAN
# ======================================
st.header("1. Tren Total Wisatawan Tahunan")

# Hitung total tahunan (tetap sama)
df_tahunan = df.groupby('Tahun')['Jumlah_Wisatawan'].sum().reset_index()
df_tahunan.columns = ['Tahun', 'Total']

# Tampilkan metrik utama (tetap sama)
col1, col2, col3 = st.columns(3)
col1.metric("Tahun Terakhir", df_tahunan['Tahun'].max())
col2.metric("Total Wisatawan Terakhir", f"{df_tahunan['Total'].iloc[-1]:,.0f}")
growth = (df_tahunan['Total'].iloc[-1] - df_tahunan['Total'].iloc[-2]) / df_tahunan['Total'].iloc[-2] * 100
col3.metric("Pertumbuhan (%)", f"{growth:.1f}%")

# Perubahan utama di sini - ukuran figure diperkecil
fig1, ax1 = plt.subplots(figsize=(10, 4))  # Diubah dari (12,6) ke (10,4)

ax1.bar(df_tahunan['Tahun'], df_tahunan['Total'], color='#1f77b4', width=0.6)  # Width bar disempitkan
ax1.plot(df_tahunan['Tahun'], df_tahunan['Total'], color='#ff7f0e', marker='o', markersize=5)  # Marker diperkecil

# Formatting yang lebih compact
ax1.set_title('Total Wisatawan Mancanegara per Tahun', fontsize=14, pad=10)  # Font dan padding dikurangi
ax1.tick_params(axis='both', which='major', labelsize=10)  # Ukuran tick dikurangi

# Nilai di atas bar disesuaikan
for index, row in df_tahunan.iterrows():
    ax1.text(row['Tahun'], row['Total'], f"{int(row['Total']/1000):.0f}k",  # Format angka disederhanakan
             ha='center', va='bottom', fontsize=9)  # Font size dikurangi

st.pyplot(fig1, bbox_inches='tight')  # bbox_inches untuk meminimalisir whitespace

# ======================================
# 3. Visualisasi Top 10 Pintu Masuk - DIUBAH UKURAN
# ======================================
st.header("2. Top 10 Pintu Masuk Wisatawan")

# Data preparation (tetap sama)
df_filtered = df[df['Pintu Masuk'].str.lower().isin([x.lower() for x in airport_names])]
df_top10 = df_filtered.groupby('Pintu Masuk')['Jumlah_Wisatawan'].sum().nlargest(10).reset_index()
df_top10 = df_top10.sort_values('Jumlah_Wisatawan', ascending=True)

# Perubahan utama di sini
fig2, ax2 = plt.subplots(figsize=(8, 5))  # Diubah dari (12,8) ke (8,5)

# Batang grafik dibuat lebih compact
bar_height = 0.6  # Tinggi bar dikurangi
ax2.barh(df_top10['Pintu Masuk'], df_top10['Jumlah_Wisatawan'], 
        height=bar_height, color=plt.cm.tab10(range(10)))

# Formatting yang lebih rapat
ax2.set_title('10 Pintu Masuk Wisatawan Terbanyak', fontsize=14, pad=10)
ax2.tick_params(axis='both', labelsize=9)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

# Nilai di bar disesuaikan
for i, v in enumerate(df_top10['Jumlah_Wisatawan']):
    ax2.text(v, i, f" {int(v/1000):.0f}k", va='center', fontsize=9)

st.pyplot(fig2, bbox_inches='tight')

# ======================================
# 4. Visualisasi Tren Bulanan - DIUBAH UKURAN
# ======================================
st.header("3. Tren Wisatawan Bulanan")

# Data preparation (tetap sama)
monthly_avg = df.groupby(['Bulan', 'Bulan_Nama'])['Jumlah_Wisatawan'].mean().reset_index()
monthly_avg = monthly_avg.sort_values('Bulan')

# Perubahan utama di sini
fig3, ax3 = plt.subplots(figsize=(9, 4))  # Diubah dari (12,6) ke (9,4)

# Line plot yang lebih compact
ax3.plot(monthly_avg['Bulan_Nama'], monthly_avg['Jumlah_Wisatawan'], 
        marker='o', markersize=4, linewidth=1.5, color='#2ca02c')

# Formatting
ax3.set_title('Rata-Rata Wisatawan per Bulan', fontsize=14, pad=10)
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.tick_params(axis='y', labelsize=9)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.set_ylim(0, monthly_avg['Jumlah_Wisatawan'].max() * 1.1)

# Nilai titik disederhanakan
for i, row in monthly_avg.iterrows():
    if i % 2 == 0:  # Hanya tampilkan label setiap 2 bulan agar tidak terlalu padat
        ax3.text(row['Bulan_Nama'], row['Jumlah_Wisatawan'], f"{int(row['Jumlah_Wisatawan']/1000):.1f}k",
                 ha='center', va='bottom', fontsize=8)

st.pyplot(fig3, bbox_inches='tight')
# ======================================
# 5. Informasi Data
# ======================================
st.header("📋 Informasi Dataset")
st.write(f"**Periode Data:** {df['Tahun'].min()} - {df['Tahun'].max()}")
st.write(f"**Jumlah Pintu Masuk:** {df['Pintu Masuk'].nunique()}")
st.write(f"**Total Data:** {len(df):,} observasi")

with st.expander("🔍 Lihat Contoh Data"):
    st.dataframe(df.sample(10, random_state=42))
