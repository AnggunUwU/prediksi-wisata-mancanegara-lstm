import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Konfigurasi Aplikasi
st.set_page_config(page_title="📊 Analisis Wisatawan", layout="centered")  # Layout diubah ke centered
st.title('📊 Analisis Data Wisatawan Mancanegara')

# ======================================
# 1. Load Data (Tetap Sama)
# ======================================
@st.cache_data
def load_data():
    url = "https://github.com/AnggunUwU/prediksi-wisata-mancanegara-lstm/raw/main/data.xlsx"
    try:
        all_sheets = pd.read_excel(url, sheet_name=None)
        df = pd.concat(all_sheets.values(), ignore_index=True)
        df = df.dropna(subset=['Pintu Masuk'])
        df = df[df['Pintu Masuk'] != 'Pintu Masuk']
        
        if 'Januari' in df.columns:
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
        
        df['Bulan_Nama'] = df['Tahun-Bulan'].dt.month_name()
        return df.sort_values(['Pintu Masuk', 'Tahun-Bulan'])
    
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ======================================
# 2. Compact Visualisasi Total Tahunan
# ======================================
st.header("1. Tren Tahunan Wisatawan", divider='gray')

# Hitung metrik utama
df_tahunan = df.groupby('Tahun')['Jumlah_Wisatawan'].sum().reset_index()
latest_year = df_tahunan.iloc[-1]
prev_year = df_tahunan.iloc[-2]
growth_pct = (latest_year['Jumlah_Wisatawan'] - prev_year['Jumlah_Wisatawan']) / prev_year['Jumlah_Wisatawan'] * 100

# Tampilkan metrik dalam columns
cols = st.columns(3)
cols[0].metric("Tahun Terakhir", latest_year['Tahun'])
cols[1].metric("Total Wisatawan", f"{latest_year['Jumlah_Wisatawan']:,.0f}")
cols[2].metric("Pertumbuhan", f"{growth_pct:.1f}%", 
               delta_color="normal" if growth_pct >= 0 else "inverse")

# Buat grafik yang lebih compact
fig1, ax1 = plt.subplots(figsize=(8, 3.5))  # Ukuran lebih kecil

# Warna dan styling
bar_color = '#4C72B0'
line_color = '#DD8452'

ax1.bar(df_tahunan['Tahun'], df_tahunan['Jumlah_Wisatawan'], 
       width=0.7, color=bar_color, alpha=0.7, label='Total')
ax1.plot(df_tahunan['Tahun'], df_tahunan['Jumlah_Wisatawan'], 
        marker='o', markersize=4, linewidth=1.5, color=line_color, label='Trend')

# Formatting minimalis
ax1.set_title('Total Wisatawan per Tahun', fontsize=12, pad=10)
ax1.set_ylabel('Jumlah', fontsize=9)
ax1.tick_params(axis='both', labelsize=8)
ax1.grid(axis='y', linestyle=':', alpha=0.6)

# Sederhanakan label nilai
for idx, row in df_tahunan.iterrows():
    if idx % 2 == 0 or idx == len(df_tahunan)-1:  # Label setiap 2 tahun + tahun terakhir
        ax1.text(row['Tahun'], row['Jumlah_Wisatawan'], 
                f"{row['Jumlah_Wisatawan']/1e6:.1f}M", 
                ha='center', va='bottom', fontsize=8)

st.pyplot(fig1, use_container_width=True)

# ======================================
# 3. Compact Visualisasi Top 10 Pintu Masuk
# ======================================
st.header("2. Top 10 Pintu Masuk", divider='gray')

# Siapkan data
airport_names = [
    'Ngurah Rai', 'Soekarno-Hatta', 'Juanda', 'Kualanamu', 'Husein Sastranegara',
    'Adi Sucipto', 'Bandara Int. Lombok', 'Sam Ratulangi', 'Minangkabau',
    'Sultan Syarif Kasim II'
]

df_top10 = df[df['Pintu Masuk'].isin(airport_names)]
df_top10 = df_top10.groupby('Pintu Masuk')['Jumlah_Wisatawan'].sum().nlargest(10).reset_index()
df_top10 = df_top10.sort_values('Jumlah_Wisatawan')

# Grafik horizontal compact
fig2, ax2 = plt.subplots(figsize=(7, 4))  # Lebar dikurangi

# Warna gradasi
colors = plt.cm.Blues(np.linspace(0.4, 1, len(df_top10)))

ax2.barh(df_top10['Pintu Masuk'], df_top10['Jumlah_Wisatawan'], 
        height=0.5, color=colors, edgecolor='grey', linewidth=0.5)

# Formatting
ax2.set_title('10 Pintu Masuk dengan Wisatawan Terbanyak', fontsize=12, pad=10)
ax2.set_xlabel('Total Wisatawan', fontsize=9)
ax2.tick_params(axis='both', labelsize=8)
ax2.xaxis.set_major_formatter(lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K")

# Nilai di bar
for i, val in enumerate(df_top10['Jumlah_Wisatawan']):
    ax2.text(val, i, f" {val/1e6:.1f}M" if val >= 1e6 else f" {val/1e3:.0f}K", 
            va='center', fontsize=8, color='black')

st.pyplot(fig2, use_container_width=True)
# ======================================
# 4. Compact Visualisasi Tren Bulanan
# ======================================
st.header("3. Tren Bulanan", divider='gray')

# Hitung rata-rata bulanan
monthly_avg = df.groupby(['Bulan', 'Bulan_Nama'])['Jumlah_Wisatawan'].mean().reset_index()
monthly_avg = monthly_avg.sort_values('Bulan')

# Grafik line compact
fig3, ax3 = plt.subplots(figsize=(8, 3.5))  # Ukuran lebih pendek

# Style modern
ax3.plot(monthly_avg['Bulan_Nama'], monthly_avg['Jumlah_Wisatawan'],
        marker='o', markersize=4, linewidth=1.5, color='#55A868',
        markerfacecolor='white', markeredgewidth=1)

# Formatting
ax3.set_title('Rata-Rata Kunjungan Bulanan', fontsize=12, pad=10)
ax3.set_ylabel('Wisatawan', fontsize=9)
ax3.tick_params(axis='x', rotation=45, labelsize=8)
ax3.tick_params(axis='y', labelsize=8)
ax3.grid(axis='y', linestyle=':', alpha=0.5)

# Label titik penting saja
highlight_months = [0, 5, 11]  # Jan, Jun, Des
for i in highlight_months:
    ax3.text(monthly_avg['Bulan_Nama'].iloc[i], 
            monthly_avg['Jumlah_Wisatawan'].iloc[i],
            f"{monthly_avg['Jumlah_Wisatawan'].iloc[i]/1e3:.0f}K",
            ha='center', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

st.pyplot(fig3, use_container_width=True)

# ======================================
# 5. Informasi Dataset (Compact)
# ======================================
st.header("📋 Metadata", divider='gray')

col1, col2 = st.columns(2)
with col1:
    st.metric("Periode Data", 
             f"{df['Tahun'].min()} - {df['Tahun'].max()}")
    st.metric("Jumlah Observasi", 
             f"{len(df):,}")

with col2:
    st.metric("Pintu Masuk Unik", 
             f"{df['Pintu Masuk'].nunique()}")
    st.metric("Rata-Rata Bulanan", 
             f"{df['Jumlah_Wisatawan'].mean():,.0f}")

with st.expander("🔍 Contoh Data", expanded=False):
    st.dataframe(df.sample(5, random_state=42).style.format({
        'Jumlah_Wisatawan': '{:,.0f}',
        'Tahun-Bulan': lambda x: x.strftime('%b %Y')
    }), height=150, use_container_width=True)
