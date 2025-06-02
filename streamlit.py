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

dan 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Konfigurasi Aplikasi
st.set_page_config(page_title="📅 Prediksi Wisatawan - LSTM", layout="wide")
st.title('📅 Prediksi Jumlah Wisatawan per Pintu Masuk')

# ======================================
# 1. Load dan Persiapkan Data
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

# Pilih Pintu Masuk
pintu_masuk = df['Pintu Masuk'].unique()
selected_pintu = st.selectbox("Pilih Pintu Masuk", pintu_masuk)

# Filter Data
df_filtered = df[df['Pintu Masuk'] == selected_pintu].sort_values('Tahun-Bulan')

# Validasi Data
if len(df_filtered) < 24:
    st.error(f"⚠️ Data historis untuk {selected_pintu} hanya {len(df_filtered)} bulan, minimal 24 bulan diperlukan")
    st.stop()

# Tampilkan data
with st.expander(f"🔍 Lihat Data Historis {selected_pintu}"):
    st.dataframe(df_filtered, height=200)

# ======================================
# 2. Panel Kontrol - Dipindahkan ke Main Content
# ======================================
st.subheader("⚙️ Parameter Model")

# Buat columns untuk layout parameter
col1, col2, col3 = st.columns(3)

with col1:
    time_steps = st.selectbox("Jumlah Bulan Lookback", [6, 12, 24], index=1)

with col2:
    epochs = st.slider("Jumlah Epoch", 50, 300, 100)

with col3:
    future_months = st.number_input("Prediksi Berapa Bulan ke Depan?", 
                                  min_value=1, max_value=36, value=12)

# Tombol untuk memulai prediksi - dipindahkan ke bawah parameter
start_prediction = st.button("🚀 Mulai Prediksi", type="primary")

if not start_prediction:
    st.info("Silakan atur parameter di atas dan klik tombol '🚀 Mulai Prediksi' untuk memulai")
    st.stop()

# ======================================
# 3. Preprocessing Data
# ======================================

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_filtered[['Jumlah_Wisatawan']])

def create_dataset(data, steps):
    X, y = [], []
    for i in range(len(data)-steps):
        X.append(data[i:(i+steps), 0])
        y.append(data[i+steps, 0])
    return np.array(X), np.array(y)

try:
    X, y = create_dataset(data_scaled, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)
except Exception as e:
    st.error(f"Error dalam preprocessing data: {str(e)}")
    st.stop()

# ======================================
# 4. Training Model
# ======================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(time_steps, 1), return_sequences=True),
    LSTM(32, activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

with st.spinner(f'Melatih model untuk {selected_pintu} ({epochs} epoch)...'):
    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=0
        )
    except Exception as e:
        st.error(f"Error saat training model: {str(e)}")
        st.stop()

# ======================================
# 5. Evaluasi Model
# ======================================
def calculate_metrics(actual, predicted):
    actual = actual.flatten()
    predicted = predicted.flatten()
    mask = actual != 0  # Hindari division by zero
    mae = mean_absolute_error(actual[mask], predicted[mask])
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return mae, mape

try:
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_mae, train_mape = calculate_metrics(y_train_actual, train_pred)
    test_mae, test_mape = calculate_metrics(y_test_actual, test_pred)
except Exception as e:
    st.error(f"Error dalam evaluasi model: {str(e)}")
    st.stop()

except Exception as e:
    st.error(f"Error dalam evaluasi model: {str(e)}")
    st.stop()


# Tampilkan metrik
st.subheader("📊 Evaluasi Model")
col1, col2 = st.columns(2)

col1.metric("Test MAE", f"{test_mae:,.0f}")

col2.metric("Test MAPE", f"{test_mape:.1f}%", 
           "Baik" if test_mape < 10 else "Cukup" if test_mape < 20 else "Perlu Perbaikan")
# ======================================
# 6. Visualisasi Hasil 
# ======================================
st.subheader("📈 Grafik Hasil")

try:
    # Tab 1: Training vs Test
    tab1, tab2 = st.tabs(["Training vs Test", "Prediksi Masa Depan"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df_filtered['Tahun-Bulan'].iloc[time_steps:split+time_steps], 
                y_train_actual, label='Train Aktual', color='blue')
        ax1.plot(df_filtered['Tahun-Bulan'].iloc[split+time_steps:], 
                y_test_actual, label='Test Aktual', color='green')
        ax1.plot(df_filtered['Tahun-Bulan'].iloc[time_steps:split+time_steps], 
                train_pred, label='Prediksi Train', linestyle='--', color='red')
        ax1.plot(df_filtered['Tahun-Bulan'].iloc[split+time_steps:], 
                test_pred, label='Prediksi Test', linestyle='--', color='orange')
        ax1.set_title(f'Perbandingan Data Aktual vs Prediksi - {selected_pintu}')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)

    with tab2:
        # Prediksi masa depan - PERBAIKAN UTAMA DI SINI
        last_sequence = data_scaled[-time_steps:]
        predictions = []

        for _ in range(future_months):
            next_pred = model.predict(last_sequence.reshape(1, time_steps, 1), verbose=0)
            predictions.append(next_pred[0,0])
            # Perbaikan: Pastikan sequence tetap memiliki panjang time_steps
            last_sequence = np.append(last_sequence[1:], next_pred)[-time_steps:]

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Debugging: Tampilkan jumlah prediksi
        st.write(f"Jumlah prediksi yang dihasilkan: {len(predictions)} (diminta: {future_months})")
        
        # Pastikan jumlah prediksi sesuai dengan yang diminta
        if len(predictions) != future_months:
            st.error(f"Jumlah prediksi ({len(predictions)}) tidak sesuai dengan yang diminta ({future_months})")
            st.stop()
        
        # Buat tanggal prediksi
        pred_dates = pd.date_range(
            start=df_filtered['Tahun-Bulan'].iloc[-1] + pd.DateOffset(months=1),
            periods=future_months,
            freq='MS'
        )

        # Plot prediksi
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df_filtered['Tahun-Bulan'], df_filtered['Jumlah_Wisatawan'], 
                label='Data Historis', color='blue')
        ax2.plot(pred_dates, predictions, 
                label='Prediksi', color='red', marker='o')
        
        # Anotasi nilai prediksi - tampilkan semua bulan
        for i, (date, pred) in enumerate(zip(pred_dates, predictions)):
            ax2.text(date, pred[0], f"{int(pred[0]):,}", 
                     ha='center', va='bottom', fontsize=9)

        ax2.set_title(f'Prediksi {future_months} Bulan ke Depan - {selected_pintu}')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        # Tabel hasil - tampilkan semua bulan
        pred_df = pd.DataFrame({
            'Bulan': pred_dates.strftime('%B %Y'),
            'Prediksi': predictions.flatten().astype(int),
            'Perubahan (%)': np.round(
                np.insert(
                    np.diff(predictions.flatten()) / predictions.flatten()[:-1] * 100, 
                0, 0
            ), 1)
        })

        st.dataframe(
            pred_df.style.format({
                'Prediksi': '{:,.0f}',
                'Perubahan (%)': '{:.1f}%'
            }).background_gradient(cmap='Blues', subset=['Perubahan (%)']),
            height=min(400, 35*future_months),
            use_container_width=True
        )

        # Ekspor hasil
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediksi (CSV)",
            data=csv,
            file_name=f"prediksi_{selected_pintu.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

except Exception as e:
    st.error(f"Error dalam visualisasi: {str(e)}")
