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

# ======================================
# 7. Tambahan Visualisasi - Grafik Total Tahunan dan Top 10 Bandara
# ======================================
st.subheader("📊 Analisis Tambahan")

# Tab untuk visualisasi tambahan
tab_analysis1, tab_analysis2 = st.tabs(["Total Tahunan Wisatawan", "Top 10 Bandara"])

with tab_analysis1:
    st.write("### Total Tahunan Wisatawan di Indonesia")
    
    # Hitung total tahunan
    df_tahunan = df.groupby('Tahun')['Jumlah_Wisatawan'].sum().reset_index()
    df_tahunan.columns = ['Tahun', 'Tahunan']
    
    # Buat plot
    fig_tahunan, ax_tahunan = plt.subplots(figsize=(10, 6))
    ax_tahunan.bar(df_tahunan['Tahun'], df_tahunan['Tahunan'], color='skyblue')
    ax_tahunan.set_ylim(0, max(df_tahunan['Tahunan']) * 1.1)
    ax_tahunan.set_title('Total Tahunan Wisatawan di Indonesia', fontsize=14)
    ax_tahunan.set_xlabel('Tahun', fontsize=12)
    ax_tahunan.set_ylabel('Total Tahunan', fontsize=12)
    ax_tahunan.grid(True, linestyle='--', alpha=0.7)
    
    # Tambahkan nilai di atas setiap bar
    for index, row in df_tahunan.iterrows():
        ax_tahunan.text(row['Tahun'], row['Tahunan'], f"{int(row['Tahunan']):,}", 
                       ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig_tahunan)

with tab_analysis2:
    st.write("### Top 10 Bandara dengan Wisatawan Terbanyak")
    
    # Daftar bandara utama
    airport_names = [
        'ngurah rai', 'soekarno-hatta', 'juanda', 'kualanamu', 'husein sastranegara',
        'adi sucipto', 'bandara int. lombok', 'sam ratulangi', 'minangkabau',
        'sultan syarif kasim ii', 'sultan iskandar muda', 'ahmad yani', 'supadio',
        'hasanuddin', 'sultan badaruddin ii'
    ]
    
    # Filter data untuk bandara dan tahun 2017-2024
    df_filtered_airports = df[
        (df['Pintu Masuk'].str.lower().isin(airport_names)) & 
        (df['Tahun'] >= 2017) & 
        (df['Tahun'] <= 2024)
    ]
    
    # Kelompokkan dan jumlahkan
    df_grouped = df_filtered_airports.groupby(['Pintu Masuk'])['Jumlah_Wisatawan'].sum().reset_index()
    df_grouped.columns = ['Pintu Masuk', 'Total']
    
    # Ambil top 10 dan urutkan
    top_10 = df_grouped.nlargest(10, 'Total').sort_values('Total', ascending=True)
    
    # Buat plot
    fig_airport, ax_airport = plt.subplots(figsize=(10, 6))
    ax_airport.barh(top_10['Pintu Masuk'], top_10['Total'], 
                   color=plt.cm.Paired(range(len(top_10))))
    
    # Format angka dan tambahkan judul
    ax_airport.set_xscale('log')
    ax_airport.set_title('Top 10 Bandara dengan Wisatawan Terbanyak (2017-2024)', fontsize=14)
    ax_airport.set_xlabel('Total Wisatawan (skala logaritmik)', fontsize=12)
    ax_airport.set_ylabel('Bandara', fontsize=12)
    ax_airport.grid(True, linestyle='--', alpha=0.7)
    
    # Tambahkan nilai di setiap bar
    for index, value in enumerate(top_10['Total']):
        ax_airport.text(value, index, f" {int(value):,}", va='center', fontsize=9)
    
    st.pyplot(fig_airport)
