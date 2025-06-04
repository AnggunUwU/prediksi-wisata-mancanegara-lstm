ini adalah kodingan streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Konfigurasi Aplikasi
st.set_page_config(page_title="📅 Prediksi Wisatawan - LSTM", layout="wide")
st.title('📅 Prediksi Jumlah Wisatawan per Pintu Masuk')

# Tambahkan penjelasan tentang parameter
with st.expander("ℹ️ Panduan Penggunaan"):
    st.markdown("""
    ### 🎛️ Panduan Parameter:
    
    **Jumlah Bulan Lookback:**
    - Pilihan: 3, 6, 9, 12, 18, atau 24 bulan
    - Default: 12 bulan (optimal untuk pola tahunan)
    - Nilai lebih tinggi untuk pola jangka panjang
    
    **Jumlah Epoch:**
    - Range: 50-300
    - Default: 100
    - Lebih tinggi = lebih akurat tapi lebih lama
    
    **Bulan Prediksi:**
    - Pilihan: Range 1 - 24 bulan
    - Default: 12 bulan
    
    ### 🛠️ Cara Penggunaan:
    1. Pilih pintu masuk
    2. Atur parameter
    3. Klik 'Jalankan Model'
    4. Lihat hasil di tab Prediksi
    5. Download hasil jika perlu
    """)

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
st.subheader("🔘 Pilih Data")
pintu_masuk = df['Pintu Masuk'].unique()
selected_pintu = st.selectbox("Pintu Masuk", pintu_masuk)

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
# 2. Panel Kontrol - DI MAIN AREA
# ======================================
st.subheader("⚙️ Parameter Model")

# Buat dalam bentuk columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Konfigurasi Model**")
    time_steps = st.selectbox(
        "Jumlah Bulan Lookback",
        options=[3, 6, 9, 12, 18, 24],
        index=3,  # Default ke 12 bulan
        help="Jumlah bulan sebelumnya yang digunakan untuk prediksi"
    )
    
    # Validasi lookback
    if time_steps >= len(df_filtered):
        st.error(f"⚠️ Lookback ({time_steps} bulan) melebihi data historis ({len(df_filtered)} bulan)")
        st.stop()

with col2:
    st.markdown("**🔄 Pelatihan Model**")
    epochs = st.slider(
        "Jumlah Epoch", 
        min_value=50,
        max_value=300,
        help="Jumlah iterasi pelatihan model"
    )

with col3:
    st.markdown("**🔮 Prediksi**")
    future_months = st.number_input(
        "Prediksi Berapa Bulan ke Depan?",
        min_value=1, 
        max_value=36, 
        value=12
    )

# Tombol Aksi
st.markdown("---")
run_model = st.button("🚀 Jalankan Model", type="primary", use_container_width=True)

if not run_model:
    st.info("ℹ️ Silakan atur parameter dan klik 'Jalankan Model' untuk memulai prediksi")
    st.stop()
# ======================================
# 3. Preprocessing Data
# ======================================
with st.spinner('🔨 Mempersiapkan data...'):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

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

progress_bar = st.progress(0)
status_text = st.empty()

for epoch in range(epochs):
    history = model.fit(
        X_train, y_train,
        epochs=1,
        validation_data=(X_test, y_test),
        verbose=0
    )
    progress = (epoch + 1) / epochs
    progress_bar.progress(progress)
    status_text.text(f"⏳ Training model: Epoch {epoch+1}/{epochs} selesai")

progress_bar.empty()
status_text.empty()

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

# Tampilkan metrik - YANG SUDAH DIPERBAIKI
st.subheader("📊 Evaluasi Model")
col1, col2 = st.columns(2)
col1.metric("Test MAE", f"{test_mae:,.0f}")
col2.metric("Test MAPE", f"{test_mape:.1f}%",
           "Baik" if test_mape < 10 else "Cukup" if test_mape < 20 else "Perlu Perbaikan")

# ======================================
# 6. Visualisasi Hasil
# ======================================
st.subheader("📈 Hasil Prediksi")

try:
    # Tab 1: Training vs Test
    tab1, tab2 = st.tabs(["📉 Training vs Test", "🔮 Prediksi Masa Depan"])

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
        # Prediksi masa depan
        last_sequence = data_scaled[-time_steps:]
        predictions = []

        for _ in range(future_months):
            next_pred = model.predict(last_sequence.reshape(1, time_steps, 1), verbose=0)
            predictions.append(next_pred[0,0])
            last_sequence = np.append(last_sequence[1:], next_pred)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
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

        # Anotasi nilai prediksi
        for i, (date, pred) in enumerate(zip(pred_dates, predictions)):
            if i % max(1, future_months//6) == 0 or i == len(pred_dates)-1:
                ax2.text(date, pred[0], f"{int(pred[0]):,}",
                         ha='center', va='bottom', fontsize=9)

        ax2.set_title(f'Prediksi {future_months} Bulan ke Depan - {selected_pintu}')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        # Tabel hasil
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
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name=f"prediksi_{selected_pintu.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True
        )

except Exception as e:
    st.error(f"Error dalam visualisasi: {str(e)}")

dan dibawah adalah kodingan ipnyb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
# Membaca file Excel untuk setiap tahun
data_2017 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2017.xlsx',header=0)
data_2018 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2018.xlsx',header=0)
data_2019 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2019.xlsx',header=0)
data_2020 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2020.xlsx',header=0)
data_2021 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2021.xlsx',header=0)
data_2022 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2022.xlsx',header=0)
data_2023 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2023.xlsx',header=0)
data_2024 = pd.read_excel('/content/drive/MyDrive/skripsi/pintu/wisman2024.xlsx',header=0)

# Menggabungkan data dari semua tahun
df = pd.concat([data_2017, data_2018, data_2019, data_2020, data_2021, data_2022, data_2023, data_2024], axis=0)
df = df.replace('-',0)
df = df.fillna(0)
df = df.replace(',','')
df['Tahun'] = df['Tahun'].astype(int)
# Memilih kolom numerik
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Menambahkan kolom 'Tahunan' yang berisi jumlah nilai setiap baris hanya dari kolom numerik
# Menggunakan .sum(axis=1) untuk menjumlahkan nilai-nilai dalam setiap baris dari kolom numerik
df['Tahunan'] = df[numeric_columns].sum(axis=1)
df['Pintu Masuk'] = df['Pintu Masuk'].str.lower().str.strip()
df_save = df[df['Pintu Masuk'].isin(['total'])]
df_save.to_excel('data.xlsx'),
df0=pd.read_excel('data.xlsx')
# Mengubah format data dari bentuk lebar (wide) ke panjang (long)
df0 = df0.melt(id_vars=['Pintu Masuk', 'Tahun'],
                  value_vars=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli',
                              'Agustus', 'September', 'Oktober', 'November', 'Desember'],
                  var_name='Bulan', value_name='Jumlah_Wisatawan')
bulan_mapping = {"Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
                 "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12}
df0['Bulan'] = df0['Bulan'].map(bulan_mapping)
# Mengonversi kolom 'Tahun' ke string, lalu menggabungkan dengan kolom 'Bulan'
df0['Tahun-Bulan'] = pd.to_datetime(df0['Tahun'].astype(str) + '-' + df0['Bulan'].astype(str) + '-01')
df0 = df0.sort_values('Tahun-Bulan')
data = df0['Jumlah_Wisatawan'].values.reshape(-1, 1)
# 2. Normalisasi
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
# 3. Membuat Dataset
def create_dataset(data, time_steps=12):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:(i+time_steps), 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, time_steps=12)
X = X.reshape(X.shape[0], X.shape[1], 1)
# 4. Split Data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build Model - Correct for time_steps=1
model = Sequential([
    LSTM(64, activation='relu', input_shape=(12, 1), return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
# 6. Training
history = model.fit(X_train, y_train, epochs=80, batch_size=32,
                   validation_data=(X_test, y_test), verbose=1)

# 7. Prediksi
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# Inverse Scaling
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8. Evaluasi
def calculate_mape(actual, predicted):
    # Mengabaikan data dengan nilai nol
    non_zero_indices = actual != 0
    actual_non_zero = actual[non_zero_indices]
    predicted_non_zero = predicted[non_zero_indices]
    return np.mean(np.abs((actual_non_zero - predicted_non_zero) / actual_non_zero)) * 100

train_mae = mean_absolute_error(y_train_actual, train_predict)
test_mae = mean_absolute_error(y_test_actual, test_predict)
train_mape = calculate_mape(y_train_actual, train_predict)
test_mape = calculate_mape(y_test_actual, test_predict)

print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Train MAPE: {train_mape:.2f}%")
print(f"Test MAPE: {test_mape:.2f}%")

# Ambil 12 bulan terakhir sebagai input prediksi
last_12_months = df0['Jumlah_Wisatawan'].values[-12:].reshape(1, -1, 1)  # Data sudah dinormalisasi
last_12_months_scaled = scaler.transform(last_12_months.reshape(-1, 1)).reshape(1, 12, 1)
# Fungsi prediksi multi-step
def predict_next_six_months(model, initial_input):
    predictions = []
    current_input = initial_input.copy()

    for _ in range(6):  # Prediksi 6 bulan ke depan
        next_pred_scaled = model.predict(current_input, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
        predictions.append(next_pred)
        # Update input untuk prediksi berikutnya
        current_input = np.append(current_input[:, 1:, :], next_pred_scaled.reshape(1, 1, 1), axis=1)

    return predictions
# Dapatkan prediksi
monthly_predictions = predict_next_six_months(model, last_12_months_scaled)
months_2025 = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni']
# Tampilkan hasil
print("=== Prediksi Kedatangan Wisatawan 2025 ===")
for month, pred in zip(months_2025, monthly_predictions):
    print(f"{month}: {int(pred):,} wisatawan")
# Timeline untuk training, test, dan prediksi 2025
train_dates = df0['Tahun-Bulan'].iloc[12 : split+12]  # 12 bulan pertama jadi input
test_dates = df0['Tahun-Bulan'].iloc[split+12 : split+12+len(test_predict)]
prediction_date = pd.to_datetime('2025-06-01')
