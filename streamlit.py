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
# 2. Panel Kontrol
# ======================================
st.sidebar.header("⚙️ Parameter Model")
time_steps = st.sidebar.selectbox("Jumlah Bulan Lookback", [6, 12, 24], index=1)
epochs = st.sidebar.slider("Jumlah Epoch", 50, 300, 100)
future_months = st.sidebar.number_input("Prediksi Berapa Bulan ke Depan?", 
                                      min_value=1, max_value=36, value=12)

# Tombol untuk memulai prediksi
start_prediction = st.sidebar.button("🚀 Mulai Prediksi", type="primary")

if not start_prediction:
    st.info("Silakan atur parameter di sidebar dan klik tombol '🚀 Mulai Prediksi' untuk memulai")
    st.stop()

# ======================================
# 3. Preprocessing Data
# ======================================
scaler = RobustScaler()
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
    test_pred = scaler.inverse_transform(model.predict(X_test))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
   
    test_mae, test_mape = calculate_metrics(y_test_actual, test_pred)
except Exception as e:
    st.error(f"Error dalam evaluasi model: {str(e)}")
    st.stop()

# Tampilkan metrik - Hanya Test MAE dan Test MAPE
st.subheader("📊 Evaluasi Model")
col1, col2 = st.columns(2)
col1.metric("Test MAE", f"{test_mae:,.0f}")
col2.metric("Test MAPE", f"{test_mape:.1f}%", 
           "Baik" if test_mape < 10 else "Cukup" if test_mape < 20 else "Perlu Perbaikan")
# ======================================
# 6. Visualisasi Hasil - BAGIAN YANG DIPERBAIKI
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
