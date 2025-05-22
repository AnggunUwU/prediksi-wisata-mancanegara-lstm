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
st.title('📅 Prediksi Jumlah Wisatawan dengan LSTM')

# ======================================
# 1. Load Data dari GitHub
# ======================================
url = "https://raw.githubusercontent.com/AnggunUwU/prediksi-wisata-mancanegara-lstm/main/data.xlsx"

try:
    df = pd.read_excel(url)
    df['Tahun-Bulan'] = pd.to_datetime(df['Tahun-Bulan'])
    df = df.sort_values('Tahun-Bulan')
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")

# Tampilkan data
with st.expander("🔍 Lihat Data Historis"):
    st.dataframe(df, height=200)

# Periksa kolom yang ada
st.write("Kolom yang tersedia:", df.columns.tolist())

# ======================================
# 2. Pilih Bandara untuk Prediksi
# ======================================
bandara_list = df['pintu Masuk'].unique()
bandara = st.selectbox("Pilih Bandara untuk Prediksi", bandara_list)

# Filter data berdasarkan bandara yang dipilih
df_bandara = df[df['pintu Masuk'] == bandara]

# ======================================
# 3. Panel Kontrol
# ======================================
col1, col2, col3 = st.columns(3)

with col1:
    time_steps = st.selectbox("Jumlah Bulan Lookback", [6, 12, 24], index=1)

with col2:
    epochs = st.slider("Jumlah Epoch", 50, 300, 100)

with col3:
    future_months = st.number_input("Prediksi Berapa Bulan ke Depan?", min_value=1, max_value=36, value=12)

# ======================================
# 4. Preprocessing Data
# ======================================
scaler = RobustScaler()  # menggunakan median & IQR, tidak terpengaruh outlier
data_scaled = scaler.fit_transform(df_bandara['Jumlah_Wisatawan'].values.reshape(-1, 1))

def create_dataset(data, steps):
    X, y = [], []
    for i in range(len(data)-steps):
        X.append(data[i:(i+steps), 0])
        y.append(data[i+steps, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Tampilkan bentuk data
st.write("Bentuk data X:", X.shape)
st.write("Bentuk data y:", y.shape)

# ======================================
# 5. Training Model
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

with st.spinner(f'Melatih model dengan {epochs} epoch...'):
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=0)

# ======================================
# 6. Evaluasi Model
# ======================================
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, mape

train_pred = scaler.inverse_transform(model.predict(X_train))
test_pred = scaler.inverse_transform(model.predict(X_test))

train_mae, train_mape = calculate_metrics(scaler.inverse_transform(y_train.reshape(-1, 1)), train_pred)
test_mae, test_mape = calculate_metrics(scaler.inverse_transform(y_test.reshape(-1, 1)), test_pred)

# Tampilkan metrik
st.subheader("📊 Evaluasi Model")
col1, col2, col3 = st.columns(3)
col1.metric("Train MAE", f"{train_mae:,.0f}")
col2.metric("Test MAE", f"{test_mae:,.0f}", delta=f"{(test_mae-train_mae)/train_mae*100:.1f}% vs Train")
col3.metric("Test MAPE", f"{test_mape:.1f}%", "Baik" if test_mape < 10 else "Cukup")

# ======================================
# 7. Visualisasi Hasil
# ======================================
st.subheader("📈 Grafik Hasil")

tab1, tab2 = st.tabs(["Training vs Test", "Prediksi Masa Depan"])

with tab1:
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(df_bandara['Tahun-Bulan'][time_steps:split+time_steps], scaler.inverse_transform(y_train.reshape(-1, 1)),
             label='Train Aktual', color='blue')
    plt.plot(df_bandara['Tahun-Bulan'][split+time_steps:], scaler.inverse_transform(y_test.reshape(-1, 1)),
             label='Test Aktual', color='green')
    plt.plot(df_bandara['Tahun-Bulan'][time_steps:split+time_steps], train_pred,
             label='Prediksi Train', linestyle='--', color='red')
    plt.plot(df_bandara['Tahun-Bulan'][split+time_steps:], test_pred,
             label='Prediksi Test', linestyle='--', color='orange')
    plt.title('Perbandingan Data Aktual vs Prediksi')
    plt.legend()
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
        start=df_bandara['Tahun-Bulan'].iloc[-1] + pd.DateOffset(months=1),
        periods=future_months,
        freq='MS'
    )

    # Tampilkan hasil
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(df_bandara['Tahun-Bulan'], df_bandara['Jumlah_Wisatawan'], label='Data Historis', color='blue')
    plt.plot(pred_dates, predictions, label='Prediksi', color='red', marker='o')

    # Anotasi nilai prediksi
    for i, (date, pred) in enumerate(zip(pred_dates, predictions)):
        if i % 3 == 0 or i == len(pred_dates)-1:  # Label setiap 3 bulan
            plt.text(date, pred[0], f"{int(pred[0]):,}",
                     ha='center', va='bottom')

    plt.title(f'Prediksi {future_months} Bulan ke Depan')
    plt.legend()
    st.pyplot(fig2)

    # Tabel hasil
    pred_df = pd.DataFrame({
        'Bulan': pred_dates.strftime('%B %Y'),
        'Prediksi': predictions.flatten().astype(int),
        'Perubahan (%)': np.insert(np.diff(predictions.flatten()) / predictions.flatten()[:-1] * 100, 0, 0)
    })

    st.dataframe(
        pred_df.style.format({
            'Prediksi': '{:,.0f}',
            'Perubahan (%)': '{:.1f}%'
        }),
        height=400
    )

    # Ekspor hasil
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediksi (CSV)",
        data=csv,
        file_name=f"prediksi_wisatawan_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
