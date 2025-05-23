import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import plotly.express as px

# Konfigurasi Aplikasi
st.set_page_config(page_title="📅 Prediksi Wisatawan - LSTM", layout="wide")
st.title('📅 Prediksi Jumlah Wisatawan dengan LSTM')

# ======================================
# 1. Load Data dari GitHub
# ======================================
url = "https://raw.githubusercontent.com/AnggunUwU/prediksi-wisata-mancanegara-lstm/main/data.xlsx"

try:
    df = pd.read_excel(url, engine='openpyxl')
    df['Tahun-Bulan'] = pd.to_datetime(df['Tahun-Bulan'])
    df = df.sort_values('Tahun-Bulan')
    
    # Validasi kolom
    required_columns = ['pintu Masuk', 'Jumlah_Wisatawan', 'Tahun-Bulan']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Kolom yang diperlukan tidak ditemukan. Pastikan file memiliki kolom: {required_columns}")
        st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {str(e)}")
    st.stop()

# Tampilkan data
with st.expander("🔍 Lihat Data Historis"):
    st.dataframe(df, height=200)

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
st.sidebar.header("⚙️ Parameter Model")
time_steps = st.sidebar.selectbox("Jumlah Bulan Lookback", [6, 12, 24], index=1)
epochs = st.sidebar.slider("Jumlah Epoch", 50, 300, 100)
future_months = st.sidebar.number_input("Prediksi Berapa Bulan ke Depan?", min_value=1, max_value=36, value=12)

# ======================================
# 4. Preprocessing Data
# ======================================
scaler = RobustScaler()
data_scaled = scaler.fit_transform(df_bandara['Jumlah_Wisatawan'].values.reshape(-1, 1))

def create_dataset(data, steps):
    X, y = [], []
    for i in range(len(data)-steps):
        X.append(data[i:(i+steps), 0])
        y.append(data[i+steps, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)

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
                        callbacks=[early_stop],
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

# Gabungkan data aktual dan prediksi
df_actual = df_bandara[['Tahun-Bulan', 'Jumlah_Wisatawan']].copy()
df_actual['Type'] = 'Aktual'

df_pred = pd.DataFrame({
    'Tahun-Bulan': pred_dates,
    'Jumlah_Wisatawan': predictions.flatten(),
    'Type': 'Prediksi'
})

df_combined = pd.concat([df_actual, df_pred])

# Grafik interaktif dengan Plotly
fig = px.line(df_combined, x='Tahun-Bulan', y='Jumlah_Wisatawan', 
              color='Type', title=f'Prediksi Wisatawan untuk {bandara}',
              labels={'Jumlah_Wisatawan': 'Jumlah Wisatawan', 'Tahun-Bulan': 'Tanggal'},
              hover_data={'Jumlah_Wisatawan': ':,.0f'})

fig.update_traces(mode='lines+markers')
fig.update_layout(hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

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
    }).highlight_max(axis=0),
    height=400
)

# Ekspor hasil
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Prediksi (CSV)",
    data=csv,
    file_name=f"prediksi_wisatawan_{bandara}_{datetime.now().strftime('%Y%m%d')}.csv",
    mime='text/csv'
)
