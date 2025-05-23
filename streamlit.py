import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
from io import BytesIO

# Judul aplikasi
st.title('Prediksi Kedatangan Wisatawan Mancanegara dengan LSTM')

# 1. Load Data dari GitHub
@st.cache_data
def load_data():
    url = "https://github.com/AnggunUwU/prediksi-wisata-mancanegara-lstm/raw/main/data.xlsx"
    response = requests.get(url)
    excel_data = pd.ExcelFile(BytesIO(response.content))
    
    # Baca semua sheet dan gabungkan
    all_data = pd.DataFrame()
    for sheet_name in excel_data.sheet_names:
        if sheet_name.startswith('Sheet'):
            df = pd.read_excel(excel_data, sheet_name=sheet_name)
            # Ekstrak tahun dari sheet name atau kolom
            if 'Tahun' in df.columns:
                all_data = pd.concat([all_data, df], ignore_index=True)
    
    # Bersihkan data
    all_data = all_data.dropna(subset=['Pintu Masuk'])
    all_data = all_data[all_data['Pintu Masuk'] != 'Pintu Masuk']  # Hapus baris header yang terduplikat
    
    # Ubah ke format long
    df_long = all_data.melt(id_vars=['Pintu Masuk', 'Tahun'], 
                          value_vars=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                                     'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'],
                          var_name='Bulan', value_name='Jumlah_Wisatawan')
    
    bulan_mapping = {"Januari":1, "Februari":2, "Maret":3, "April":4, "Mei":5, "Juni":6,
                     "Juli":7, "Agustus":8, "September":9, "Oktober":10, "November":11, "Desember":12}
    df_long['Bulan'] = df_long['Bulan'].map(bulan_mapping)
    df_long['Tahun-Bulan'] = pd.to_datetime(df_long['Tahun'].astype(str) + '-' + df_long['Bulan'].astype(str) + '-01')
    df_long = df_long.sort_values(['Pintu Masuk', 'Tahun-Bulan'])
    
    return df_long

df = load_data()

# 2. Pilih Bandara
st.sidebar.header("Pengaturan Prediksi")
all_airports = df['Pintu Masuk'].unique()
selected_airport = st.sidebar.selectbox("Pilih Bandara/Pintu Masuk:", all_airports)

# Input untuk epoch, batch size, lookback, dan target prediksi
epochs = st.sidebar.number_input("Jumlah Epoch:", min_value=1, value=80, step=1)
batch_size = st.sidebar.number_input("Ukuran Batch:", min_value=1, value=32, step=1)
lookback = st.sidebar.number_input("Lookback (Langkah Waktu):", min_value=1, value=12, step=1)
target_months = st.sidebar.number_input("Target Bulan untuk Prediksi:", min_value=1, value=6, step=1)

# Filter data berdasarkan bandara yang dipilih
airport_data = df[df['Pintu Masuk'] == selected_airport].copy()

# 3. Preprocessing
scaler = RobustScaler()
data_scaled = scaler.fit_transform(airport_data[['Jumlah_Wisatawan']])

def create_dataset(data, time_steps=lookback):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:(i+time_steps), 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, time_steps=lookback)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4. Split Data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Bangun dan Latih Model
@st.cache_resource
def build_and_train_model(_X_train, _y_train, _X_test, _y_test):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(_X_train, _y_train, epochs=epochs, batch_size=batch_size, 
              validation_data=(_X_test, _y_test), verbose=0)
    return model

model = build_and_train_model(X_train, y_train, X_test, y_test)

# 8. Prediksi N Bulan ke Depan
def predict_next_n_months(_model, initial_input, n_months):
    predictions = []
    current_input = initial_input.copy()
    
    for _ in range(n_months):
        next_pred_scaled = _model.predict(current_input, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
        predictions.append(next_pred)
        current_input = np.append(current_input[:, 1:, :], next_pred_scaled.reshape(1, 1, 1), axis=1)
    
    return predictions

last_12_months = airport_data['Jumlah_Wisatawan'].values[-lookback:].reshape(1, -1, 1)
last_12_months_scaled = scaler.transform(last_12_months.reshape(-1, 1)).reshape(1, lookback, 1)
monthly_predictions = predict_next_n_months(model, last_12_months_scaled, target_months)

# Generate tanggal prediksi
last_date = airport_data['Tahun-Bulan'].iloc[-1]
pred_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=target_months,
    freq='MS'
)

# Timeline untuk plotting
train_dates = airport_data['Tahun-Bulan'].iloc[12 : split+12]
test_dates = airport_data['Tahun-Bulan'].iloc[split+12 : split+12+len(test_predict)]

# 9. Visualisasi di Streamlit
st.header(f'Prediksi untuk {selected_airport}')

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Data Points", len(airport_data))
with col2:
    st.metric("Data Terakhir", airport_data['Tahun-Bulan'].iloc[-1].strftime('%B %Y'))

fig, ax = plt.subplots(figsize=(12, 6))

# Plot data aktual
ax.plot(airport_data['Tahun-Bulan'], airport_data['Jumlah_Wisatawan'], 
        label='Data Aktual', color='#1f77b4', linewidth=2)

# Plot prediksi training
ax.plot(train_dates, train_predict, 
        label='Prediksi Training', linestyle='--', color='#ff7f0e')

# Plot prediksi testing
ax.plot(test_dates, test_predict, 
        label='Prediksi Testing', linestyle='--', color='#2ca02c')

# Plot prediksi 6 bulan ke depan
ax.plot(pred_dates, monthly_predictions, 
        label='Prediksi Masa Depan', color='red', marker='o', linestyle=':', linewidth=2)

# Anotasi nilai prediksi
for date, pred in zip(pred_dates, monthly_predictions):
    ax.annotate(f"{int(pred):,}", (date, pred), 
                textcoords="offset points", xytext=(0,10), ha='center')

# Format plot
ax.set_title(f'Prediksi Kedatangan Wisatawan di {selected_airport}')
ax.set_xlabel('Tahun-Bulan')
ax.set_ylabel('Jumlah Wisatawan')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Tampilkan metrik evaluasi
st.subheader('Evaluasi Model')
col1, col2 = st.columns(2)
col1.metric("MAE (Training)", f"{train_mae:,.2f}")
col2.metric("MAPE (Training)", f"{train_mape:.2f}%")

col1, col2 = st.columns(2)
col1.metric("MAE (Testing)", f"{test_mae:,.2f}")
col2.metric("MAPE (Testing)", f"{test_mape:.2f}%")

# Tampilkan prediksi 6 bulan ke depan
st.subheader('Prediksi 6 Bulan Ke Depan')
months = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni']
pred_df = pd.DataFrame({
    'Bulan': months,
    'Tahun': pred_dates.year,
    'Prediksi': monthly_predictions
})

st.dataframe(pred_df.style.format({'Prediksi': '{:,.0f}'}))

# Download hasil prediksi
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Prediksi sebagai CSV",
    data=csv,
    file_name=f'prediksi_{selected_airport.replace(" ", "_")}.csv',
    mime='text/csv'
)

# Tampilkan data historis
if st.checkbox("Tampilkan Data Historis"):
    st.subheader(f'Data Historis {selected_airport}')
    st.dataframe(airport_data.sort_values('Tahun-Bulan', ascending=False))
