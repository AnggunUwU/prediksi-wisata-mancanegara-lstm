import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Konfigurasi Aplikasi
st.set_page_config(page_title="📅 Prediksi Wisatawan - LSTM", layout="wide")
st.title('📅 Prediksi Jumlah Wisatawan per Pintu Masuk')

# Tambahkan penjelasan tentang parameter
with st.expander("ℹ️ Panduan Penggunaan"):
    st.markdown("""
    ### 🎛️ Panduan Parameter:
    
    *Jumlah Bulan Lookback:*
    - Pilihan: 1- 12 bulan
    - Default: 12 bulan (optimal untuk pola tahunan)
    
    *Jumlah Epoch:*
    - Range: 50-300
    - Default: 100
    - Lebih tinggi = lebih akurat tapi lebih lama kemungkinan bisa overfitting
    
    *Bulan Prediksi:*
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
df_filtered = df[df['Pintu Masuk'] == selected_pintu].sort_values('Tahun-Bulan')

if len(df_filtered) < 24:
    st.error(f"⚠️ Data historis untuk {selected_pintu} hanya {len(df_filtered)} bulan, minimal 24 bulan diperlukan")
    st.stop()
    
# Tampilkan data
with st.expander(f"🔍 Lihat Data Historis {selected_pintu}"):
    st.dataframe(df_filtered, height=200)
# ======================================
# 2. Panel Kontrol
# ======================================
st.subheader("⚙️ Parameter Model")
col1, col2, col3 = st.columns(3)

with col1:
    time_steps = st.selectbox(
        "Jumlah Bulan Lookback",
        options=[*range(1, 13)],  # Hanya 1-12
        index=11  # Default ke 12 (indeks terakhir)
    )
    
with col2:
    epochs = st.slider("Jumlah Epoch", min_value=50, max_value=300, value=100)

with col3:
    future_months = st.number_input("Prediksi Berapa Bulan ke Depan?", min_value=1, max_value=36, value=12)

if st.button("🚀 Jalankan Model", type="primary", use_container_width=True):
    # ======================================
    # 3. Preprocessing Data
    # ======================================
    with st.spinner('🔨 Mempersiapkan data...'):
        data = df_filtered[['Jumlah_Wisatawan']].values.astype('float32')
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        def create_dataset(data, steps):
            X, y = [], []
            for i in range(len(data)-steps):
                X.append(data[i:(i+steps), 0])
                y.append(data[i+steps, 0])
            return np.array(X), np.array(y)

        X, y = create_dataset(data_scaled, time_steps)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

    # ======================================
    # 4. Training Model
    # ======================================
    progress_bar = st.progress(0)
    status_text = st.empty()

    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(time_steps, 1), return_sequences=True),
        LSTM(32, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer='Adam', loss='mse')

    for epoch in range(epochs):
        history = model.fit(
            X_train, y_train,
            epochs=1,
            validation_data=(X_test, y_test),
            verbose=0
        )
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"⏳🤖 Training model: Epoch {epoch+1}/{epochs} selesai")

    progress_bar.empty()
    status_text.empty()

    # ======================================
    # 5. Evaluasi Model
    # ======================================
    def calculate_metrics(actual, predicted):
        actual = actual.flatten()
        predicted = predicted.flatten()
        mask = actual != 0
        mae = mean_absolute_error(actual[mask], predicted[mask])
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mae, mape

    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_mae, train_mape = calculate_metrics(y_train_actual, train_pred)
    test_mae, test_mape = calculate_metrics(y_test_actual, test_pred)

    st.subheader("📊 Evaluasi Model")
    col1, col2 = st.columns(2)
    col1.metric("Test MAE", f"{test_mae:,.0f}")
    col2.metric("Test MAPE", f"{test_mape:.1f}%")

    # ======================================
    # 6. Visualisasi Interaktif dengan Plotly 
    #    (TAMPILAN LOADING TETAP SAMA)
    # ======================================
    st.subheader("📈 Hasil Prediksi Interaktif")
    tab1, tab2 = st.tabs(["📊 Training vs Test", "🔮 Prediksi Masa Depan"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[time_steps:split+time_steps],
            y=y_train_actual.flatten(),
            name='Train Aktual',
            line=dict(color='blue', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[split+time_steps:],
            y=y_test_actual.flatten(),
            name='Test Aktual',
            line=dict(color='green', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[time_steps:split+time_steps],
            y=train_pred.flatten(),
            name='Prediksi Train',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[split+time_steps:],
            y=test_pred.flatten(),
            name='Prediksi Test',
            line=dict(color='orange', width=2, dash='dash')
        ))
        fig1.update_layout(
            title=f'Perbandingan Data Aktual vs Prediksi - {selected_pintu}',
            xaxis_title='Tanggal',
            yaxis_title='Jumlah Wisatawan',
            hovermode='x unified',
            height=600
        )
        st.plotly_chart(fig1, use_container_width=True)

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

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'],
            y=df_filtered['Jumlah_Wisatawan'],
            name='Data Historis',
            line=dict(color='blue', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions.flatten(),
            name='Prediksi',
            line=dict(color='red', width=2),
            mode='lines+markers'
        ))
        fig2.update_layout(
            title=f'Prediksi {future_months} Bulan ke Depan - {selected_pintu}',
            xaxis_title='Tanggal',
            yaxis_title='Jumlah Wisatawan',
            hovermode='x unified',
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

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
