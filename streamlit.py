import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler  # Changed from RobustScaler
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
    - Pilihan: 1- 24 bulan
    - Default: 12 bulan (optimal untuk pola tahunan)
    
    **Jumlah Epoch:**
    - Range: 50-300
    - Default: 100
    - Lebih tinggi = lebih akurat tapi lebih lama kemungkinan bisa overfitting
    
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
        options=[*range(1, 11), 12, 18, 24],  # 1-10, lalu 12,18,24
        index=11,  # Default ke 12 bulan
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
# ======================================
# 3. Preprocessing Data
# ======================================
with st.spinner('🔨 Mempersiapkan data...'):
    # First, extract the target values from filtered dataframe
    data = df_filtered[['Jumlah_Wisatawan']].values.astype('float32')  # Make sure to extract the values
    
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
    LSTM(64, activation='relu', input_shape=(time_steps, 1), return_sequences=True),
    LSTM(32, activation='relu'),
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
# 6. Enhanced Interactive Visualization
# ======================================
st.subheader("📈 Hasil Prediksi")

try:
    # Tab 1: Training vs Test
    tab1, tab2 = st.tabs(["📉 Training vs Test", "🔮 Prediksi Masa Depan"])

    with tab1:
        import plotly.graph_objects as go
        
        # Create figure
        fig1 = go.Figure()
        
        # Add traces
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[time_steps:split+time_steps],
            y=y_train_actual.flatten(),
            name='Train Aktual',
            line=dict(color='blue', width=2),
            mode='lines'
        ))
        
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[split+time_steps:],
            y=y_test_actual.flatten(),
            name='Test Aktual',
            line=dict(color='green', width=2),
            mode='lines'
        ))
        
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[time_steps:split+time_steps],
            y=train_pred.flatten(),
            name='Prediksi Train',
            line=dict(color='red', width=2, dash='dash'),
            mode='lines'
        ))
        
        fig1.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'].iloc[split+time_steps:],
            y=test_pred.flatten(),
            name='Prediksi Test',
            line=dict(color='orange', width=2, dash='dash'),
            mode='lines'
        ))
        
        # Update layout
        fig1.update_layout(
            title=f'Perbandingan Data Aktual vs Prediksi - {selected_pintu}',
            xaxis_title='Bulan',
            yaxis_title='Jumlah Wisatawan',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white',
            height=600,
            margin=dict(l=50, r=50, b=50, t=80)
        )
        
        # Add range slider
        fig1.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
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

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)
        pred_dates = pd.date_range(
            start=df_filtered['Tahun-Bulan'].iloc[-1] + pd.DateOffset(months=1),
            periods=future_months,
            freq='MS'
        )

        # Create figure
        fig2 = go.Figure()
        
        # Add historical data
        fig2.add_trace(go.Scatter(
            x=df_filtered['Tahun-Bulan'],
            y=df_filtered['Jumlah_Wisatawan'],
            name='Data Historis',
            line=dict(color='blue', width=2),
            mode='lines'
        ))
        
        # Add predictions
        fig2.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions.flatten(),
            name='Prediksi',
            line=dict(color='red', width=2),
            mode='lines+markers',
            marker=dict(size=8)
        ))
        
        # Add confidence interval (example using simple moving std)
        if future_months > 1:
            rolling_std = np.std(df_filtered['Jumlah_Wisatawan'].rolling(12).std().dropna())
            fig2.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions.flatten() + 1.96 * rolling_std,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig2.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions.flatten() - 1.96 * rolling_std,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='95% Confidence Interval',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
        
        # Update layout
        fig2.update_layout(
            title=f'Prediksi {future_months} Bulan ke Depan - {selected_pintu}',
            xaxis_title='Bulan',
            yaxis_title='Jumlah Wisatawan',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white',
            height=600,
            margin=dict(l=50, r=50, b=50, t=80)
        )
        
        # Add range slider
        fig2.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)

        # Tabel hasil interaktif
        pred_df = pd.DataFrame({
            'Bulan': pred_dates.strftime('%B %Y'),
            'Prediksi': predictions.flatten().astype(int),
            'Perubahan (%)': np.round(
                np.insert(
                    np.diff(predictions.flatten()) / predictions.flatten()[:-1] * 100,
                0, 0
            ), 1)
        })
        
        # Create interactive table with AgGrid
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder
            
            gb = GridOptionsBuilder.from_dataframe(pred_df)
            gb.configure_default_column(
                filterable=True,
                sortable=True,
                resizable=True,
                editable=False
            )
            gb.configure_column('Prediksi', type=['numericColumn','numberColumnFilter','customNumericFormat'], 
                               valueFormatter="value.toLocaleString('en-US')")
            gb.configure_column('Perubahan (%)', type=['numericColumn','numberColumnFilter'], 
                               cellStyle={'color': 'white', 'background-color': '#1f77b4'})
            
            grid_options = gb.build()
            
            AgGrid(
                pred_df,
                gridOptions=grid_options,
                height=min(400, 35*future_months),
                width='100%',
                theme='streamlit',
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True
            )
        except:
            # Fallback to st.dataframe if AgGrid not available
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
