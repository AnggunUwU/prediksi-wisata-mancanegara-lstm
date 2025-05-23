# ======================================
# 6. Visualisasi Hasil - Perbaikan Timeline
# ======================================
with tab1:
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Dapatkan tanggal yang sesuai untuk plotting
    train_dates = df_filtered['Tahun-Bulan'].iloc[time_steps:time_steps+len(X_train)]
    test_dates = df_filtered['Tahun-Bulan'].iloc[time_steps+len(X_train):time_steps+len(X_train)+len(X_test)]
    
    ax1.plot(train_dates, y_train_actual, label='Train Aktual', color='blue')
    ax1.plot(test_dates, y_test_actual, label='Test Aktual', color='green')
    ax1.plot(train_dates, train_pred, label='Prediksi Train', linestyle='--', color='red')
    ax1.plot(test_dates, test_pred, label='Prediksi Test', linestyle='--', color='orange')
    
    # Tambahkan garis vertikal pemisah
    split_line_date = test_dates.iloc[0] if len(test_dates) > 0 else train_dates.iloc[-1]
    ax1.axvline(x=split_line_date, color='gray', linestyle=':', label='Pemisah Train-Test')
    
    ax1.set_title(f'Perbandingan Data Aktual vs Prediksi - {selected_pintu}')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig1)
