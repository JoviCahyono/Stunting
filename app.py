# Aplikasi Streamlit
import streamlit as st
from joblib import load

# Memuat model yang disimpan
loaded_model = load('model.joblib')

# Memuat objek scaler
scaler = load('scaler.joblib')

# Mendefinisikan kamus untuk memetakan label numerik ke kategori deskriptif
status_mapping = {0: 'Sangat Stunting', 1: 'Stunting', 2: 'Normal', 3: 'Tinggi'}

# Fungsi untuk memprediksi status stunting
def prediksi_status_pendek(usia_bulan, jenis_kelamin_numerik, tinggi_cm):
    # Menskalakan fitur input
    data_input_skala = scaler.transform([[usia_bulan, jenis_kelamin_numerik, tinggi_cm]])
    # Melakukan prediksi menggunakan model yang dimuat
    prediksi = loaded_model.predict(data_input_skala)
    return prediksi

# Antarmuka Streamlit
st.title('Aplikasi Prediksi Stunting')
st.write("Aplikasi ini bertujuan untuk memprediksi status stunting berdasarkan usia, jenis kelamin, dan tinggi badan.")

# Input usia, jenis kelamin, dan tinggi badan
usia_bulan = st.number_input('Usia (bulan)', min_value=0, max_value=60, value=30)
jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
jenis_kelamin_numerik = 0 if jenis_kelamin == 'Laki-laki' else 1
tinggi_cm = st.number_input('Tinggi (cm)', min_value=0.0, max_value=128.0, value=64.0)

# Tombol Prediksi
if st.button('Prediksi'):
    prediksi = prediksi_status_pendek(usia_bulan, jenis_kelamin_numerik, tinggi_cm)
    # Memetakan prediksi numerik ke kategori deskriptif
    status_terprediksi = status_mapping[prediksi[0]]
    # Menampilkan hasil prediksi dengan tampilan yang lebih bagus
    st.write('**Hasil Prediksi:**')
    if prediksi[0] == 0:
        st.error('Status Stunting yang Diprediksi : ' + status_terprediksi)
    elif prediksi[0] == 1:
        st.warning('Status Stunting yang Diprediksi : ' + status_terprediksi)
    elif prediksi[0] == 2:
        st.success('Status Stunting yang Diprediksi : ' + status_terprediksi)
    else:
        st.info('Status Stunting yang Diprediksi : ' + status_terprediksi)
