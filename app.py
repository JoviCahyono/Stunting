import streamlit as st
from streamlit_option_menu import option_menu
from joblib import load
import pandas as pd

# Memuat model yang disimpan
loaded_model = load('model.joblib')

# Memuat objek scaler
scaler = load('scaler.joblib')

# Memetakan label numerik ke kategori deskriptif
status_mapping = {0: 'Sangat Stunting', 1: 'Stunting', 2: 'Normal', 3: 'Tinggi'}

# Inisialisasi history sebagai DataFrame kosong jika belum ada
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=['Usia (bulan)', 'Jenis Kelamin', 'Tinggi (cm)', 'Hasil Klasifikasi'])

# Fungsi untuk mengklasifikasikan status stunting
def prediksi_status_pendek(usia_bulan, jenis_kelamin_numerik, tinggi_cm):
    # Menskalakan fitur input
    data_input_skala = scaler.transform([[usia_bulan, jenis_kelamin_numerik, tinggi_cm]])
    # Melakukan klasifikasi menggunakan model yang dimuat
    prediksi = loaded_model.predict(data_input_skala)
    return prediksi

# Mengatur sidebar dengan ikon
with st.sidebar:
    option = option_menu(
        'Menu Utama',
        ['Beranda', 'Klasifikasi', 'Tentang'],
        icons=['house', 'activity', 'info-circle'],
        menu_icon="cast",
        default_index=0,
    )

if option == 'Beranda':
    st.title('Aplikasi Klasifikasi Stunting')
    st.image('Stunting.jpg', use_column_width=True)  # Ganti dengan path gambar yang sesuai
    st.write("Selamat datang di Aplikasi Klasifikasi Stunting. Silakan pilih 'Klasifikasi' di menu untuk memulai Klasifikasi.")
    
elif option == 'Klasifikasi':
    st.header("Klasifikasi Status Stunting")
    st.write("Silahkan masukkan data berikut untuk mengklasifikasikan status stunting anak:")

    # Input usia dengan ikon
    icon_usia = "👶"
    usia_bulan = st.number_input(f'{icon_usia} Usia (bulan)', min_value=0, max_value=60, value=30)
    
    # Input jenis kelamin dengan ikon
    icon_jenis_kelamin = "⚧️"
    jenis_kelamin = st.radio(f'{icon_jenis_kelamin} Jenis Kelamin', ['Laki-laki', 'Perempuan'])
    jenis_kelamin_numerik = 0 if jenis_kelamin == 'Laki-laki' else 1
    
    # Input tinggi badan dengan ikon
    icon_tinggi = "📏"
    tinggi_cm = st.number_input(f'{icon_tinggi} Tinggi (cm)', min_value=0.0, max_value=128.0, value=64.0)

    # Tombol Klasifikasi dengan ikon
    if st.button('Klasifikasi'):
        prediksi = prediksi_status_pendek(usia_bulan, jenis_kelamin_numerik, tinggi_cm)
        status_terprediksi = status_mapping[prediksi[0]]

        # Menambahkan entri ke history
        st.session_state.history_df.loc[len(st.session_state.history_df)] = [usia_bulan, jenis_kelamin, tinggi_cm, status_terprediksi]

        # Menampilkan hasil Klasifikasi dengan tampilan
        st.write('**Hasil Klasifikasi:**')
        if prediksi[0] == 0:
            st.error('Status Klasifikasi Stunting Adalah: ' + status_terprediksi)
        elif prediksi[0] == 1:
            st.warning('Status Klasifikasi Stunting Adalah: ' + status_terprediksi)
        elif prediksi[0] == 2:
            st.success('Status Klasifikasi Stunting Adalah: ' + status_terprediksi)
        else:
            st.info('Status Klasifikasi Stunting Adalah: ' + status_terprediksi)

elif option == 'Tentang':
    st.write("""
    **Tentang Aplikasi Klasifikasi Stunting**
    
    Aplikasi Klasifikasi Stunting ini dikembangkan untuk membantu mengklasifikasikan status stunting pada balita berdasarkan data usia, jenis kelamin, dan tinggi badan. Dengan menggunakan algoritma K-Nearest Neighbors (KNN), aplikasi ini mampu memberikan klasifikasi yang akurat mengenai status gizi anak, yang dikelompokkan ke dalam empat kategori: Sangat Stunting, Stunting, Normal, dan Tinggi.

    **Fitur Aplikasi:**
    - **Klasifikasi Stunting:** Input usia, jenis kelamin, dan tinggi badan untuk mendapatkan klasifikasi status stunting.
    - **Antarmuka yang Mudah Digunakan:** Desain antarmuka yang intuitif dan user-friendly.

    **Mengapa Stunting Penting?**
    Stunting adalah kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis dalam jangka waktu yang lama. Stunting berhubungan dengan perkembangan otak yang terganggu, kapasitas belajar yang rendah, dan risiko penyakit kronis di masa dewasa. Oleh karena itu, pencegahan dan penanganan stunting sejak dini sangat penting untuk memastikan masa depan yang lebih baik bagi anak-anak.

    Terima kasih telah menggunakan Aplikasi Klasifikasi Stunting. Bersama-sama, kita bisa mengatasi masalah stunting dan memastikan tumbuh kembang yang optimal bagi anak-anak kita.
    """)
    
# Menampilkan history jika tidak kosong
if not st.session_state.history_df.empty:
    st.header('History Klasifikasi')
    st.write(st.session_state.history_df)
