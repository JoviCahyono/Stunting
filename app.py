import streamlit as st
from streamlit_option_menu import option_menu
from joblib import load
import base64

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

# Mengatur sidebar dengan ikon
with st.sidebar:
    option = option_menu(
        'Menu Utama',
        ['Beranda', 'Prediksi', 'Tentang'],
        icons=['house', 'activity', 'info-circle'],
        menu_icon="cast",
        default_index=0,
    )

if option == 'Beranda':
    st.title('Aplikasi Prediksi Stunting')
    st.image('Stunting.jpg', use_column_width=True)  # Ganti dengan path gambar yang sesuai
    st.write("Selamat datang di Aplikasi Prediksi Stunting. Silakan pilih 'Prediksi' di menu untuk memulai prediksi.")
    
elif option == 'Prediksi':
    st.header("Prediksi Status Stunting")
    st.write("Silahkan masukkan data berikut untuk memprediksi status stunting anak:")

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

    # Tombol Prediksi dengan ikon
    if st.button('Prediksi'):
        prediksi = prediksi_status_pendek(usia_bulan, jenis_kelamin_numerik, tinggi_cm)
        status_terprediksi = status_mapping[prediksi[0]]
        
        # Menampilkan hasil prediksi dengan tampilan yang lebih bagus
        st.write('**Hasil Prediksi:**')
        if prediksi[0] == 0:
            st.error('Status Stunting yang Diprediksi: ' + status_terprediksi)
        elif prediksi[0] == 1:
            st.warning('Status Stunting yang Diprediksi: ' + status_terprediksi)
        elif prediksi[0] == 2:
            st.success('Status Stunting yang Diprediksi: ' + status_terprediksi)
        else:
            st.info('Status Stunting yang Diprediksi: ' + status_terprediksi)

elif option == 'Tentang':
    st.write("""
    **Tentang Aplikasi Prediksi Stunting**
    
    Aplikasi Prediksi Stunting ini dikembangkan untuk membantu memprediksi status stunting pada balita berdasarkan data usia, jenis kelamin, dan tinggi badan. Dengan menggunakan algoritma K-Nearest Neighbors (KNN), aplikasi ini mampu memberikan prediksi yang akurat mengenai status gizi anak, yang dikelompokkan ke dalam empat kategori: Sangat Stunting, Stunting, Normal, dan Tinggi.

    **Fitur Aplikasi:**
    - **Prediksi Stunting:** Input usia, jenis kelamin, dan tinggi badan untuk mendapatkan prediksi status stunting.
    - **Antarmuka yang Mudah Digunakan:** Desain antarmuka yang intuitif dan user-friendly.

    **Mengapa Stunting Penting?**
    Stunting adalah kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis dalam jangka waktu yang lama. Stunting berhubungan dengan perkembangan otak yang terganggu, kapasitas belajar yang rendah, dan risiko penyakit kronis di masa dewasa. Oleh karena itu, pencegahan dan penanganan stunting sejak dini sangat penting untuk memastikan masa depan yang lebih baik bagi anak-anak.

    Terima kasih telah menggunakan Aplikasi Prediksi Stunting. Bersama-sama, kita bisa mengatasi masalah stunting dan memastikan tumbuh kembang yang optimal bagi anak-anak kita.
    """)
