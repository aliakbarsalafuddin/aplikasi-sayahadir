import streamlit as st
import pandas as pd
import face_recognition
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
from geopy.distance import geodesic
from streamlit_geolocation import streamlit_geolocation

# --- KONFIGURASI ---
# Ganti dengan koordinat lokasi kantor/area yang diizinkan
LOKASI_KANTOR = (-7.257472, 112.752088)  # Contoh: Monas, Jakarta
RADIUS_MAKSIMAL_METER = 100  # Radius maksimal dalam meter dari lokasi kantor
PATH_DATABASE_WAJAH = "database_wajah"
FILE_LOG_ABSENSI = "absensi.csv"

# --- FUNGSI-FUNGSI BANTUAN ---

# Fungsi untuk menghitung jarak antara dua koordinat
def hitung_jarak(koordinat_user, koordinat_kantor):
    """Menghitung jarak dalam meter."""
    if koordinat_user:
        return geodesic(koordinat_user, koordinat_kantor).meters
    return float('inf')

# Menggunakan cache untuk mempercepat loading, karena kita tidak perlu load gambar setiap saat
@st.cache_resource
def muat_wajah_dikenal():
    """Memuat encoding wajah dari folder database_wajah."""
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(PATH_DATABASE_WAJAH):
        st.error(f"Direktori database wajah tidak ditemukan di: {PATH_DATABASE_WAJAH}")
        return [], []

    for image_name in os.listdir(PATH_DATABASE_WAJAH):
        image_path = os.path.join(PATH_DATABASE_WAJAH, image_name)
        try:
            image = face_recognition.load_image_file(image_path)
            # Ambil encoding wajah pertama yang ditemukan
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                face_encoding = face_encodings[0]
                # Nama file tanpa ekstensi menjadi nama orang
                person_name = os.path.splitext(image_name)[0].replace("_", " ")
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
        except Exception as e:
            st.warning(f"Gagal memproses gambar {image_name}: {e}")
            
    return known_face_encodings, known_face_names

def catat_absensi(nama):
    """Mencatat nama, tanggal, dan waktu ke file CSV."""
    if not os.path.exists(FILE_LOG_ABSENSI):
        # Buat file baru dengan header jika belum ada
        df = pd.DataFrame(columns=["Nama", "Tanggal", "Waktu"])
        df.to_csv(FILE_LOG_ABSENSI, index=False)

    # Baca data yang ada
    df = pd.read_csv(FILE_LOG_ABSENSI)
    
    # Dapatkan waktu saat ini
    now = datetime.now()
    tanggal = now.strftime("%Y-%m-%d")
    waktu = now.strftime("%H:%M:%S")

    # Cek apakah orang ini sudah absen hari ini
    sudah_absen_hari_ini = ((df['Nama'] == nama) & (df['Tanggal'] == tanggal)).any()

    if not sudah_absen_hari_ini:
        # Tambahkan catatan baru
        new_entry = pd.DataFrame([[nama, tanggal, waktu]], columns=["Nama", "Tanggal", "Waktu"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(FILE_LOG_ABSENSI, index=False)
        st.success(f"Absensi untuk {nama} pada {tanggal} pukul {waktu} berhasil dicatat.")
    else:
        st.warning(f"{nama} sudah melakukan absensi hari ini.")


# --- APLIKASI UTAMA STREAMLIT ---

st.set_page_config(page_title="Sistem Absensi Wajah", layout="wide")
st.title("Sistem Absensi Karyawan Berbasis Wajah & Lokasi")
st.write("Aplikasi ini akan memverifikasi lokasi Anda dan mengenali wajah Anda untuk mencatat kehadiran.")

# Muat data wajah yang sudah dikenal
known_face_encodings, known_face_names = muat_wajah_dikenal()

if not known_face_names:
    st.error("Tidak ada data wajah yang dimuat. Pastikan folder 'database_wajah' berisi gambar.")
else:
    # Langkah 1: Verifikasi Lokasi (Geofencing)
    st.header("1. Verifikasi Lokasi Anda")
    # Dapatkan lokasi pengguna dari browser
    location = streamlit_geolocation()

    if location:
        user_coords = (location['latitude'], location['longitude'])
        jarak = hitung_jarak(user_coords, LOKASI_KANTOR)
        
        st.write(f"Lokasi Anda: ({user_coords[0]:.6f}, {user_coords[1]:.6f})")
        st.write(f"Jarak Anda dari kantor: **{jarak:.2f} meter**")

        if jarak <= RADIUS_MAKSIMAL_METER:
            st.success("✅ Anda berada dalam jangkauan yang diizinkan untuk absensi.")
            
            # Langkah 2: Ambil Gambar & Pengenalan Wajah
            st.header("2. Ambil Foto Wajah Anda")
            img_file_buffer = st.camera_input("Silakan posisikan wajah Anda di depan kamera dan klik 'Take photo'")

            if img_file_buffer is not None:
                # Ubah buffer gambar menjadi format yang bisa dibaca OpenCV/face_recognition
                bytes_data = img_file_buffer.getvalue()
                image = Image.open(img_file_buffer).convert("RGB")
                frame = np.array(image)

                # Temukan semua wajah dan encodingnya di frame saat ini
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                nama_dikenali = "Tidak Dikenali"

                if not face_encodings:
                    st.warning("⚠️ Tidak ada wajah yang terdeteksi. Mohon coba lagi.")
                else:
                    # Ambil encoding wajah pertama yang terdeteksi
                    captured_face_encoding = face_encodings[0]

                    # Bandingkan dengan wajah yang sudah dikenal
                    matches = face_recognition.compare_faces(known_face_encodings, captured_face_encoding)
                    
                    # Cari kecocokan terbaik
                    face_distances = face_recognition.face_distance(known_face_encodings, captured_face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        nama_dikenali = known_face_names[best_match_index]

                    # Tampilkan hasil dan catat absensi
                    if nama_dikenali != "Tidak Dikenali":
                        st.success(f"🎉 Wajah dikenali sebagai: **{nama_dikenali}**")
                        # Catat absensi jika wajah dikenali
                        catat_absensi(nama_dikenali)
                    else:
                        st.error("❌ Wajah tidak dikenali. Pastikan wajah Anda terdaftar dan pencahayaan cukup.")
        else:
            st.error("❌ Anda berada di luar jangkauan yang diizinkan untuk absensi.")
            st.info("Harap mendekat ke lokasi kantor yang telah ditentukan.")

    else:
        st.warning("Mohon izinkan akses lokasi di browser Anda untuk melanjutkan.")


# Menampilkan log absensi terakhir
st.header("Log Absensi Hari Ini")
if os.path.exists(FILE_LOG_ABSENSI):
    df_log = pd.read_csv(FILE_LOG_ABSENSI)
    # Filter log untuk hari ini saja
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_hari_ini = df_log[df_log["Tanggal"] == today_date]
    st.dataframe(log_hari_ini, use_container_width=True)
else:
    st.info("Belum ada data absensi yang tercatat.")