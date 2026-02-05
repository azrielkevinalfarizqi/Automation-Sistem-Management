import streamlit as st
import pandas as pd

# =====================================================
# KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# =====================================================
st.set_page_config(page_title="SIPLah Automation System", layout="wide")


# =====================================================
# FUNGSI NAVIGASI HALAMAN
# =====================================================
def main():

    # Sidebar Navigasi
    st.sidebar.title("Navigasi Sistem")

    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["Login", "Dashboard", "Excel Processing"]
    )

    if page == "Login":
        page_login()

    elif page == "Dashboard":
        page_dashboard()

    elif page == "Excel Processing":
        page_excel()



# =====================================================
# PAGE 1 – LOGIN (BAGIAN KEVIN)
# =====================================================
def page_login():

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("assets/logo ladang.png", width=180)

    st.title("MAGANG SANTAI PT LADANG")

    st.header("SIPLah Data Flow Automation: Sistem Cerdas Pengelompokan Produk E-Commerce untuk Mengurangi Bias Kategori dan Meningkatkan Akurasi Analisis")

    st.divider()

    st.subheader("Halaman Login")

    # =============================
    # BAGIAN INI UNTUK KEVIN EDIT
    # =============================

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.success("Login berhasil (contoh tampilan sementara)")

    # --------------------------------------------------
    # SILAKAN TAMBAHKAN LOGIKA LOGIN DI BAWAH INI
    # - Validasi akun
    # - Session state
    # - Redirect halaman
    # --------------------------------------------------




# =====================================================
# PAGE 2 – DASHBOARD (BAGIAN DEVINA)
# =====================================================
def page_dashboard():

    st.title("Dashboard Utama")

    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
            .header-style {
                color: #1E88E5;
                text-align: center;
                padding: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.divider()

    st.write("Day-1 nyoba magang")

    st.title("CSV VIEW_SHE")

    file_name = "data_klasifikasi.csv"

    try:
        df = pd.read_csv(file_name)
        st.subheader("Data CSV:")
        st.dataframe(df)
    except FileNotFoundError:
        st.error(f"File '{file_name}' tidak ditemukan!")

    # --- FILTER / SEARCH ---
    search_term = st.text_input("Cari data amana (ketik kata kunci):")

    if search_term:
        filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
        
        st.subheader(f"Hasil pencarian: '{search_term}'")
        if filtered_df.empty:
            st.warning("Tidak ada data yang cocok dengan kata kunci.")
        else:
            st.dataframe(filtered_df)


    # ===================================================
    # BAGIAN DEVINA UNTUK MENGEMBANGKAN DASHBOARD
    # ===================================================

    st.divider()
    st.subheader("Bagian Dashboard Devina")

    # ---------------------------------------------------
    # SILAKAN TAMBAHKAN:
    # - Grafik visualisasi
    # - Insight wilayah
    # - Analisis produk
    # - Trend data
    # ---------------------------------------------------




# =====================================================
# PAGE 3 – EXCEL PROCESSING (BAGIAN SHEIRA)
# =====================================================
def page_excel():

    st.title("Excel Processing Page")

    st.divider()
    st.subheader("Query Suggestion (Auto-Completion)")

    file_name = "data_klasifikasi.csv"

    try:
        df = pd.read_csv(file_name)
    except:
        st.error("Dataset tidak ditemukan")
        st.stop()

    # Ambil semua nilai unik dari dataframe
    try:
        all_values = pd.unique(df.astype(str).values.ravel())
        all_values = sorted([v for v in all_values if v != "nan"])
    except:
        st.stop()

    suggest_query = st.text_input("Ketik untuk mendapatkan suggestion:")

    if suggest_query:
        matched = [
            v for v in all_values
            if suggest_query.lower() in v.lower()
        ]
    else:
        matched = all_values[:20]

    selected_value = st.selectbox(
        "Pilih dari suggestion:",
        options=matched[:50]
    )

    if selected_value:
        suggestion_df = df[
            df.apply(
                lambda row: row.astype(str)
                .str.contains(selected_value, case=False)
                .any(),
                axis=1
            )
        ]

        st.subheader("Hasil berdasarkan suggestion:")
        st.dataframe(suggestion_df)


    # ===================================================
    # BAGIAN SHEIRA UNTUK DIKEMBANGKAN
    # ===================================================

    st.divider()
    st.subheader("Bagian Excel Processing Sheira")

    # ---------------------------------------------------
    # SILAKAN TAMBAHKAN:
    # - Upload file excel/csv
    # - Cleaning data
    # - Download hasil
    # - Normalisasi kolom
    # ---------------------------------------------------




# =====================================================
# JALANKAN APLIKASI
# =====================================================
if __name__ == "__main__":
    main()
