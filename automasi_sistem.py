import streamlit as st
import pandas as pd

# =====================================================
# KONFIGURASI HALAMAN
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

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False


    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #1E88E5;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


    logo_col = st.columns([1, 1, 1])

    with logo_col[1]:
        st.image("assets/logo ladang.png", use_container_width=True)


    if st.session_state.logged_in:

        st.success("LOGIN BERHASIL!")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

        st.markdown(
            "<p style='text-align: center;'>Silakan pilih halaman melalui menu navigasi di sidebar.</p>",
            unsafe_allow_html=True
        )

        return

    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "ladang123"


    col_left, col_form, col_right = st.columns([1, 1.2, 1])

    with col_form:

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login_btn = st.button("Login", use_container_width=True)

        if login_btn:

            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.success("Login berhasil! Selamat datang di sistem.")
                st.experimental_rerun()

            else:
                st.error("Username atau Password salah!")


        st.markdown("""
            <div style="
                background-color: #FFF3CD;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #FFEEBA;
                text-align: center;
                margin-top: 10px;
                ">
                ⚠️ <b>Hanya admin yang bisa mengakses sistem ini</b>
            </div>
        """, unsafe_allow_html=True)



# ===================================SAMPE SINI=====================================================

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
