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

<<<<<<< HEAD

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
=======
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
>>>>>>> 7d55fbbb5d59dea85afa388a43a250bea9a7f17a

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

    # ===================================================
    # BAGIAN DEVINA UNTUK MENGEMBANGKAN DASHBOARD
    # ===================================================

    st.subheader("Bagian Dashboard Devina")

    # ------------IMPORT DATA----------------------------
    st.subheader("import data")
    upload_file = st.file_uploader("Unggah file data dalam format CSV/XLSX", type=["csv","xlsx"])
    
    if upload_file is not None:
        try:
            if upload_file.name.endswith(".csv"):
                df = pd.read_csv(upload_file)
            else:
                df = pd.read_excel(upload_file)    
            
            st.success("File berhasil diunggah")
            st.dataframe(df, use_container_width=True)
        
        except Exception as e:
            st.info("Silahkan unggah file untuk memulai proses")
            df = None
    
    #-------------Tabel Output-------------------------------
    st.subheader("Hasil Pengolahan data akan ditampikan disini:")
    st.empty()
    st.title("CSV VIEW_SHE")
<<<<<<< HEAD

    file_name = "Data_Dummy.csv"
=======
    file_name = "data_klasifikasi.csv"
>>>>>>> 7d55fbbb5d59dea85afa388a43a250bea9a7f17a

    try:
        df = pd.read_csv(file_name)
        st.subheader("Data CSV:")
        st.dataframe(df)
    except FileNotFoundError:
        st.error(f"File '{file_name}' tidak ditemukan!")

    # --- FILTER / SEARCH ---
    search_term = st.text_input("Cari data aman (ketik kata kunci):")

    if search_term:
        filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
        
        st.subheader(f"Hasil pencarian: '{search_term}'")
        if filtered_df.empty:
            st.warning("Tidak ada data yang cocok dengan kata kunci.")
        else:
            st.dataframe(filtered_df)

    #------------------INSIGHT SUMMARY-------------------------
    st.subheader("Data Insight")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(label="Dominasi Wilayah",
                value="-",
                delta= "Belum ada data")
    
    col2.metric(label="Kategori Banyak Dikunjungi",
                value="-",
                delta= "Belum ada data")
    
    col3.metric(label="Total Nominal Transaksi",
                value="-",
                delta= "Belum ada data")
    
    col4.metric(label="Total Produk Terjual",
                value="-",
                delta= "Belum ada data")

    col5.metric(label="Masuk Keranjang",
                value="-",
                delta= "Belum ada data")
    
#--------------------VISUALISASI TREN---------------------
    st.subheader("Data Tren")
    with st.container():
        st.info("Visualisasi Tren akan ditampilkan disini")
        st.empty()


# PAGE 3 – EXCEL PROCESSING (BAGIAN SHEIRA)
# =====================================================
def page_excel():
    st.title("Excel Processing Page")

    file_name = "Data_Dummy.csv"

    # ================= LOAD DATA =================
    try:
        df = pd.read_csv(file_name)
    except:
        st.error("Dataset tidak ditemukan")
        st.stop()

    st.divider()
    st.subheader("Filter Tampilan Data")

    filtered_df = df.copy()

    # ================= PILIH KOLOM =================
    selected_columns = st.multiselect(
        "Pilih kolom yang ingin ditampilkan:",
        options=filtered_df.columns.tolist(),
        default=filtered_df.columns.tolist()
    )

    # ================= KATEGORI PRODUK =================
    category_col = "kategori_produk"
    if category_col in filtered_df.columns:
        categories = sorted(filtered_df[category_col].dropna().astype(str).unique())
        st.write("Pilih kategori produk:")

        # Inisialisasi session_state untuk checkbox individual
        for cat in categories:
            key = f"cat_{cat}"
            if key not in st.session_state:
                st.session_state[key] = False

        # Callback untuk Select ALL
        def toggle_all_cat():
            for cat in categories:
                st.session_state[f"cat_{cat}"] = st.session_state["cat_all"]

        # Checkbox ALL dengan callback
        st.checkbox("ALL Kategori", key="cat_all", on_change=toggle_all_cat)

        # Render checkbox individual
        cols = st.columns(3)
        selected_categories = []
        for i, cat in enumerate(categories):
            key = f"cat_{cat}"
            checked = cols[i % 3].checkbox(cat, value=st.session_state[key], key=key)
            if checked:
                selected_categories.append(cat)

        if selected_categories:
            filtered_df = filtered_df[filtered_df[category_col].astype(str).isin(selected_categories)]
    else:
        st.warning(f"Kolom '{category_col}' tidak ditemukan")

    # ================= WILAYAH =================
    wilayah_col = "wilayah"
    if wilayah_col in filtered_df.columns:
        wilayah_values = sorted(filtered_df[wilayah_col].dropna().astype(str).unique())
        st.write("Pilih wilayah:")

        # Inisialisasi session_state
        for w in wilayah_values:
            key = f"wil_{w}"
            if key not in st.session_state:
                st.session_state[key] = False

        # Callback Select ALL wilayah
        def toggle_all_wil():
            for w in wilayah_values:
                st.session_state[f"wil_{w}"] = st.session_state["wil_all"]

        st.checkbox("ALL Wilayah", key="wil_all", on_change=toggle_all_wil)

        # Render checkbox individual
        cols_w = st.columns(3)
        selected_wilayah = []
        for i, w in enumerate(wilayah_values):
            key = f"wil_{w}"
            checked = cols_w[i % 3].checkbox(w, value=st.session_state[key], key=key)
            if checked:
                selected_wilayah.append(w)

        if selected_wilayah:
            filtered_df = filtered_df[filtered_df[wilayah_col].astype(str).isin(selected_wilayah)]
    else:
        st.warning(f"Kolom '{wilayah_col}' tidak ditemukan")

    # ================= APPLY COLUMN FILTER =================
    if selected_columns:
        filtered_df = filtered_df[selected_columns]

    # ================= OUTPUT =================
    st.subheader("Hasil Filter:")
    st.dataframe(filtered_df)

# =====================================================
# =====================================================
# JALANKAN APLIKASI
# =====================================================
if __name__ == "__main__":
    main()
