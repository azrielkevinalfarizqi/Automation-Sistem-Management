import streamlit as st
import pandas as pd

st.set_page_config(page_title="Auto-in", layout="wide")


# =================Login Page=================
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
                st.rerun()
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



# =================Dashboard=================
def page_dashboard():
    st.write("")   # spacer
    st.write("")
    st.write("")   # spacer
    st.write("") 

# =================Upload Dataset=================#sheira hikangin nama dan nambahin css
    st.markdown("""
<style>

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.12);
}

[data-testid="stFileUploader"] section {
    border: 2px dashed #42A5F5;
    background: #FAFAFA;
}

</style>
""", unsafe_allow_html=True)
    upload_file = st.file_uploader(
    "",
    type=["csv", "xlsx"]
)
    df = None

    if upload_file is not None:
        try:
            if upload_file.name.endswith(".csv"):
                df = pd.read_csv(upload_file)
            else:
                df = pd.read_excel(upload_file)

            st.success("File berhasil diunggah")
            # st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error("Terjadi kesalahan saat membaca file")
            df = None

#she nambahin ini untuk tulisan agar di dalam kotak

 # =================dashboard=================#she nambahin ini
    st.markdown(
    """
    <h3 style='
        font-family: Poppins SemiBold;
        color: #165E76;
        font-size: 24px;
        font-weight: 600;
        margin-top: 30px;
    '>
    Business Insight
    </h3>
    """,
    unsafe_allow_html=True
)
    col1, col2, col3, col4, col5 = st.columns(5)

    if df is not None:

        df.columns = [c.lower() for c in df.columns]

        # ---- Dominasi Wilayah ----
        if "wilayah" in df.columns:
            top_wilayah = df["wilayah"].value_counts().idxmax()
            top_wilayah_count = df["wilayah"].value_counts().max()
            col1.metric("Dominasi Wilayah", top_wilayah, f"{top_wilayah_count} transaksi")
        else:
            col1.metric("Dominasi Wilayah", "-", "Kolom tidak ada")

        # ---- Kategori Terpopuler ----
        if "kategori_produk" in df.columns:
            top_kategori = df["kategori_produk"].value_counts().idxmax()
            top_kategori_count = df["kategori_produk"].value_counts().max()
            col2.metric("Kategori Banyak Dikunjungi", top_kategori, f"{top_kategori_count} data")
        else:
            col2.metric("Kategori Banyak Dikunjungi", "-", "Kolom tidak ada")

        # ---- Total Nominal ----
        if "nominal" in df.columns:
            total_nominal = df["nominal"].sum()
            col3.metric("Total Nominal Transaksi", f"Rp {total_nominal:,.0f}")
        else:
            col3.metric("Total Nominal Transaksi", "-", "Kolom tidak ada")

        # ---- Total Produk Terjual ----
        if "qty" in df.columns:
            total_qty = df["qty"].sum()
            col4.metric("Total Produk Terjual", total_qty)
        else:
            col4.metric("Total Produk Terjual", "-", "Kolom tidak ada")

        # ---- Jumlah Baris Data ----
        col5.metric("Total Data Masuk", len(df))


    else:
        col1.metric("Dominasi Wilayah", "-", "Belum ada data")
        col2.metric("Kategori Banyak Dikunjungi", "-", "Belum ada data")
        col3.metric("Total Nominal Transaksi", "-", "Belum ada data")
        col4.metric("Total Produk Terjual", "-", "Belum ada data")
        col5.metric("Total Data Masuk", "-", "Belum ada data")

# =================Visualisasi Simpel=================
    st.subheader("Data Tren")

    if df is not None:

        colA, colB = st.columns(2)

        # -------- TREND BERDASARKAN TANGGAL --------
        with colA:
            st.write("Tren Jumlah Transaksi per Tanggal")

            if "tanggal" in df.columns:
                try:
                    df["tanggal"] = pd.to_datetime(df["tanggal"])
                    tren = df.groupby("tanggal").size().reset_index(name="jumlah_transaksi")

                    st.line_chart(tren.set_index("tanggal"))

                except:
                    st.warning("Format kolom tanggal tidak sesuai")
            else:
                st.info("Tidak ada kolom 'tanggal' untuk visualisasi tren waktu")


        # -------- BAR CHART KATEGORI --------
        with colB:
            st.write("Distribusi Kategori Produk")

            if "kategori_produk" in df.columns:

                kategori_chart = df["kategori_produk"].value_counts()

                st.bar_chart(kategori_chart)

            else:
                st.info("Tidak ada kolom 'kategori_produk' untuk visualisasi")


    else:
        st.info("Silahkan upload dataset untuk melihat visualisasi")

# =================Filter Data=================

    if df is not None:

        st.divider()
        st.subheader("Filter Tampilan Data")

        filtered_df = df.copy()

        # ================= PILIH KOLOM =================
        selected_columns = st.multiselect(
            "Pilih kolom yang ingin ditampilkan:",
            options=filtered_df.columns.tolist(),
            default=filtered_df.columns.tolist()
        )

        # ================= FILTER KATEGORI =================
        category_col = "kategori_produk"

        if category_col in filtered_df.columns:

            categories = sorted(filtered_df[category_col].dropna().astype(str).unique())

            st.write("Pilih kategori produk:")

            for cat in categories:
                key = f"cat_{cat}"
                if key not in st.session_state:
                    st.session_state[key] = False

            def toggle_all_cat():
                for cat in categories:
                    st.session_state[f"cat_{cat}"] = st.session_state["cat_all"]

            st.checkbox("ALL Kategori", key="cat_all", on_change=toggle_all_cat)

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
            st.warning("Kolom kategori_produk tidak ditemukan")


        # ================= FILTER WILAYAH =================
        wilayah_col = "wilayah"

        if wilayah_col in filtered_df.columns:

            wilayah_values = sorted(filtered_df[wilayah_col].dropna().astype(str).unique())

            st.write("Pilih wilayah:")

            for w in wilayah_values:
                key = f"wil_{w}"
                if key not in st.session_state:
                    st.session_state[key] = False

            def toggle_all_wil():
                for w in wilayah_values:
                    st.session_state[f"wil_{w}"] = st.session_state["wil_all"]

            st.checkbox("ALL Wilayah", key="wil_all", on_change=toggle_all_wil)

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
            st.warning("Kolom wilayah tidak ditemukan")


        # ================= PAKAI FILTER =================
        if selected_columns:
            filtered_df = filtered_df[selected_columns]

        # ================= OUTPUT =================
        st.subheader("Hasil Filter:")
        st.dataframe(filtered_df)
def set_bg(image_path):
    import base64

    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    css = f"""
    <style>

    /* Background utama */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* HEADER STREAMLIT */
    header {{
        background-color: rgba(0,0,0,0) !important;
    }}

    .stApp > header {{
        background-color: transparent !important;
    }}

    .block-container {{
        padding-top: 2rem;
    }}


    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def main():
    set_bg("assets/background.png") # she nambahin ini
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        page_login()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
