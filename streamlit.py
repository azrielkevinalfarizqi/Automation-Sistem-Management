import streamlit as st

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("assets/logo ladang.png", width=180)

st.title("MAGANG SANTAI PT LADANG")

# Tampilkan nama dengan format yang menarik
st.header("SIPLah Data Flow Automation: Sistem Cerdas Pengelompokan Produk E-Commerce untuk Mengurangi Bias Kategori dan Meningkatkan Akurasi Analisis")

# Tambahkan sedikit styling
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

# Tambahkan garis pembatas
st.divider()

# Tambahkan teks tambahan
st.write("Day-1 nyoba magang")
import pandas as pd

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
    # Filter semua kolom
    filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    
    st.subheader(f"Hasil pencarian: '{search_term}'")
    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan kata kunci.")
    else:
        st.dataframe(filtered_df)

# ===============================
# QUERY SUGGESTION / AUTOCOMPLETE
# ===============================

st.divider()
st.subheader("Query Suggestion (Auto-Completion)")

# Ambil semua nilai unik dari dataframe
try:
    all_values = pd.unique(df.astype(str).values.ravel())
    all_values = sorted([v for v in all_values if v != "nan"])
except:
    st.stop()

# Input query tambahan untuk suggestion
suggest_query = st.text_input("Ketik untuk mendapatkan suggestion:")

# Filter suggestion
if suggest_query:
    matched = [
        v for v in all_values
        if suggest_query.lower() in v.lower()
    ]
else:
    matched = all_values[:20]  # default tampil sebagian saja

# Tampilkan suggestion (batasi agar tidak berat)
selected_value = st.selectbox(
    "Pilih dari suggestion:",
    options=matched[:50]
)

# Filter dataframe berdasarkan pilihan suggestion
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