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