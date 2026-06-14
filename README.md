# **Sistem Dashboard Kategorisasi Produk SIPLAH**

## **Deskripsi Proyek**

Proyek ini merupakan pengembangan sebuah sistem klasifikasi kategori produk berbasis Natural Language Processing (NLP) yang dilengkapi dengan dashboard analitik sederhana untuk membantu analisis data produk pada platform e-commerce SIPLAH.

Permasalahan utama yang melatarbelakangi pengembangan sistem ini adalah ketidaksesuaian antara nama produk yang diinput oleh pengguna dengan kategori produk yang seharusnya. Ketidaksesuaian ini dapat menyebabkan berbagai kendala dalam proses analisis data, monitoring performa produk, serta pengelolaan katalog pada platform e-commerce.

Untuk mengatasi permasalahan tersebut, sistem ini dikembangkan dengan dua tujuan utama:

1. Mengotomatisasi proses kategorisasi produk menggunakan model machine learning berbasis teks.
2. Menyediakan dashboard analitik sederhana untuk membantu pengguna memahami performa produk mereka.

Sistem ini memanfaatkan model bahasa IndoBERT untuk memahami konteks teks produk dalam Bahasa Indonesia dan mengklasifikasikannya ke dalam struktur kategori bertingkat.

Antarmuka aplikasi dibangun menggunakan framework Streamlit, sehingga sistem dapat digunakan secara langsung melalui browser tanpa memerlukan instalasi tambahan pada perangkat pengguna.

Aplikasi kemudian di-deploy pada server Ubuntu sehingga dapat diakses melalui jaringan menggunakan alamat IP server.

---

## **Tujuan Pengembangan Sistem**

Pengembangan sistem ini bertujuan untuk:
1. Mengotomatisasi proses pembersihan dan pengolahan data transaksi dan interaksi pengguna.
2. Menyediakan dashboard visualisasi data yang sederhana dan informatif.
3. Mengurangi bias dalam pengkategorian produk dengan melakukan pengelompokan data produk yang diunggah oleh principal (penjual).
4. Menyajikan tabel data terstruktur dan terklasifikasi yang dapat di filter berdasarkan kategori atau nama produk.
5. Menyediakan sistem berbasis web yang mudah diakses dan aman, dengan mekanisme autentikasi sederhana untuk penggunaan internal.
6. Mendukung pengambilan keputusan bisnis berbasis data yang lebih objektif, efisien, dan akurat bagi pengguna.

---

## **Fitur Utama Sistem**
Sistem ini memiliki beberapa fitur utama sebagai berikut:

1. Klasifikasi Kategori Produk Otomatis
Sistem dapat memprediksi kategori produk berdasarkan teks produk menggunakan model NLP berbasis IndoBERT.
Kategori produk diprediksi dalam beberapa level hierarki:
* Level 1
* Level 2
* Level 3
* Level 4
* Level 5
* Level 6

Pendekatan ini memungkinkan sistem memahami struktur kategori produk secara lebih mendalam.

2. Upload Dataset Produk
Pengguna dapat mengunggah dataset produk dalam format:
* CSV (.csv)
* Excel (.xlsx)
Dataset yang diunggah akan diproses oleh sistem untuk menghasilkan prediksi kategori produk.

3. Dashboard Analitik Produk
Sistem menyediakan dashboard analitik sederhana yang menampilkan beberapa informasi penting, antara lain:
* KPI (Key Performance Indicator)
* Distribusi kategori produk
* Produk dengan performa tertinggi
* Analisis interaksi pengguna terhadap produk
* Data Grid hasil automasi kategori
* Dashboard ini membantu pengguna memahami performa katalog produk mereka dengan lebih mudah.

4. Export Hasil Analisis
Pengguna dapat mengunduh hasil prediksi kategori produk dalam bentuk file yang dapat digunakan untuk analisis lebih lanjut.

---

## **Teknologi yang Digunakan**

Sistem ini dibangun menggunakan beberapa teknologi berikut:
1. Bahasa Pemrograman:
Python

2. Framework Aplikasi:
Streamlit

3. Model Bahasa:
IndoBERT

4. Library Python:
Beberapa library utama yang digunakan dalam pengembangan sistem ini antara lain:
* PyTorch
* Transformers
* Pandas
* Scikit-learn
* Joblib
* Gdown

5. Sistem Operasi Deployment:
Ubuntu Linux

6. Struktur Dataset:
Dataset yang digunakan oleh sistem harus memiliki beberapa kolom utama sebagai berikut:

* nama_produk                   :Nama produk yang dijual
* deskripsi                     :Deskripsi singkat produk (opsional)
* kawasan                       :Wilayah atau area tempat produk dijual
* item_dilihat                  :Jumlah produk dilihat oleh pengguna
* pendapatan_item               :Total pendapatan dari produk tersebut
* item_dibeli                   :Jumlah transaksi pembelian produk
* item_ditambahkan_keranjang    :Jumlah produk yang ditambahkan ke keranjang

Kolom nama_produk merupakan kolom wajib karena digunakan sebagai input utama pada proses klasifikasi produk.

---

## **Arsitektur Sistem**

Sistem ini dikembangkan melalui beberapa tahapan utama dalam pipeline machine learning:

1. Data Preparation
Pembersihan dan normalisasi data produk sebelum digunakan dalam pelatihan model.

2. Feature Engineering
Ekstraksi fitur teks menggunakan model bahasa IndoBERT.

3. Label Encoding
Transformasi kategori produk menjadi representasi numerik menggunakan LabelEncoder.

4. Model Training
Pelatihan model klasifikasi multi-level untuk memprediksi kategori produk.

5. Model Saving
Penyimpanan model hasil pelatihan dalam bentuk file yang dapat digunakan pada tahap deployment.

6. Application Development
Pembuatan aplikasi berbasis Streamlit untuk menghubungkan model dengan antarmuka pengguna.

7. Deployment
Aplikasi di-deploy pada server Ubuntu sehingga dapat diakses melalui jaringan.

--- 

## **Struktur Repository**

Berikut contoh struktur folder proyek:
project_directory
│
├── app.py
├── requirements.txt
│
├── model_artifacts
│   ├── config.json
│   ├── label_encoders.pkl
│   ├── tree_maps.pkl
│   └── model_HMC.pt
│
├── training_model.ipynb
│
└── dataset

Penjelasan:

File	
* app.py                :Script utama aplikasi Streamlit
* requirements.txt      :Daftar dependency Python
* model_artifacts       :Folder penyimpanan model
* training_model.ipynb  :Notebook untuk training model
* dataset               :Dataset pelatihan

---

## **Dokumentasi Panduan Penggunaan**

Untuk mempermudah penggunaan dan pengelolaan sistem, telah disediakan beberapa dokumen panduan terpisah.

1. Panduan Deployment Sistem
Dokumen ini menjelaskan langkah-langkah teknis untuk melakukan deployment aplikasi pada server Ubuntu, mulai dari instalasi dependency hingga menjalankan aplikasi pada server.
Silakan akses panduan lengkap melalui tautan berikut

Guidebook Deployment Ubuntu
Klik tautan berikut:
[[Panduan Deployment Sistem](https://docs.google.com/document/d/1YCWwaBj38rg4NyVZngEvR_t_q4MobA_GrqFUAdOQheQ/edit?usp=sharing)]

2. Panduan Pengguna (User Guide)
Dokumen ini menjelaskan cara menggunakan aplikasi dari sudut pandang pengguna, termasuk:
* cara mengunggah dataset
* cara menjalankan proses kategorisasi
* cara membaca dashboard analitik
* cara mengunduh hasil analisis
Silakan akses panduan lengkap melalui tautan berikut:

Guidebook User
Klik tautan berikut:
[[Panduan Pengguna](https://docs.google.com/document/d/1Mf9-M8Y0q0eEiL9qFSYp9cdHrfRs4q1D__aFZAVQ1Bo/edit?usp=sharing)]

3. Panduan Developer (Developer Guide)
Dokumen ini ditujukan bagi pengembang yang ingin memahami struktur sistem serta melakukan maintenance atau pengembangan lebih lanjut.
Panduan ini mencakup:
* arsitektur sistem
* pipeline machine learning
* proses retraining model
* penggantian model pada server
* proses deployment ulang aplikasi
Silakan akses panduan lengkap melalui tautan berikut:

Guidebook Developer
Klik tautan berikut:
[[Panduan Developer]([https://docs.google.com/document/d/1L7Te-uzbWoPafjgZvKKcMlSWmK5cpON7z7tWu178uF0/edit?usp=sharing)]

--- 

## **Maintenance Sistem**

Model machine learning dapat mengalami penurunan performa seiring waktu akibat perubahan pola data produk. Oleh karena itu, sistem perlu dilakukan pemeliharaan secara berkala melalui:

* Evaluasi performa model
* Pembaruan dataset pelatihan
* Retraining model menggunakan data terbaru
* Deployment ulang model yang telah diperbarui

Panduan lengkap mengenai proses maintenance tersedia pada Developer Guide.

---

## **Kesimpulan**

Sistem ini dirancang untuk membantu pengguna dalam mengelola dan menganalisis data produk pada platform e-commerce SIPLAH secara lebih efektif.
Dengan menggabungkan teknologi Natural Language Processing, machine learning, dan dashboard interaktif, sistem ini mampu memberikan solusi yang praktis untuk:
* mengotomatisasi proses kategorisasi produk
* meningkatkan kualitas pengelolaan katalog
* menyediakan analisis data produk yang lebih mudah dipahami.

---

## **Tim Penyusun**

1. Azriel Kevin Alfarizqi - Sains Data - Universitas Negeri Surabaya
2. Sheira EN Nadia - Sains Data - Universitas Negeri Surabaya
3. Devina Gusriani - Sains Data - Universitas Negeri Surabaya


© Data Flow Automation Project
