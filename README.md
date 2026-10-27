# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan yang telah berdiri sejak tahun 2000 dan telah meluluskan banyak siswa dengan reputasi baik. Namun, institusi ini menghadapi tantangan signifikan terkait tingginya angka siswa yang tidak menyelesaikan pendidikannya alias dropout.

Tingginya tingkat dropout ini menjadi masalah serius karena berdampak pada reputasi institusi, efektivitas proses pembelajaran, dan potensi kehilangan sumber daya manusia yang seharusnya dapat berkontribusi positif. Dampak negatif yang muncul antara lain:

- Penurunan Tingkat Kelulusan dan Reputasi
Siswa yang berhenti sebelum menyelesaikan studi menurunkan rasio kelulusan, yang pada akhirnya dapat memengaruhi persepsi calon siswa dan pihak terkait terhadap kualitas institusi.

- Pemborosan Sumber Daya Pendidikan
Dropout yang tinggi menyebabkan penggunaan sumber daya pendidikan, waktu, dan biaya menjadi tidak efisien karena investasi yang dilakukan tidak berbuah hasil sesuai harapan.

- Gangguan dalam Perencanaan Akademik dan Administratif
Ketidakterdugaan jumlah siswa yang keluar mempersulit pengelolaan kelas, alokasi dosen, dan perencanaan anggaran pendidikan.

Manajemen Jaya Jaya Institut menyadari pentingnya deteksi dini terhadap siswa yang berisiko dropout agar dapat diberikan bimbingan khusus dan intervensi tepat waktu. Oleh karena itu, proyek ini bertujuan mengembangkan sistem prediksi risiko dropout berbasis data untuk membantu institusi mengambil langkah preventif, meningkatkan retensi siswa, dan menjaga kualitas pendidikan secara berkelanjutan.

### Permasalahan Bisnis
Proyek ini berangkat dari sejumlah tantangan nyata yang dihadapi oleh institusi pendidikan, antara lain:

- Tingginya angka dropout mahasiswa yang melebihi 30% dalam beberapa tahun terakhir, yang berdampak pada efisiensi operasional dan reputasi institusi.

- Minimnya pemahaman menyeluruh mengenai faktor-faktor yang mendorong mahasiswa berhenti studi, baik dari sisi akademik, finansial, maupun latar belakang sosial.

- Kesulitan dalam mengidentifikasi mahasiswa yang berisiko tinggi mengalami dropout sejak dini, sehingga upaya intervensi sering kali terlambat dilakukan.

- Tidak adanya sistem pendukung berbasis data yang mampu membantu pihak institusi memantau dan menganalisis risiko dropout secara real-time dan terstruktur.

Dengan mengatasi permasalahan-permasalahan tersebut, institusi diharapkan dapat meningkatkan efektivitas strategi retensi mahasiswa, mengurangi biaya operasional akibat tingkat dropout yang tinggi, serta memperkuat citra sebagai lembaga pendidikan yang peduli terhadap keberhasilan studi mahasiswanya.

### Cakupan Proyek

1. **Eksplorasi dan Pemahaman Data**

    Melakukan analisis awal terhadap dataset data.csv untuk memahami karakteristik siswa dan distribusi tingkat dropout.

2. **Data Preparation**

    Membersihkan dan mentransformasi data agar siap digunakan dalam proses pemodelan machine learning.

3. **Analisis Faktor Penyebab Dropout**

    Membuat visualisasi hubungan antara setiap fitur dengan variabel Target untuk mengidentifikasi pola yang signifikan.

3. **Modeling**

    Membangun model prediktif untuk mengidentifikasi siswa yang berpotensi keluar.

4. **Pembuatan Business Dashboard**

    Menyajikan hasil analisis dan model dalam bentuk dashboard interaktif yang mudah dipahami dan dapat digunakan oleh tim Manajemen Jaya Jaya Institut.

5. **Kesimpulan dan Rekomendasi**

    Menyimpulkan hasil analisis dan pemodelan, serta memberikan rekomendasi strategis untuk mengurangi siswa dropout.


### Persiapan

Sumber data: [Student's Performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

- Setup environment - Anaconda:
```
conda create --name main-ds python=3.9```

conda activate main-ds

pip install -r requirements.txt
```

- Setup environment - Shell/Terminal:

``` 
pip install pipenv 
pipenv install
pipenv shell
pip install -r requirements.txt 
```


1. Menjalankan `notebook.ipynb`
- Install `requirements.txt`
- Jalankan seluruh cell pada notebook menggunakan Google Colab/Jupyter Notebook untuk melihat hasil analisis data.

2. Menjalankan Dashboard

Pastikan docker sudah terinstall.
- Jalankan perintah berikut:

```
docker pull metabase/metabase:v0.46.4
```

- Jalankan container Metabase menggunakan perintah berikut:

```
docker run -p 5000:3000 --name student-dropout metabase/metabase
```

- Login ke Metabase menggunakan username dan password berikut:

```
username : spectre02black@gmail.com
password : studentDropout123
```


## Business Dashboard
Hasil analisis dan model prediktif divisualisasikan dalam dashboard untuk membantu tim Manajemen Jaya Jaya Institut memantau dan memahami faktor-faktor yang memengaruhi tingkat dropout siswa. Berikut adalah elemen-elemen utama dalam dashboard:
1. **Distribusi Siswa berdasarkan Status**:
   - Visualisasi jumlah siswa berdasarkan status (dropout, aktif, lulus)
2. **Feature Importance**:
   - Visualisasi 10 fitur yang berpengaruh terhadap tingkat dropout siswa berdasarkan model Random Forest.

## Menjalankan Sistem Machine Learning

**Live Demo**: [Streamlit App](https://student-dropout-prediction-wqkmsvfa765gpc8fxayi33.streamlit.app/)

**Jalankan di lokal**:

```
pip install -r requirements.txt
streamlit run app.py
```


## Conclusion

Proyek ini bertujuan untuk memahami faktor-faktor yang memengaruhi tingkat **dropout** (keluar) siswa di **Jaya Jaya Institut** dan membangun model prediktif untuk mengidentifikasi siswa dengan risiko keluar tinggi. Berikut adalah temuan utama dan insight yang diperoleh:

#### **1. Faktor-Faktor Penyebab dropout**
Berdasarkan hasil analisis model Random Forest, berikut adalah faktor-faktor utama yang berkontribusi terhadap kemungkinan seorang mahasiswa mengalami dropout:
1. **overall_approval_rate**:
   - Mahasiswa dengan rasio kelulusan seluruh mata kuliahnya rendah cenderung mengalami kesulitan akademik, yang menjadi indikator kuat terhadap risiko dropout.
   

2. **approval_rate_2nd**:
   - Semester kedua mencerminkan adaptasi lanjutan mahasiswa terhadap sistem perkuliahan. Jika performanya menurun atau tetap rendah, besar kemungkinan mahasiswa mulai kehilangan motivasi atau mengalami tekanan akademik.
   

3. **Admission grade**:
   - Mahasiswa dengan nilai masuk rendah cenderung memiliki kesiapan akademik yang kurang, sehingga berisiko lebih tinggi untuk tidak dapat mengikuti beban studi.

4. **Fitur Pendukung Lain**:
   - Selain ketiga fitur utama di atas, terdapat pula beberapa fitur pendukung lain yang turut memengaruhi keputusan mahasiswa untuk dropout, seperti jumlah total mata kuliah yang lulus (total_approved), persentase kelulusan semester pertama (approval_rate_1st), tingkat keikutsertaan evaluasi di awal studi (eval_ratio_1st), serta kondisi pembayaran biaya kuliah (tuition fees up to date). Faktor-faktor ini secara kolektif memberikan gambaran mengenai konsistensi performa akademik, keterlibatan mahasiswa, dan dukungan finansial yang mereka miliki. Meskipun pengaruhnya tidak sebesar tiga fitur utama, keberadaan fitur-fitur ini tetap penting untuk memberikan konteks yang lebih menyeluruh dalam memprediksi risiko dropout.

#### **2. Pemilihan Model Prediktif**
Setelah melakukan evaluasi beberapa algoritma, model terbaik yang dipilih adalah Regression, dengan alasan model ini memiliki recall yang lebih tinggi daripada model lain.
- **Accuracy**: 82%
- **Precision**: 85%
- **Recall**: 81%
- **F1-Score**: 83%


#### **3. Feature Importance**
Dari analisis feature importance menggunakan model Random Forest:
- **overall_approval_rate** menjadi prediktor terkuat terhadap risiko dropout.
- **approval_rate_2nd** dan **Admission grade** juga memiliki peran penting.


---

### **Jawaban terhadap Pertanyaan Bisnis**
1. **Apa faktor utama yang memengaruhi dropout?**
   - **overall_approval_rate**, **approval_rate_2nd**, dan **Admission grade** menjadi faktor dominan.

2. **Bagaimana pengaruh latar belakang pendidikan orang tua terhadap risiko dropout, dan bagaimana institusi pendidikan dapat merespons temuan ini?**
   - Latar belakang pendidikan orang tua memiliki pengaruh terhadap risiko dropout siswa, meskipun tidak sebesar faktor-faktor akademik seperti nilai ujian masuk atau tingkat kelulusan mata kuliah. Siswa yang orang tuanya memiliki pendidikan yang lebih rendah cenderung menghadapi risiko dropout yang sedikit lebih tinggi. Hal ini mungkin disebabkan oleh keterbatasan dukungan akademik dan motivasi di lingkungan rumah.

3. **Apa pola umum dari siswa berisiko tinggi dropout?**
   - Umumnya adalah siswa yang ratio kelulusan mata kuliahnya rendah.

4. **Apakah kita bisa memonitor risiko dropout secara aktif?**
   - Ya, model prediktif yang dibangun dapat diintegrasikan ke dalam dashboard sebagai alat bantu pemantauan real-time.

---

### **Karakteristik Umum siswa yang Melakukan dropout**
Berdasarkan analisis data, berikut adalah karakteristik umum siswa yang melakukan dropout:
- **Status Perkawinan**: Mayoritas siswa yang dropout adalah yang masih menikah secara hukum tapi sudah pisah secara legal, menikah, atau cerai.
- **Kelas dan Demografi**: Kebanyakan siswa perempuan, mengikuti kelas malam, bukan mahasiswa internasional, dan tidak menggunakan beasiswa.
- **Keuangan**: Siswa yang belum membayar SPP dan memiliki pinjaman lebih rentan dropout.
- **Jalur Masuk**: Siswa yang masuk lewat jalur usia di atas 23 tahun, jalur peraturan pemerintah, atau yang sudah pernah kuliah/diploma khusus.
- **Latar Belakang Orang Tua**: Orang tua dengan pendidikan rendah atau tidak sekolah.
- **Faktor Akademik**:
   - Nilai ujian masuk rendah.

   - Usia saat enrollment tinggi.

   - Rasio kelulusan mata kuliah (approval_rate_1st, approval_rate_2nd, overall_approval_rate) jauh dari sempurna.

   - Jumlah mata kuliah lulus dan diambil selama 2 semester di bawah 10.



---

### **Rekomendasi Action Items untuk Institut**
1. **Optimalkan Beban Studi dan Pengambilan Mata Kuliah**:
   - Pantau dan bimbing siswa agar mengambil jumlah mata kuliah yang sesuai kemampuan, terutama di dua semester awal, supaya tidak mengalami overload yang berisiko menyebabkan dropout.

2. **Tingkatkan Pendampingan Akademik Berdasarkan Approval Rate**
   - Fokuskan program pembinaan pada siswa dengan approval rate (tingkat kelulusan mata kuliah) rendah agar mereka mendapat dukungan belajar tambahan, seperti tutor atau bimbingan khusus.

3. **Perbaiki Proses Seleksi dan Onboarding Siswa Baru**:
   - Gunakan Admission Grade sebagai indikator awal untuk memberikan perhatian lebih pada siswa dengan nilai masuk rendah, serta kembangkan program orientasi dan mentoring agar adaptasi lebih baik.

4. **Manfaatkan Model Prediktif untuk Identifikasi Risiko Dropout**:
   - Terapkan model prediktif yang menggabungkan fitur-fitur utama (overall approval rate, approval rate semester 2, admission grade) dalam dashboard manajemen agar pengambilan keputusan berbasis data lebih cepat dan tepat sasaran.
