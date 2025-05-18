# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan yang telah berdiri sejak tahun 2000 dan telah meluluskan banyak siswa dengan reputasi baik. Namun, institusi ini menghadapi tantangan signifikan terkait tingginya angka siswa yang tidak menyelesaikan pendidikannya alias dropout.

Tingginya tingkat dropout ini menjadi masalah serius karena berdampak pada reputasi institusi, efektivitas proses pembelajaran, dan potensi kehilangan sumber daya manusia yang seharusnya dapat berkontribusi positif. Dampak negatif yang muncul antara lain:

- Penurunan Tingkat Kelulusan dan Reputasi
Siswa yang berhenti sebelum menyelesaikan studi menurunkan rasio kelulusan, yang pada akhirnya dapat memengaruhi persepsi calon siswa dan pihak terkait terhadap kualitas institusi.

- Penurunan Produktivitas Organisasi
Pergantian karyawan yang tinggi berdampak pada hilangnya pengalaman kerja yang berharga, sehingga memengaruhi pencapaian target bisnis. Variabel seperti YearsAtCompany dan YearsWithCurrManager menjadi indikator penting untuk memahami kontribusi pengalaman terhadap stabilitas dan produktivitas tim.

- Pemborosan Sumber Daya Pendidikan
Dropout yang tinggi menyebabkan penggunaan sumber daya pendidikan, waktu, dan biaya menjadi tidak efisien karena investasi yang dilakukan tidak berbuah hasil sesuai harapan.

- Gangguan dalam Perencanaan Akademik dan Administratif
Ketidakterdugaan jumlah siswa yang keluar mempersulit pengelolaan kelas, alokasi dosen, dan perencanaan anggaran pendidikan.

Manajemen Jaya Jaya Institut menyadari pentingnya deteksi dini terhadap siswa yang berisiko dropout agar dapat diberikan bimbingan khusus dan intervensi tepat waktu. Oleh karena itu, proyek ini bertujuan mengembangkan sistem prediksi risiko dropout berbasis data untuk membantu institusi mengambil langkah preventif, meningkatkan retensi siswa, dan menjaga kualitas pendidikan secara berkelanjutan.


### Permasalahan Bisnis

Permasalahan bisnis yang akan diselesaikan melalui proyek ini meliputi:

1. Mengidentifikasi faktor-faktor utama yang memengaruhi tingkat dropout siswa, mulai dari aspek demografis, prestasi akademik, hingga kondisi finansial dan latar belakang keluarga.

2. Mengukur pengaruh variabel-variabel seperti nilai ujian masuk, rasio kelulusan mata kuliah, dan usia saat pendaftaran terhadap risiko siswa melakukan dropout.

3. Menganalisis pola-pola karakteristik siswa yang berisiko tinggi dropout, termasuk jumlah mata kuliah yang diambil dan diselesaikan, serta keterkaitan dengan faktor-faktor lain seperti jalur masuk dan status pembayaran.

4. Menyediakan alat bantu analisis dalam bentuk dashboard interaktif yang memudahkan pihak institusi dalam memantau, menganalisis, dan mengambil tindakan preventif terhadap siswa yang berpotensi dropout.

Dengan menyelesaikan permasalahan-permasalahan tersebut, perusahaan diharapkan dapat:

- Mengurangi tingkat dropout melalui intervensi dan bimbingan yang lebih tepat sasaran.

- Menyusun strategi retensi siswa yang efektif berdasarkan data yang akurat.

- Meningkatkan kualitas pembelajaran dan reputasi institusi secara keseluruhan.


### Cakupan Proyek

1. **Eksplorasi dan Pemahaman Data**

    Melakukan analisis awal terhadap dataset data.csv untuk memahami karakteristik siswa dan distribusi tingkat dropout.

2. **Data Preparation**

    Membersihkan dan mentransformasi data agar siap digunakan dalam proses pemodelan machine learning.

3. **Analisis Faktor Penyebab Dropout**

    Membuat visualisasi hubungan antara setiap fitur dengan variabel Target untuk mengidentifikasi pola yang signifikan.

3. **Modeling**

    Membangun model prediktif untuk mengidentifikasi karyawan yang berpotensi keluar dari perusahaan.

4. **Pembuatan Business Dashboard**

    Menyajikan hasil analisis dan model dalam bentuk dashboard interaktif yang mudah dipahami dan dapat digunakan oleh tim Manajemen Jaya Jaya Institut.

5. **Kesimpulan dan Rekomendasi**

    Menyimpulkan hasil analisis dan pemodelan, serta memberikan rekomendasi strategis untuk mengurangi siswa dropout.


### Persiapan

Sumber data: [Student's Performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

Setup environment:

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

Hasil dari analisis dan model prediktif dapat divisualisasikan dalam bentuk dashboard untuk membantu tim HR memantau dan memahami attrition secara real-time. Berikut adalah elemen-elemen yang dapat disertakan dalam dashboard:
1. **Distribusi Attrition**:
   - Visualisasi attrition per Departemen dan JobRole.
2. **Feature Importance**:
   - Visualisasi kontribusi 5 fitur paling penting terhadap risiko attrition berdasarkan model Logistic Regression.
3. **Prediksi Risiko**:
   - Menampilkan daftar karyawan dengan risiko tinggi berdasarkan hasil prediksi model Logistic Regression.


## Conclusion

Proyek ini berhasil mencapai tujuan utama yaitu:
1. **Mengidentifikasi faktor utama penyebab attrition**:
   - Faktor terpenting adalah rasio kelulusan seluruh mata kuliah (**overall_approval_rate**), rasio kelulusan mata kuliah di semester 2 (**approval_rate_2nd**), dan nilai ujian masuk (**Admission grade**).

2. **Membangun model prediktif yang akurat**:
   - Model Random Forest memberikan performa yang baik dengan **Recall 81%** dan metrik lainnya memiliki nilai yang sedikit lebih tinggi.

3. **Memberikan rekomendasi actionable**:
   - Tim Manajemen Jaya Jaya Institut dapat mengimplementasikan kebijakan untuk mengurangi tingkat siswa dropout.


### **Rekomendasi Action Items untuk Institut**
1. **Optimalkan Beban Studi dan Pengambilan Mata Kuliah**:
   - Pantau dan bimbing siswa agar mengambil jumlah mata kuliah yang sesuai kemampuan, terutama di dua semester awal, supaya tidak mengalami overload yang berisiko menyebabkan dropout.

2. **Tingkatkan Pendampingan Akademik Berdasarkan Approval Rate**
   - Fokuskan program pembinaan pada siswa dengan approval rate (tingkat kelulusan mata kuliah) rendah agar mereka mendapat dukungan belajar tambahan, seperti tutor atau bimbingan khusus.

3. **Perbaiki Proses Seleksi dan Onboarding Siswa Baru**:
   - Gunakan Admission Grade sebagai indikator awal untuk memberikan perhatian lebih pada siswa dengan nilai masuk rendah, serta kembangkan program orientasi dan mentoring agar adaptasi lebih baik.

4. **Manfaatkan Model Prediktif untuk Identifikasi Risiko Dropout**:
   - Terapkan model prediktif yang menggabungkan fitur-fitur utama (overall approval rate, approval rate semester 2, admission grade) dalam dashboard manajemen agar pengambilan keputusan berbasis data lebih cepat dan tepat sasaran.