# Laporan Proyek Machine Learning - Auliya Sabrina Vyantika

## Domain Proyek

erkembangan teknologi informasi berbasis komputer telah mempengaruhi berbagai sektor, termasuk dalam dunia perbankan. Salah satu permasalahan yang sering dihadapi oleh Bank ABC adalah kesalahan dalam pengambilan keputusan pemberian kredit. Hal ini dapat mengakibatkan kerugian bagi pihak bank jika kredit yang diberikan tidak dapat dibayar kembali oleh nasabah, atau jika terjadi kredit macet. Untuk itu, bank perlu mengevaluasi kelayakan nasabah dalam memenuhi kewajiban pembayaran pinjaman sebelum memberikan persetujuan kredit.

Masalah ini dapat diatasi dengan menggunakan pendekatan machine learning, khususnya dengan metode klasifikasi. Algoritma klasifikasi seperti decision tree, random forest, support vector machine, dan lainnya yang dapat digunakan untuk menganalisis data nasabah, seperti status pembayaran cicilan kredit sebelumnya. Setiap algoritma memiliki keunggulan tersendiri dalam menangani jenis data yang berbeda dan memprediksi apakah nasabah berpotensi mengalami kesulitan dalam pembayaran pinjaman. Dengan membandingkan kinerja berbagai algoritma, bank dapat memilih model yang paling tepat untuk meminimalkan risiko kesalahan dalam pengambilan keputusan kredit, sehingga dapat memberikan keputusan yang lebih akurat dan efisien. Proses evaluasi kredit yang lebih berbasis data ini dapat mengurangi risiko kerugian akibat kredit macet.
  
  [PENERAPAN METODE PERBANDINGAN EKSPONENSIAL PADA SISTEM PENDUKUNG KEPUTUSAN PEMBERIAN KREDIT PADA BANK XYZ](http://jurnal.borneo.ac.id/index.php/borneo_saintek/article/view/911)

  [Perkembangan Teknologi Informasi Terhadap Peningkatan Bisnis Online](http://interdisiplin.my.id/index.php/i/article/view/5)


## Business Understanding

Pada bagian ini, akan dibahas mengenai klarifikasi masalah yang ada serta tujuan dan solusi yang dapat dicapai dengan menggunakan pendekatan machine learning dalam proses pengambilan keputusan pemberian kredit di Bank ABC.

### Problem Statements

1. Bank ABC sering menghadapi kesalahan dalam pengambilan keputusan pemberian kredit kepada nasabah, yang dapat mengakibatkan kredit macet dan kerugian finansial. 
2. Tidak adanya sistem yang efisien dan berbasis data untuk mengevaluasi kelayakan nasabah dalam membayar kembali pinjaman, sehingga keputusan pemberian kredit kurang tepat.
3. Kurangnya pengolahan data historis yang relevan, seperti riwayat pembayaran cicilan kredit, yang dapat digunakan untuk memprediksi kemampuan nasabah dalam memenuhi kewajiban kreditnya.

### Goals

1. Menerapkan model machine learning untuk menganalisis data nasabah secara lebih akurat dan mengurangi kesalahan dalam pengambilan keputusan pemberian kredit, sehingga mengurangi risiko kredit macet.
2. Mengembangkan sistem berbasis machine learning yang dapat mengevaluasi kelayakan nasabah dalam membayar pinjaman secara otomatis dan efisien, dengan mempertimbangkan data historis dan karakteristik nasabah.
3. Menggunakan algoritma klasifikasi seperti decision tree, random forest, dan support vector machine untuk menganalisis data nasabah dan memprediksi kemungkinan terjadinya kredit macet berdasarkan riwayat pembayaran cicilan sebelumnya.

### Solution Statements

1. Menggunakan beberapa algoritma klasifikasi seperti decision tree, random forest, dan support vector machine untuk membandingkan kinerjanya dalam memprediksi nasabah yang berpotensi mengalami kesulitan pembayaran. Model yang memberikan akurasi tertinggi akan dipilih untuk diterapkan pada sistem evaluasi kredit.
2. Melakukan tuning hyperparameter pada model baseline yang telah diterapkan untuk meningkatkan kinerja model. Dengan teknik seperti grid search atau random search, parameter yang optimal dapat ditemukan untuk meningkatkan akurasi dan performa model dalam memprediksi kredit macet.

### Metrik Evaluasi

- **Accuracy**: Untuk mengukur seberapa banyak prediksi yang benar dibandingkan dengan total prediksi yang dibuat oleh model.
- **Precision**: Mengukur seberapa tepat model dalam memprediksi nasabah yang akan mengalami kredit macet (positif).
- **Recall**: Mengukur seberapa baik model dalam menangkap semua nasabah yang berpotensi mengalami kredit macet.
- **F1-Score**: Menyediakan keseimbangan antara precision dan recall, untuk memastikan bahwa model tidak hanya tepat, tetapi juga tidak melewatkan banyak nasabah yang berisiko.


## Data Understanding

Dataset yang digunakan berfokus pada data nasabah bank, yang mencakup informasi terkait status pembayaran kredit dan berbagai karakteristik lainnya yang berhubungan dengan kelayakan pemberian kredit. Dataset ini digunakan untuk menganalisis dan memprediksi kemungkinan kredit macet berdasarkan data historis nasabah.

Dataset yang digunakan dalam proyek ini dapat diunduh melalui [GitHub](https://github.com/Salmanab16/kredit-macet). Dataset ini terdiri dari beberapa fitur, seperti status pembayaran kredit sebelumnya, umur, penghasilan, jumlah pinjaman, dan lainnya. Dengan data ini, model machine learning akan dilatih untuk memprediksi apakah seorang nasabah berpotensi mengalami kredit macet atau tidak.

Data ini dapat digunakan untuk pelatihan model klasifikasi dengan berbagai algoritma seperti decision tree, random forest, dan support vector machine, untuk membantu pihak bank dalam pengambilan keputusan pemberian kredit yang lebih akurat dan efisien.

### Variabel-variabel pada kredit-macet dataset adalah sebagai berikut:
- **`jenis_kelamin`**: Jenis kelamin nasabah yang terdiri dari dua kategori, yaitu P (Perempuan) dan L (Laki-laki).
- **`umur`**: Usia nasabah yang menunjukkan usia saat aplikasi kredit diajukan.
- **`jml_pinjaman`**: Jumlah pinjaman yang diajukan oleh nasabah.
- **`jkw`**: Jangka waktu pinjaman dalam satuan bulan.
- **`jml_angsuran_per_bulan`**: Jumlah angsuran yang harus dibayar nasabah setiap bulan.
- **`type_pinjaman`**: Tipe pinjaman yang diberikan, seperti konsumsi, investasi, atau lainnya.
- **`jenis_pinjaman`**: Jenis pinjaman yang diajukan nasabah, misalnya pinjaman untuk rumah, kendaraan, atau pendidikan.
- **`bi_sektor_ekonomi`**: Kode sektor ekonomi yang digunakan oleh Bank Indonesia (BI) untuk mengklasifikasikan sektor ekonomi nasabah.
- **`bi_golongan_debitur`**: Golongan debitur yang diberikan oleh Bank Indonesia berdasarkan tingkat risiko.
- **`bi_gol_penjamin`**: Golongan penjamin yang digunakan oleh Bank Indonesia untuk menilai jenis jaminan yang diberikan.
- **`saldo_nominatif`**: Saldo nominatif nasabah, yang menunjukkan saldo utang yang terdaftar pada bank.
- **`tunggakan_pokok`**: Tunggakan pokok yang harus dibayar oleh nasabah, yaitu jumlah pokok yang tertunggak dalam pembayaran.
- **`tunggakan_bunga`**: Tunggakan bunga yang harus dibayar oleh nasabah, yaitu bunga yang belum dibayar hingga saat ini.
- **`status kredit`**: Status kredit nasabah, apakah nasabah tersebut lancar atau mengalami masalah seperti kredit macet.

Variabel-variabel ini digunakan untuk menganalisis dan memprediksi kemungkinan terjadinya kredit macet pada nasabah, yang akan membantu bank dalam pengambilan keputusan pemberian kredit yang lebih tepat dan efisien.

### Exploratory Data Analysis (EDA)

Pada tahap awal, dilakukan beberapa analisis untuk memahami data lebih dalam. Hasil analisis menunjukkan beberapa hal penting sebagai berikut:

1. **Ukuran Dataset**:
   Dataset terdiri dari 766 baris data dengan 16 kolom.

   ![Deskripsi Gambar](https://drive.google.com/uc?export=view&id=1SHY3ez8cRwliTN2zxmZjb8_KYGI3VWF0)



## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

