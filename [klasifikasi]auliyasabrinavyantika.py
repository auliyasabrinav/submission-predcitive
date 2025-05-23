# -*- coding: utf-8 -*-
"""[Klasifikasi]AuliyaSabrinaVyantika.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10P2F1VsNDrpzGqWpEKzjHgGeeRWyr9Ax

# **1. Import Library**

Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
!pip install scikit-optimize
from skopt import BayesSearchCV

"""# **2. Memuat Dataset**"""

# Baca file CSV dari URL
df = pd.read_csv('/content/drive/MyDrive/Data Science/Data/creditapproval-data_kotor.csv')

# Tampilkan DataFrame untuk memastikan telah dibaca dengan benar
df.head()

"""**Pendefinisian Variabel**

<br>

`jenis_kelamin` = Jenis kelamin terdiri dari P dan L \
`umur`          = usia nasabah \
`jml_pinjaman`  = jumlah pinjaman nasabah \
`jkw`           = jangka waktu (bulan) \
`jml_angsuran_per_bulan` = jumlah angsuran yang harus dibayar tiap bulan \
`type_pinjaman`  = tipe pinjaman \
`jenis_pinjaman`  = jenis pinjaman \
`bi_sektor_ekonomi` = Sektor Ekonomi BI \
`col` \
`bi_golongan_debitur` = golongan debitur \
`bi_gol_penjamin` = golongan penjamin \
`saldo_nominatif` = saldo nominatif nasabah
<br>
`tunggakan_pokok` = tunggakan pokok yang harus dibayar nasabah
<br>
`tunggakan_bunga` = tunggakan bunga yang harus dibayar nasabah  <br>
`status kredit` = status kredit nasabah

# **3. Data Preprosessing**
"""

df.shape
#menghitung jumlah baris dan kolom

print(df.info())

duplicate = df[df.duplicated()]
print("Jumlah Data yang Duplikat : ", duplicate.shape)

#mengecek jumlah missing value
df.isnull().sum()

"""- terdapat 9 row kolom `umur` memiliki missing value
- terdapat 8 row kolom `jkw` memiliki missing value
- terdapat 1 row kolom `bi_sektor_ekonomi` memiliki missing value
"""

data = df.dropna() # data dihapus tapi file asli masih utuh
#df.dropna(inplace = True) akan menghapus juga di data aslinya
print(data.info())

#mengecek kembali jumlah missing value
data.isnull().sum()

data.describe(include='all')

data.nunique()

"""Dapat dilihat terdapat beberapa kolom memiliki tipe data `object` sedangkan seharusnya adalah `category` agar lebih efektif karena memiliki jumlah nilai unik sedikit. Kolom tersebut adalah `Jenis Kelamin` dan kolom `status kredit`."""

# Mengumpulkan kolom-kolom yang dingin diubah pada 1 list
kolom_diubah = ['jenis_kelamin', 'status kredit']

data[kolom_diubah] = data[kolom_diubah].astype('category')
data.dtypes

data['jenis_kelamin'].unique()

data["jenis_kelamin"] = data["jenis_kelamin"].replace("WANITA", "P")
data["jenis_kelamin"] = data["jenis_kelamin"].replace("PEREMPUAN", "P")
data["jenis_kelamin"] = data["jenis_kelamin"].replace("LAKI-LAKI", "L")
data["jenis_kelamin"] = data["jenis_kelamin"].replace("PRIA", "L")

data['jenis_kelamin'].unique()

target = [ 'jenis_kelamin']
label_encoder = LabelEncoder()

# Apply Label Encoding to the target columns
for column in target:
    data[column] = label_encoder.fit_transform(data[column])

# Display the DataFrame 'data' after Label Encoding
data[target]

"""# **3. Data Splitting**

Tahap Data Splitting bertujuan untuk memisahkan dataset menjadi dua bagian: data latih (training set) dan data uji (test set).
"""

# Buat instance MinMaxScaler
scaler = MinMaxScaler()

# Normalisasi semua kolom numerik
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Pisahkan fitur (X) dan target (y)
kolom = ['nama_nasabah','status kredit', 'col', 'bi_golongan_debitur',
         'bi_gol_penjamin', 'bi_sektor_ekonomi','type_pinjaman','jenis_pinjaman']
X = data.drop(columns=kolom)
y = data['status kredit']

# Pastikan target (y) adalah tipe kategori atau integer
y = y.astype('category')


# Split data menjadi set pelatihan dan set uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tampilkan bentuk set pelatihan dan set uji untuk memastikan split
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

"""# **4. Membangun Model Klasifikasi**

## **a. Membangun Model Klasifikasi**

Setelah memilih algoritma klasifikasi yang sesuai, langkah selanjutnya adalah melatih model menggunakan data latih.

Berikut adalah rekomendasi tahapannya.
1. Pilih algoritma klasifikasi yang sesuai, seperti Logistic Regression, Decision Tree, Random Forest, atau K-Nearest Neighbors (KNN).
2. Latih model menggunakan data latih.
"""

# Part 1: Model Training
# Train each classifier separately
knn = KNeighborsClassifier().fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)

print("Model training selesai.")

"""Dalam proyek ini, saya menerapkan beberapa algoritma untuk melakukan klasifikasi pada data hasil clustering. Tujuan utama dari penerapan berbagai algoritma ini adalah untuk membandingkan akurasi yang dihasilkan oleh masing-masing model dan menemukan algoritma terbaik yang dapat digunakan untuk klasifikasi data secara optimal.

a. K-Nearest Neighbors (KNN)
- Metode berbasis kedekatan antar data (proximity-based).
- Menentukan kelas suatu data berdasarkan mayoritas kelas dari k-tetangga terdekatnya.

b. Decision Tree (DT)
- Algoritma berbasis pohon keputusan yang membagi data ke dalam cabang-cabang berdasarkan aturan keputusan.
- Cocok untuk data dengan relasi non-linear yang kompleks.

c. Random Forest (RF)
- Kombinasi dari banyak pohon keputusan (ensemble learning) untuk meningkatkan akurasi dan mengurangi overfitting.
- Setiap pohon dalam Random Forest dilatih pada subset data yang berbeda.

d. Support Vector Machine (SVM)
- Mencari hyperplane terbaik yang memisahkan data ke dalam kelas-kelas yang berbeda.
- Cocok untuk data dengan dimensi tinggi dan pola yang kompleks.

e. Naïve Bayes (NB)
- Algoritma berbasis probabilitas yang menggunakan Teorema Bayes.
- Mengasumsikan bahwa setiap fitur bersifat independen satu sama lain.

## **b. Evaluasi Model Klasifikasi**

Berikut adalah **rekomendasi** tahapannya.
1. Lakukan prediksi menggunakan data uji.
2. Hitung metrik evaluasi seperti Accuracy dan F1-Score (Opsional: Precision dan Recall).
3. Buat confusion matrix untuk melihat detail prediksi benar dan salah.
"""

# Fungsi untuk mengevaluasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results = {
        'Confusion Matrix': cm,
        'True Positive (TP)': tp,
        'False Positive (FP)': fp,
        'False Negative (FN)': fn,
        'True Negative (TN)': tn,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label='MACET'),
        'Recall': recall_score(y_test, y_pred, pos_label='MACET'),
        'F1-Score': f1_score(y_test, y_pred, pos_label='MACET'),
        'y_pred': y_pred  # simpan prediksi buat nanti plotting
    }
    return results


# Plot Confusion Matrix untuk setiap model (DIPISAH)
for model_name, metrics in results.items():
    cm = metrics['Confusion Matrix']
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Fungsi untuk mengevaluasi dan mengembalikan hasil sebagai kamus
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results = {
        'Confusion Matrix': cm,
        'True Positive (TP)': tp,
        'False Positive (FP)': fp,
        'False Negative (FN)': fn,
        'True Negative (TN)': tn,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label='MACET'),
        'Recall': recall_score(y_test, y_pred, pos_label='MACET'),
        'F1-Score': f1_score(y_test, y_pred, pos_label='MACET')
    }
    return results


# Mengevaluasi setiap model dan mengumpulkan hasilnya
results = {
    'K-Nearest Neighbors (KNN)': evaluate_model(knn, X_test, y_test),
    'Decision Tree (DT)': evaluate_model(dt, X_test, y_test),
    'Random Forest (RF)': evaluate_model(rf, X_test, y_test),
    'Support Vector Machine (SVM)': evaluate_model(svm, X_test, y_test),
    'Naive Bayes (NB)': evaluate_model(nb, X_test, y_test)
}

# Buat DataFrame untuk meringkas hasil
summary_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Isi DataFrame dengan hasil
rows = []
for model_name, metrics in results.items():
    rows.append({
        'Model': model_name,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score']
    })

# Konversi daftar kamus ke DataFrame
summary_df = pd.DataFrame(rows)

# Tampilkan DataFrame
print(summary_df)

"""# **Analisis Hasil Evaluasi Model**

Berdasarkan data evaluasi yang diperbarui, berikut adalah analisis untuk masing-masing model dengan mempertimbangkan metrik **Accuracy, Precision, Recall, dan F1-Score**.

---

## **1. K-Nearest Neighbors (KNN)**
- **Accuracy**: 90.07%
- **Precision**: 94.59%
- **Recall**: 92.11%
- **F1-Score**: 93.33%

**Analisis**:  
KNN menunjukkan performa yang baik dengan akurasi **90.07%**.  
Precision (**94.59%**) dan Recall (**92.11%**) menunjukkan keseimbangan cukup baik antara presisi dan sensitivitas.  
F1-Score **93.33%** mengindikasikan bahwa model ini mampu mengklasifikasikan data dengan cukup akurat.

---

## **2. Decision Tree (DT)**
- **Accuracy**: 97.35%
- **Precision**: 98.25%
- **Recall**: 98.25%
- **F1-Score**: 98.25%

**Analisis**:  
Decision Tree memiliki performa sangat baik dengan akurasi tinggi **97.35%**.  
Precision dan Recall yang sama tinggi (**98.25%**) menunjukkan model ini konsisten dalam klasifikasinya.  
F1-Score **98.25%** memperkuat keandalan model ini dalam mengklasifikasikan data.

---

## **3. Random Forest (RF)**
- **Accuracy**: 98.01%
- **Precision**: 97.44%
- **Recall**: 100.00%
- **F1-Score**: 98.70%

**Analisis**:  
Random Forest mencatat performa terbaik dengan **akurasi 98.01%**.  
Recall sempurna (**100.00%**) menunjukkan bahwa semua kasus positif berhasil terdeteksi.  
F1-Score **98.70%** mengindikasikan bahwa model ini sangat stabil dan efektif untuk klasifikasi.

---

## **4. Support Vector Machine (SVM)**
- **Accuracy**: 75.50%
- **Precision**: 75.50%
- **Recall**: 100.00%
- **F1-Score**: 86.04%

**Analisis**:  
SVM memiliki Recall tertinggi (**100.00%**) namun akurasi keseluruhan rendah (**75.50%**).  
Precision yang cukup (**75.50%**) menunjukkan banyaknya false positive.  
F1-Score **86.04%** mengindikasikan bahwa model ini tetap efektif jika Recall menjadi prioritas utama.

---

## **5. Naive Bayes (NB)**
- **Accuracy**: 64.90%
- **Precision**: 94.20%
- **Recall**: 57.02%
- **F1-Score**: 71.04%

**Analisis**:  
Naive Bayes memiliki akurasi terendah (**64.90%**) dengan Precision tinggi (**94.20%**) namun Recall rendah (**57.02%**).  
F1-Score **71.04%** menunjukkan bahwa model ini kurang seimbang dalam mendeteksi semua kasus positif.  
Model ini mungkin cocok jika lebih mementingkan Precision dibandingkan Recall.

---

# **Kesimpulan**

1. **Random Forest (RF)** memiliki performa terbaik secara keseluruhan dengan **akurasi 98.01%**, **Recall 100%**, dan **F1-Score 98.70%**.
2. **Decision Tree (DT)** juga menunjukkan performa yang sangat baik dengan **akurasi 97.35%**.
3. **KNN** cukup stabil dengan akurasi **90.07%**, sedangkan **SVM** dan **Naive Bayes** lebih lemah dari segi akurasi.

---

# **Rekomendasi**

- Jika mengutamakan **akurasi dan kestabilan klasifikasi**, maka **Random Forest (RF)** atau **Decision Tree (DT)** sangat disarankan.
- Jika memprioritaskan **Recall** (menangkap seluruh kasus positif), baik **Random Forest** maupun **SVM** bisa dipertimbangkan, namun SVM memiliki akurasi keseluruhan lebih rendah.
- **Naive Bayes** bisa dipilih bila **Precision** lebih penting, namun perlu hati-hati karena Recall-nya rendah.

## **c. Tuning Model Klasifikasi (Optional)**

Gunakan GridSearchCV, RandomizedSearchCV, atau metode lainnya untuk mencari kombinasi hyperparameter terbaik
"""

# Definisi model
dt = DecisionTreeClassifier()

# Hyperparameter yang akan diuji
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV dengan cross-validation
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit model dengan GridSearchCV
grid_search.fit(X_train, y_train)

# Menampilkan hasil terbaik
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Definisi model
rf = RandomForestClassifier()

# Hyperparameter yang akan diuji
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV dengan cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit model dengan GridSearchCV
grid_search.fit(X_train, y_train)

# Menampilkan hasil terbaik
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Definisikan model Decision Tree
dt = DecisionTreeClassifier()

# Definisikan ruang pencarian untuk Bayesian Optimization
param_space = {
    'criterion': ['gini', 'entropy'],
    'max_depth': (5, 50),  # Rentang kedalaman pohon
    'min_samples_split': (2, 10),  # Rentang jumlah sampel minimum untuk membagi node
    'min_samples_leaf': (1, 5)  # Rentang jumlah sampel minimum di setiap leaf node
}

# Inisialisasi BayesSearchCV
bayes_search = BayesSearchCV(estimator=dt, search_spaces=param_space, n_iter=32, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Lakukan pencarian hyperparameter terbaik
bayes_search.fit(X_train, y_train)

# Output hasil terbaik
print(f"Best parameters (Bayesian Optimization): {bayes_search.best_params_}")
best_dt_bayes = bayes_search.best_estimator_

# Evaluasi performa model pada test set
bayes_search_score = best_dt_bayes.score(X_test, y_test)
print(f"Accuracy after Bayesian Optimization: {bayes_search_score:.2f}")

# Definisikan ruang pencarian untuk Bayesian Optimization
param_space = {
    'n_estimators': (100, 500),
    'max_depth': (10, 50),
    'min_samples_split': (2, 10),
    'criterion': ['gini', 'entropy']
}

# Inisialisasi BayesSearchCV
bayes_search = BayesSearchCV(estimator=rf, search_spaces=param_space, n_iter=32, cv=3, n_jobs=-1, verbose=2, random_state=42)
bayes_search.fit(X_train, y_train)

# Output hasil terbaik
print(f"Best parameters (Bayesian Optimization): {bayes_search.best_params_}")
best_rf_bayes = bayes_search.best_estimator_

"""## **d. Evaluasi Model Klasifikasi setelah Tuning (Optional)**

Berikut adalah rekomendasi tahapannya.
1. Gunakan model dengan hyperparameter terbaik.
2. Hitung ulang metrik evaluasi untuk melihat apakah ada peningkatan performa.
"""

from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluasi performa model pada test set
bayes_search_score = best_dt_bayes.score(X_test, y_test)
print(f"Accuracy after Bayesian Optimization: {bayes_search_score:.2f}")

# Prediksi menggunakan model terbaik
y_pred = best_dt_bayes.predict(X_test)

# Menghitung metrik lainnya
precision1 = precision_score(y_test, y_pred, average='macro')
recall1 = recall_score(y_test, y_pred, average='macro')
f11 = f1_score(y_test, y_pred, average='macro')

print(f"Precision after Bayesian Optimization: {precision1:.2f}")
print(f"Recall after Bayesian Optimization: {recall1:.2f}")
print(f"F1-Score after Bayesian Optimization: {f11:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluasi performa model pada test set
bayes_search_score = best_rf_bayes.score(X_test, y_test)
print(f"Accuracy after Bayesian Optimization: {bayes_search_score:.2f}")

# Prediksi menggunakan model terbaik
y_pred = best_rf_bayes.predict(X_test)

# Menghitung metrik lainnya
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Precision after Bayesian Optimization: {precision:.2f}")
print(f"Recall after Bayesian Optimization: {recall:.2f}")
print(f"F1-Score after Bayesian Optimization: {f1:.2f}")

"""**SEBELUM TUNING**"""

# Daftar model yang akan dievaluasi
models = {
    "Decision Tree (DT)": DecisionTreeClassifier(max_depth=None, random_state=42),
    "Random Forest (RF)": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
}

# Fungsi untuk menampilkan learning curve
def plot_learning_curve(model, model_name, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Accuracy")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Validation Accuracy")
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Iterasi untuk semua model
for model_name, model in models.items():
    plot_learning_curve(model, model_name, X_train, y_train)

"""**SETELAH TUNNIG**"""

# Gunakan model yang sudah dituning
models = {
    "Decision Tree (DT)": best_dt_bayes,  # Model DT hasil tuning
    "Random Forest (RF)": best_rf_bayes   # Model RF hasil tuning
}


# Fungsi untuk menampilkan learning curve
def plot_learning_curve(model, model_name, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Accuracy")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Validation Accuracy")
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Iterasi untuk semua model
for model_name, model in models.items():
    plot_learning_curve(model, model_name, X_train, y_train)