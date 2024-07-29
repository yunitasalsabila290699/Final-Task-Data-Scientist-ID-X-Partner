Here’s a README template for your project:

---

# Final Project ID/X Partners Data Scientist

**Created by:** Yunita Salsabila

## Table Of Contents

1. [Business Problem Understanding](#business-problem-understanding)
2. [Initial Setup](#initial-setup)
3. [Data Understanding](#data-understanding)
4. [Data Preparation](#data-preparation)
5. [Modeling](#modeling)
6. [Conclusion & Recommendation](#conclusion-recommendation)

---

## 1. Business Problem Understanding

### **Tujuan**

- **Mengurangi Risiko Kredit Macet:** Menjaga stabilitas keuangan perusahaan dan menghindari kerugian finansial dengan mencegah pemberian pinjaman kepada pelanggan berisiko tinggi.
- **Meningkatkan Pengambilan Keputusan Kredit:** Memungkinkan perusahaan untuk membuat keputusan lebih baik terkait pemberian pinjaman, meningkatkan peluang keuntungan.
- **Meminimalkan Kerugian Perusahaan:** Mengurangi kerugian akibat pinjaman yang tidak dapat dibayar dengan akurasi dalam prediksi risiko kredit.

### **Masalah**

- **Memprediksi Risiko Kredit:** Banyak faktor memengaruhi kemampuan pelanggan untuk membayar pinjaman, seperti riwayat kredit, pendapatan, pekerjaan, dan situasi keuangan.

### **Pendekatan**

- Berkomunikasi dengan tim manajemen, risiko, dan keuangan untuk memahami kebutuhan mereka.
- Mengembangkan model prediksi risiko kredit menggunakan teknik statistik dan data yang ada.

### **Kriteria Keberhasilan**

- **Meningkatkan Akurasi Prediksi:** Model prediksi yang akurat dan data yang terkini.
- **Mengurangi Risiko Kredit Macet:** Memungkinkan keputusan kredit yang lebih baik.

### **Goal Pengolahan**

- Membuat model untuk menentukan kemampuan pembayaran kredit oleh pemohon.

### **Analytic Approach**

- Menganalisis data untuk menemukan pola-pola yang membedakan karakteristik pengguna.
- Membangun model machine learning untuk prediksi yang lebih akurat dibandingkan analisis inferensial dan deskriptif.

## 2. Initial Setup

### **Libraries Used**

```python
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import RandomOverSampler
```

## 3. Data Understanding

### **Dataset Information**

- **Source:** Kaggle - Lending Club Data (2007-2014)
- **Initial Rows & Columns:** 466,285 rows and 75 columns

### **Data Overview**

- **Sample Data:** 

```plaintext
Unnamed: 0	id	member_id	loan_amnt	funded_amnt	funded_amnt_inv	term	int_rate	installment	grade	...
0	0	1077501	1296599	5000	5000	4975.0	36 months	10.65	162.87	B	...
1	1	1077430	1314167	2500	2500	2500.0	60 months	15.27	59.83	C	...
...
```

- **Data Types & Missing Values:** 

```plaintext
Columns with missing values:
- all_util: 100% missing
- total_rev_hi_lim: 15.07% missing
...
```

## 4. Data Preparation

### **Data Cleaning**

- **Original Data:** 466,285 rows, 75 columns
- **After Removing Columns with >70% Missing Values:** 466,285 rows, 53 columns

### **Loan Status Classification**

- **Categories for Excellent and Bad Loans:** 
  - Excellent: 'Current', 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'
  - Bad: 'Charged Off', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off'

### **Visualization**

```python
plt.bar(loan_category_counts.index, loan_category_counts.values, color=colors)
plt.xlabel('Loan Category')
plt.ylabel('Count')
plt.title('Distribution of Loan Categories')
```

## 5. Modeling

### **Model Selection and Training**

- **Models Used:**
  - RandomForestClassifier
  - GradientBoostingClassifier
  - LogisticRegression
  - KNeighborsClassifier
  - DecisionTreeClassifier

- **Evaluation Metrics:**
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
  - ROC Curve and AUC

## 6. Conclusion & Recommendation

- **Model Performance:** Summary of model accuracy, strengths, and areas of improvement.
- **Recommendations:** Suggestions for improving the model and applying the results in real-world scenarios.
