"""
Makine Öğrenmesi Ara Ödevi - Müşteri Ayrılma Tahmini

Amaç:
Müşterilerin çeşitli özelliklerini kullanarak müşterinin hizmetten ayrılıp
ayrılmayacağını (churn) tahmin eden temel sınıflandırma modelleri oluşturmak.

Bu projede:
- Veri seti pandas ile okunacaktır. +
- Eksik veriler kontrol edilip işlenecektir. +
- Kategorik değişkenler sayısal forma dönüştürülecektir.
- Gerekli sayısal değişkenlere ölçekleme uygulanacaktır.
- Yeni bir öznitelik üretilecektir.
- Veri train, validation ve test kümelerine ayrılacaktır.
- Logistic Regression ve KNN modelleri eğitilecektir.
- Modeller sınıflandırma metrikleri kullanılarak değerlendirilecektir.

Kullanılan kütüphaneler:
- pandas
- scikit-learn

Çalıştırma:
pip install -r requirements.txt
python churn_prediction.py
"""

# 1. GEREKLİ KÜTÜPHANELERİN IMPORT EDİLMESİ

import pandas as pd

# Farklı sütunlara farklı ön işleme yöntemleri uygulamak için kullanılır.
from sklearn.compose import ColumnTransformer

# Eksik değerleri doldurmak için kullanılır.
from sklearn.impute import SimpleImputer

# Kullanacağımız sınıflandırma modelleri.
from sklearn.linear_model import LogisticRegression

# Model değerlendirme metrikleri.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Veriyi train-validation-test olarak bölmek için kullanılır.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Ön işleme ve modeli tek bir akışta birleştirmek için kullanılır.
from sklearn.pipeline import Pipeline

# Kategorik değişkenleri dönüştürmek ve sayısal değişkenleri ölçeklemek için.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 2. VERİ SETİNİN YÜKLENMESİ

df = pd.read_csv("musteri_churn.csv")

# 3. VERİ SETİNİN TEMEL OLARAK İNCELENMESİ
print("\n" + "=" * 60)
print("VERİ SETİNİN İLK 5 SATIRI")
print("=" * 60)

# head() varsayılan olarak ilk 5 satırı gösterir.
print(df.head())

print("\n" + "=" * 60)
print("VERİ SETİNİN BOYUTU")
print("=" * 60)

# shape bize (satır sayısı, sütun sayısı) şeklinde sonuç verir.
print("Satır ve sütun sayısı:", df.shape)

print("Satır sayısı:", df.shape[0])
print("Sütun sayısı:", df.shape[1])


print("\n" + "=" * 60)
print("VERİ SETİ BİLGİLERİ")
print("=" * 60)

# info(), sütunların veri tiplerini ve boş olmayan değer sayılarını gösterir.
df.info()


print("\n" + "=" * 60)
print("CHURN DAĞILIMI")
print("=" * 60)

# Hedef değişkenimiz olan churn sütunundaki 0 ve 1 sayılarını inceliyoruz.
print(df["churn"].value_counts())

# Yüzdesel dağılımı da görmek için normalize=True kullanıyoruz.
print("\nChurn yüzdeleri:")
print(df["churn"].value_counts(normalize=True) * 100)

# 4. EKSİK VERİLERİN KONTROL EDİLMESİ

print("\n" + "=" * 60)
print("EKSİK DEĞERLER")
print("=" * 60)

print(df.isna().sum())
"""
eksik veri olan sütunlar:
    - aylik_gelir
    - sehir
    - son_giris_gun_once

eksik değerleri doğrudan doldurmak yerine Pipeline içerisinde dolduracağım.

bunun avantajı, eksik değer doldurma yöntemlerinin yalnızca eğitim verisinden öğrenilmesidir.

böylece validation ve test verilerinden eğitim aşamasına bilgi sızması önlenmiş olur.
"""
