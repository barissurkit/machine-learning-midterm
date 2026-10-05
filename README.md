# Machine Learning Midterm - Customer Churn Prediction

Bu proje, Türkiye Yapay Zeka Akademisi Makine Öğrenmesi Bootcamp'i kapsamında hazırlanan ara ödevdir.

## Projenin Amacı

Bu projenin amacı, müşteri özelliklerini kullanarak müşterinin hizmetten ayrılıp ayrılmayacağını (`churn`) tahmin eden temel makine öğrenmesi modelleri geliştirmektir.

Proje kapsamında temel bir makine öğrenmesi akışı uygulanmıştır:

- Veri setinin incelenmesi
- Eksik değerlerin kontrol edilmesi
- Öznitelik üretme
- Kategorik değişkenlerin One-Hot Encoding ile dönüştürülmesi
- Sayısal değişkenlerin ölçeklenmesi
- Train, validation ve test ayrımı
- Logistic Regression ve KNN modellerinin eğitilmesi
- Modellerin validation verisi üzerinde karşılaştırılması
- Seçilen modelin test verisi üzerinde değerlendirilmesi

## Veri Seti

Projede müşteri ayrılma tahmini için oluşturulmuş 200 satırlık sentetik bir veri seti kullanılmıştır.

Veri setindeki bazı özellikler:

- Yaş
- Aylık gelir
- Abonelik süresi
- Destek talebi sayısı
- Şehir
- Üyelik tipi
- Aylık ücret
- Son giriş zamanı
- Otomatik ödeme durumu

Hedef değişken:

- `0`: Müşteri hizmeti kullanmaya devam ediyor
- `1`: Müşteri hizmetten ayrılıyor

Veri setinde bazı eksik değerler bulunmaktadır. Sayısal eksik değerler medyan ile, kategorik eksik değerler ise `Bilinmiyor` kategorisi ile doldurulmuştur.

## Feature Engineering

Yeni öznitelik olarak `destek_talebi_var_mi` değişkeni oluşturulmuştur.

- `0`: Müşterinin destek talebi yok
- `1`: Müşterinin en az bir destek talebi var

## Veri Bölme

Veri seti stratified sampling kullanılarak aşağıdaki şekilde ayrılmıştır:

- Train: %70
- Validation: %15
- Test: %15

## Kullanılan Modeller

Projede iki sınıflandırma modeli eğitilmiştir:

- Logistic Regression
- K-Nearest Neighbors (KNN)

## Validation Sonuçları

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7000 | 0.6000 | 0.5455 | 0.5714 |
| KNN | 0.7667 | 0.7500 | 0.5455 | 0.6316 |

Validation F1-score değerine göre en başarılı model **KNN** olmuştur.

## Test Sonuçları

Seçilen KNN modelinin test sonuçları:

- Accuracy: `0.7333`
- Precision: `0.7143`
- Recall: `0.4545`
- F1-score: `0.5556`

Confusion Matrix:

```text
[[17  2]
 [ 6  5]]
```

## Teknolojiler

- Python
- pandas
- scikit-learn (Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder, LogisticRegression, KNeighborsClassifier)
- pytest (testler)

## Gereksinimler

- Python 3.12 (bu sürümle doğrulandı; güncel `numpy` Python 3.12 ve üzerini gerektirir)
- `requirements.txt` içindeki paketler: `pandas`, `scikit-learn` (sürümler sabitlenmemiştir; doğrulama pandas 3.0.6 ve scikit-learn 1.9.1 ile yapılmıştır)

## Kurulum

```bash
git clone https://github.com/barissurkit/machine-learning-midterm.git
cd machine-learning-midterm
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

Betiği repository'nin kök dizininden çalıştırın (`musteri_churn.csv` dosyasını çalışma dizininden okur):

```bash
python churn_prediction.py
```

Çıktının sonunda şuna benzer test sonuçları yazdırılır:

```text
TEST SONUÇLARI
Model: KNN
Accuracy : 0.7333
Precision: 0.7143
Recall   : 0.4545
F1-score : 0.5556

Confusion Matrix:
[[17  2]
 [ 6  5]]
```

## Testler

```bash
pip install pytest
pytest tests
```

Testler veri setinin yapısını (sütunlar, 200 satır, ikili hedef değişken, eksik değer sütunları) ve betiğin uçtan uca çalışmasını (70/15/15 bölme boyutları, iki modelin karşılaştırılması, test metrikleri ve confusion matrix) doğrular.

## Katkı

Katkı rehberi için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.

## Lisans

[MIT](LICENSE)
