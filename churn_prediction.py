"""
Makine Öğrenmesi Ara Ödevi - Müşteri Ayrılma Tahmini

Amaç:
Müşterilerin çeşitli özelliklerini kullanarak müşterinin hizmetten ayrılıp
ayrılmayacağını (churn) tahmin eden temel sınıflandırma modelleri oluşturmak.

Bu projede:
- Veri seti pandas ile okunacaktır.
- Eksik veriler kontrol edilip işlenecektir.
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

import pandas as pd
