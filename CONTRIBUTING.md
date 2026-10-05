# Katkı Rehberi

Katkıda bulunmak istediğiniz için teşekkürler!

## Kurulum

1. Repository'yi fork'layıp klonlayın.
2. Python 3.12 ile sanal ortam oluşturun ve bağımlılıkları kurun:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt pytest
   ```

## Testleri çalıştırma

```bash
pytest tests
```

## Pull request beklentileri

- `main` dalına doğrudan push yapmayın; ayrı bir dal açıp pull request gönderin.
- Pull request'i tek bir konuya odaklı tutun ve ne değiştiğini kısaca açıklayın.
- `pytest tests` yerelde geçmeli; GitHub Actions iş akışı (CI) yeşil olmalıdır.
- Model veya veri bölme ayarlarını değiştirirseniz README'deki sonuç tablolarını da güncelleyin.
- Gizli bilgi (anahtar, parola vb.) eklemeyin.
