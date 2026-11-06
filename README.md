# 🔍 Fake News Detection with Transformer Models

WELFake veri seti üzerinde DistilBERT, RoBERTa Base ve RoBERTa Large
modellerinin adil ve tekrar üretilebilir karşılaştırması için proje altyapısı.

## 📁 Project Structure

```
fake_new_detection/
├── server.py                      # FastAPI web server
├── index.html                     # Frontend UI
├── requirements.txt               # Python dependencies
├── src/
│   ├── prepare_data.py            # WELFake temizliği ve sabit veri bölmesi
│   ├── data.py                    # Veri yükleme/temizleme yardımcıları
│   ├── run_experiment.py          # Config tabanlı ortak eğitim betiği
│   ├── metrics.py                 # Ortak metrikler
│   └── plots.py                   # Türkçe başlıklı grafikler
├── configs/                       # Her model/deney için JSON ayarları
├── tests/                         # Küçük otomatik kontroller
└── artifacts/                     # Git'e gönderilmeyen üretilmiş çıktılar
```

## 🚀 Quick Start

```bash
# Paketleri kurun
pip install -r requirements.txt

# Ham WELFake CSV'sini değiştirmeden temiz veri ve sabit bölme oluşturun
python -m src.prepare_data --input WELFake_Dataset.csv --output-dir artifacts/prepared_data

# Örnek: DistilBERT 256 token temel karşılaştırması
python -m src.run_experiment --config configs/01_DistilBERT_256Token.json --data-dir artifacts/prepared_data --output-root artifacts/deneyler
```

> Eğitimler yerel bilgisayarda değil, Colab GPU üzerinde çalıştırılacaktır.
> Yukarıdaki komutlar daha sonra Colab hücrelerinde Linux satır sonu (`\`) ile
> kullanılacaktır.

## Veri hazırlama kuralları

- Yalnızca `WELFake_Dataset.csv` kullanılır.
- `text` alanı boş olan kayıtlar çıkarılır; başlığı boş ama metni dolu kayıtlar korunur.
- Başlıkları farklı olsa da aynı metne sahip yinelenen kayıtlar tekilleştirilir.
- Çelişen etiketli yinelenenler eğitimden çıkarılır ve raporlanır.
- Bölme `seed=42` ile yaklaşık %70 eğitim, %15 doğrulama, %15 test olarak oluşturulur; ilk 300 karakteri aynı olan metinler birlikte tutulur.
- Bölmeler arasında aynı metin veya aynı 300 karakterlik başlangıç varsa eğitim engellenir. Bu kontrol tüm yakın tekrarları yakalamayı garanti etmez.
- Tüm modeller aynı veri bölmelerini kullanır.

## Deney çıktıları

Her deney, örneğin `artifacts/deneyler/01_DistilBERT_256Token/` altında
ayrı tutulur. Klasörde kullanılan ayarlar, en iyi model, test metrikleri ve
aşağıdaki sunum grafikleri bulunur:

- Eğitim/doğrulama kaybı
- Doğruluk ve F1 eğrisi
- ROC eğrisi
- Precision-Recall eğrisi
- Karışıklık matrisi

Ham veri, model ağırlıkları ve sonuç klasörleri `.gitignore` içindedir;
GitHub'a yalnızca kaynak kodu ve config dosyaları gönderilir.
