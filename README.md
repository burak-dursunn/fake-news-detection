# 🔍 Fake News Detection with Transformer Models

Aktif ilk deney: yalnızca **ISOT**, temel DistilBERT modeli, 256 token, 2 epoch.
WELFake deneyleri arşiv olarak korunur; yeni eğitime katılmaz.
Sonraki aşama AVeriTeC üzerinde kanıt arama ve kaynaklı açıklama (RAG) olacaktır.

## Güncel Colab başlangıcı

### ISOT modeliyle Combined Corpus dış testi

- Notebook: [ISOT DistilBERT Combined Corpus testi](notebooks/ISOT_DistilBERT_Combined_Corpus_Test.ipynb).
- CSV dosyalarını Drive'da `fake_news_detection/data/Combined_Corpus_Test_Klasörü` altında tutun.
- Notebook'ta Reuters çıkarılmış ISOT verisiyle eğitilmiş modelin `en_iyi_model` klasörünü seçin.
- Test yeniden eğitim yapmaz; `0 = gerçek`, `1 = sahte` eşlemesini kullanır. Metinlerdeki Reuters ifadesi çıkarılır, ayrı başlık olmadığı için boş başlıkla test edilir.
- Tüm satırlar değerlendirilir; rapor tekrarlanan metin sayısını ve 256 token sınırını aşan kayıt sayısını içerir. ISOT ile benzerlik kontrolü yapılmaz.
- Metrikler, tahminler ve grafikler Drive'da `sonuclar/Combined_Corpus_Test` altında tarihli bir klasöre kaydedilir.

- Notebook: [ISOT DistilBERT 256 Token](notebooks/ISOT_DistilBERT_256Token_Egitimi.ipynb)
- [Colab'da aç](https://colab.research.google.com/github/burak-dursunn/fake-news-detection/blob/main/notebooks/ISOT_DistilBERT_256Token_Egitimi.ipynb): kod doğrudan GitHub'dan alınır; Drive'a kod paketi yüklenmez.
- Ayrıntılı kurulum ve araştırma planı: [ISOT ve AVeriTeC planı](docs/ISOT_ve_AVeriTeC_Plani.md)
- Ortam: `requirements-colab.txt`; config: `configs/ISOT_01_DistilBERT_256Token.json`.
- Veri hazırlama: `python -m src.prepare_isot --true true.csv --fake fake.csv --output-dir artifacts/isot/prepared_seed42_v1`.
- Eğitim **Colab GPU'da**: `python -u -m src.run_experiment --config configs/ISOT_01_DistilBERT_256Token.json --data-dir /content/isot_hazir_v1 --output-root /content/drive/MyDrive/fake_news_detection/sonuclar/ISOT --skip-split-audit`.

ISOT'ta benzer haberler bölmeden önce gruplanır. Kontrol raporu ve dosya özetleri
her yüklemede doğrulanır; pahalı benzerlik taraması tekrarlanmaz. Kaynak farkı ve
yaklaşık benzerlik aramasının sınırları raporda belirtilir. Yüksek skor, haberin
gerçek dünyadaki doğruluğunun bağımsız olarak doğrulandığı anlamına gelmez.

Model yalnızca `title` ve `text` alanlarını girdi olarak alır; `label` hedef sınıftır.
`subject` ve `date` eğitim tablosuna alınmaz. Ham dosyalarda saklanırlar.

**Aşağıdaki WELFake komutları eski deneylerin yeniden üretimi içindir.**

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
