# Yeni deney yolu: ISOT, ardından kanıtla doğrulama

## İlk deney

- WELFake aktif deney akışından çıkarılmıştır. Eski dosyalar ve sonuçlar arşiv niteliğinde korunur.
- Yalnızca ISOT True.csv ve Fake.csv kullanılır: 0=Gerçek, 1=Sahte.
- Önceden WELFake ile eğitilmiş ağırlıklar kullanılmaz. Başlangıç modeli Hugging Face üzerindeki `distilbert-base-uncased` olur; yalnızca Colab'a indirilir.
- Başlık + haber metni, toplam 256 token, 2 epoch; ayarlar eğitimden önce sabittir.
- `subject` (konu/kategori) ve `date` (yayın tarihi) eğitim girdisi değildir. Hazırlama kodu yalnızca `title` ve `text` okur; `label` kaynak dosyadan atanır. Ham CSV'lerde bu alanlar korunur; tarih ileride zamana göre değerlendirmede kullanılabilir.
- Boş metinler, aynı metin tekrarları ve çelişen etiketli aynı metinler çıkarılır.
- Normalize edilmiş aynı başlıklar ve aynı ilk 300 karakter birlikte gruplanır. Ayrıca 5 kelimelik parçalarla MinHash aday araması ve gerçek Jaccard >=0.85 denetimi yapılır. Bağlantılar geçişlidir: A-B ve B-C eşleşiyorsa üçü aynı gruptadır.
- Gruplar yaklaşık %70 eğitim, %15 doğrulama, %15 teste ayrılır. Oranlar grup boyutları nedeniyle tam eşit olmayabilir. Yaklaşık arama tüm anlamsal benzerlikleri yakalamaz.
- En iyi checkpoint sadece doğrulama F1 skoruyla seçilir; test, eğitim tamamlandığında bir kez ölçülür. Sonraki ayarlar bu test sonucuna göre seçilmemelidir.
- Gerçek/sahte dosyasından gelen etiketler tek tek haber doğrulaması anlamına gelmez. ISOT gerçek haberlerinin Reuters kaynaklı olması başarıyı şişirebilir. Bu deney gerçek hayat doğrulama başarısını tek başına ölçmez.

Kaynak: https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf

## Colab ve Drive

Notebook: `notebooks/ISOT_DistilBERT_256Token_Egitimi.ipynb`.

Varsayılan Drive projesi: `/content/drive/MyDrive/fake_news_detection`.

1. Yeni notebook'u Colab'da açın ve GPU çalışma zamanı seçin.
2. Kod kurulum hücresi repoyu doğrudan GitHub main dalından indirir. Mevcut temiz kopya varsa `git pull --ff-only` ile günceller; yerel düzenlemelerin üzerine yazmaz. Drive'a kod paketi yüklemeniz gerekmez. Kullanılan Git commit kimliği sonuçlarla saklanır.
3. Hazır üç dosya: `isot_temiz.csv`, `isot_split_seed42.csv`, `isot_hazirlama_raporu.json`. Notebook önce `UPLOADED_PREPARED_DIR` (varsayılan `fake_news_detection/data/raw/`), ardından `fake_news_detection/data/ISOT/hazir_v1/` ve Drive ana dizinindeki `data/raw/` klasörlerine bakar. Farklı konum için `UPLOADED_PREPARED_DIR` değişkenini düzenleyin. Seçilen klasör ekrana yazılır. Dosya özetleri ve grup üyelikleri doğrulanır; yeniden temizlik yapılmaz.
4. Hazır veri yoksa varsayılan davranış açıklayıcı hatayla durmaktır. Otomatik indirme yapılmaz. Ham veriden başlamak için `ALLOW_RAW_PREPARATION=True` seçilirse ham ISOT kullanılır/gerekirse Kaggle'dan indirilir ve temiz dosyalar Drive'a kaydedilir. Kısmi hazır dosyalar farklı klasörlerden birleştirilmez.
5. Hücreleri sırayla çalıştırın. Çıktılar Drive'da `sonuclar/ISOT/ISOT_01_DistilBERT_256Token/`, checkpointler `checkpointler/ISOT/ISOT_01_DistilBERT_256Token/` altında tutulur.

Çalışma zamanı yeniden başladığında kurulum ve veri kopyalama hücreleri yeniden çalıştırılmalıdır. Notebook daha önce oluşmuş deney klasörünü silmez; otomatik checkpoint devamı henüz yoktur. Kesilen eğitimi tekrar başlatmak için yeni bir deney adı seçin ve bunu raporda belirtin.

Paketler: `transformers==5.22.1`, `huggingface-hub>=1.32.0,<3.0`, `accelerate>=1.1,<2`. Eski Transformers 4 sabitlemesi Hub paketini düşürüp Colab Gradio/Diffusers bağımlılıklarıyla çakışıyordu. Kurulum sonrası mevcut bu paketlerin Hub gereksinimleri ve eğitim sınıflarının yüklenmesi kontrol edilir. Bu kontrol Colab'daki tüm üçüncü taraf paketler için genel bir uyumluluk garantisi değildir. Eski kurulum hücresini çalıştırdıysanız güncel notebook'u açıp çalışma zamanını yeniden başlatarak hücreleri baştan çalıştırın.

## AVeriTeC ve RAG aşaması

AVeriTeC, yalnızca sahte/gerçek haber sınıflandırmasıyla aynı görev değildir. İddia, dış kaynaklardan bulunan kanıtlarla değerlendirilir:

`İddia -> kanıt arama -> ilgili kaynak bölümlerini seçme -> iddia/kanıt karşılaştırması -> karar + kaynaklı açıklama`

- Dört kararı koruyacağız: destekleniyor, çürütülüyor, yetersiz kanıt, çelişkili kanıt.
- Resmî train/dev/test ayrımı ve seçilen sürüm kaydedilecek; yakın iddialar ayrıca denetlenecek.
- İlk teknik deneme sağlanan kanıtlarla yapılabilir. Bu, kanıt arama başarısını ölçmez; uçtan uca RAG sonucu diye sunulmayacak.
- RAG deneyinde kanıtlar arama sisteminden gelecek; test için verilmiş doğru cevaplar ve karar gerekçeleri (`justification`) arama dizinine eklenmeyecek. Etiket, doğrulama makalesinin hüküm cümlesi ve yanıtı açık eden dosya adları modele kestirme yol sağlamamalı.
- Açıklama seçilen kanıta dayanacak ve kaynak gösterecek. Kaynak yoksa model eminmiş gibi açıklama uydurmayacak.
- Başarı ayrı ölçülecek: doğru kanıtı bulma, doğru karar, açıklamanın kaynakla desteklenmesi. AVeriTeC'nin resmî değerlendirme aracı da sürümüyle birlikte kullanılacak.
- ISOT eğitimli sınıflandırıcı tek başına RAG doğrulayıcısı sayılmaz; iddia/kanıt girdisine ve dört sınıfa uyarlanmış ayrı deney gerekir.

Kaynaklar:
- https://fever.ai/dataset/averitec.html
- https://github.com/MichSchli/AVeriTeC

AVeriTeC henüz indirilmedi veya eğitime katılmadı. İlk ISOT çalışması tamamlandıktan sonra ayrı klasör ve ayrı raporla ele alınacak.
