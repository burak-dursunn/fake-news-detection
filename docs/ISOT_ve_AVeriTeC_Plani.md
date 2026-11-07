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
3. Yerelde hazırlanan üç dosyayı `fake_news_detection/data/ISOT/hazir_v1/` içine yükleyebilirsiniz: `isot_temiz.csv`, `isot_split_seed42.csv`, `isot_hazirlama_raporu.json`. Bunlar varsa tekrar temizleme yapılmaz. Dosya özetleri ve grup üyelikleri doğrulanır.
4. Bu dosyalar yoksa notebook ISOT'u Kaggle'dan indirir, ham kopyaları Drive'a kaydeder; bir kez hazırlayıp üç çıktı dosyasını da Drive'a yazar. Kısmi kopya varsa sessizce devam etmez.
5. Hücreleri sırayla çalıştırın. Çıktılar Drive'da `sonuclar/ISOT/ISOT_01_DistilBERT_256Token/`, checkpointler `checkpointler/ISOT/ISOT_01_DistilBERT_256Token/` altında tutulur.

Çalışma zamanı yeniden başladığında kurulum ve veri kopyalama hücreleri yeniden çalıştırılmalıdır. Notebook daha önce oluşmuş deney klasörünü silmez; otomatik checkpoint devamı henüz yoktur. Kesilen eğitimi tekrar başlatmak için yeni bir deney adı seçin ve bunu raporda belirtin.

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
