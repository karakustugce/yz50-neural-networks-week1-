# YZ50 Hafta 8 — Character GPT

Bu klasör multi-head attention, feed-forward, residual bağlantı, LayerNorm,
dropout ve küçük GPT modeli için PyTorch kodu içerir. Kod ve anlatım AI yardımıyla
hazırlanmıştır. Öğrenme tesliminde sonuçları kendi ortamında üret, incele ve
videonu kendi cümlelerinle kaydet.

## Google Colab çalıştırma

GPU runtime seçip Python 3.10+ ortamında hafta-8 dizininden:

    python -m pip install torch
    python test_models.py
    python train.py --language shakespeare --all-stages --device auto
    python collect_turkish.py
    python train.py --language turkish --stage gpt --device auto

Önce hızlı deneme için komutlara --steps 100 eklenebilir. Varsayılan eğitim
5.000 adım, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.1 ve
batch_size=64'tür. T4 belleği yetmezse batch_size=32; hâlâ yetmezse
n_embd=96, n_layer=3, block_size=96 deneyip seçimi raporla. Eğitim süresi
Colab oturumuna göre değişir.

Sonuçlar results/<language>/results.json ve loss_table.csv içinde; üretilen
örnekler stage_sample.txt dosyalarındadır. Parametre sayısı, val loss ve süre
çalıştırma sırasında hesaplanır. Eğitim yapılmadan sayısal sonuç yazılmamıştır.

## Aşamalı loss tablosu

--all-stages aynı derlemeyi, karakter sözlüğünü, train/val ayrımını, optimizer
adım sayısını, öğrenme oranını ve batch seed düzenini kullanır. Her model aynı
başlangıç seed'iyle yeniden kurulur.

| Sıra | Stage | Eklenen parça |
|---:|---|---|
| 1 | bigram | Geçen haftanın karakter tabanı |
| 2 | single_head | Hafta 7 tek causal attention head'i |
| 3 | multi_head | Birden fazla küçük head |
| 4 | feedforward | Token başına MLP |
| 5 | residual | Attention ve MLP residual yolları |
| 6 | layernorm | Pre-LayerNorm |
| 7 | gpt | Birden çok Transformer bloğu |

Her ekleme loss'u mutlaka düşürmez. Model boyutu ve eğitim dinamikleri de
değişir; ölçülen değerleri olduğu gibi kaydet. Loss sabit seed ile seçilen
validation batch'lerinin ortalamasıdır, tüm validation dosyasının eksiksiz
taraması değildir.

## LayerNorm ve BatchNorm

Tensör şekli (B,T,C). LayerNorm her örneğin her token'ında son C özellik
ekseni boyunca ortalama ve varyans hesaplar. Diğer batch örneklerine bağlı
değildir, dolayısıyla running_mean/running_var saklamaz; train ve eval sırasında
aynı istatistik işlemi kullanılır. Hafta 6'daki BatchNorm1d ise batch ekseninden
gelen running istatistikleri tutar ve değerlendirmede bunları kullanır. Önceki
hafta kodundaki gerçek eksen uygulamanla bu özeti karşılaştırıp kendi cümlenle
yaz.

## Türkçe metin

collect_turkish.py Vikikaynak API'sinden Ömer Seyfettin hikâyeleri,
Mehmet Âkif Ersoy şiirleri, Namık Kemal şiirleri, Tevfik Fikret şiirleri,
Ziya Gökalp şiirleri ve Türkiye'de yayımlanan kamu malı eserler
kategorilerindeki sayfaları derler. Korpus data/turkish.txt; sayfa başlığı, URL ve karakter sayısı
data/turkish_sources.csv; toplam uzunluk ve farklı Unicode karakter sayısı
data/turkish_metadata.json dosyalarına kaydolur. Karakter tokenizer'ında
ş, ğ, İ gibi karakterlerin her biri ayrı token'dır. Yaklaşık 1 MB hedefini
denetle; bu kategoriler hedefe ulaşmıyorsa daha fazla kamu malı eser
kategorisini collect_turkish.py içindeki CATEGORIES listesine ekle.
Her kaynağın sayfa lisansını incele ve atıfları kaynakça ile koru.

Shakespeare ve Türkçe val loss değerlerini doğrudan dil veya model kalitesi
karşılaştırması olarak yorumlama. Karakter dağılımları, sözlük büyüklükleri,
metin türü ve veri uzunluğu farklıdır; loss tabanları değişir.

## Bonus 5(a): paralel head hesaplama

MultiHeadAttention bütün Q/K/V projeksiyonlarını tek lineer katmanda üretir,
head eksenine ayırır ve tek batched matmul düzeninde işler. test_models.py,
ağırlıkları bağımsız Head modüllerine kopyalayıp iki yolun attention
çıktılarını allclose ile karşılaştırır.

## Kaynaklar

- Karpathy GPT videosu (1:21:59 sonrası): https://www.youtube.com/watch?v=kCc8FmEb1nY
- Kod: https://github.com/karpathy/ng-video-lecture ve https://github.com/karpathy/nanoGPT
- Vikikaynak: https://tr.wikisource.org/
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
