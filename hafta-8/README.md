# YZ50 Hafta 8 — Character GPT

Bu klasör multi-head attention, feed-forward, residual bağlantı, LayerNorm,
dropout ve küçük GPT modeli için PyTorch kodu içerir. Kod ve anlatım AI yardımıyla
hazırlanmıştır. Sonuçları Colab'da üret, kodu incele ve videonu kendi cümlelerinle kaydet.

## Google Colab çalıştırma

GPU runtime seçip Python 3.10+ ortamında hafta-8 dizininden:

    python -m pip install torch
    python test_models.py

Önce hafta 7 ile aynı ayarlarda ablation/loss tablosu:

    python train.py --language shakespeare --all-stages --steps 5000 --n-embd 32 --n-head 4 --n-layer 1 --block-size 32 --batch-size 32 --learning-rate 0.003 --dropout 0

Bu ayarlar bigram ve single-head ölçümünü hafta 7'deki block_size=32,
n_embd=32, batch=32, AdamW lr=0.003 ve 5.000 adımla hizalar. Sonra büyütülmüş
GPT'yi Shakespeare'de ve Türkçe derlemede ayrı ayrı çalıştır:

    python train.py --language shakespeare --stage gpt --steps 5000 --n-embd 128 --n-head 4 --n-layer 4 --block-size 128 --batch-size 64 --learning-rate 0.0003 --dropout 0.1
    python collect_turkish.py
    python train.py --language turkish --stage gpt --steps 5000 --n-embd 128 --n-head 4 --n-layer 4 --block-size 128 --batch-size 64 --learning-rate 0.0003 --dropout 0.1 --prompt "Bir zamanlar"

İlk önce kısa duman testi için komutlara --steps 100 ekle. Varsayılan stage
ayarları n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.1 ve
batch_size=64'tür. T4 belleği yetmezse batch_size=32; hâlâ yetmezse
n_embd=96, n_layer=3, block_size=96 dene ve seçimini raporla. Eğitim süresi
Colab oturumuna göre değişir.

Ablation sonuçları results/<language>/stages/; büyütülmüş model results/<language>/gpt/
altına yazılır. Her klasörde results.json, loss_table.csv ve stage_sample.txt
çıkar. Parametre sayısı, val loss ve süre çalıştırmada ölçülür; eğitim
yapılmadan sayısal sonuç yazılmamıştır.

## Aşamalı loss tablosu

--all-stages aynı derlemeyi, karakter sözlüğünü, train/val ayrımını, optimizer
adım sayısını, öğrenme oranını ve batch seed düzenini kullanır. Her model aynı
başlangıç seed'iyle yeniden kurulur. Ablation'da dropout kapalıdır; büyütülmüş
GPT'ye dropout eklenir.

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
validation batch'lerinin ortalamasıdır, bütün validation dosyasının eksiksiz
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
kategorilerindeki sayfaları derler. Korpus data/turkish.txt; sayfa başlığı, URL
ve karakter sayısı data/turkish_sources.csv; toplam uzunluk ve farklı Unicode
karakter sayısı data/turkish_metadata.json dosyalarına kaydolur. Karakter
tokenizer'ında ş, ğ, İ gibi karakterlerin her biri ayrı token'dır. Yaklaşık 1 MB
hedefini denetle; gerçek ölçümleri raporla ve her kaynağın sayfa lisans/atıf
bilgisini kaynakça ile koru.

Shakespeare ve Türkçe val loss değerlerini doğrudan dil veya model kalitesi
karşılaştırması olarak yorumlama. Karakter dağılımları, sözlük büyüklükleri,
metin türü ve veri uzunluğu farklıdır; loss tabanları değişir.

## Bonus 5(a): paralel head hesaplama

MultiHeadAttention bütün Q/K/V projeksiyonlarını tek lineer katmanda üretir,
head eksenine ayırır ve batched matmul ile işler. test_models.py ağırlıkları
bağımsız Head modüllerine kopyalayıp iki yolun attention çıktılarını allclose
ile karşılaştırır.

## Kaynaklar

- Karpathy GPT videosu (1:21:59 sonrası): https://www.youtube.com/watch?v=kCc8FmEb1nY
- Kod: https://github.com/karpathy/ng-video-lecture ve https://github.com/karpathy/nanoGPT
- Vikikaynak: https://tr.wikisource.org/
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
