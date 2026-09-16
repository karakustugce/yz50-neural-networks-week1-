# Teslim Videosu Konuşma Akışı

## 1. Giriş

## 2. Veri seti ve embedding (`01_dataset_embedding.py`)

- Her ismin başına üç nokta koyup pencereyi harf harf kaydırdım. `X` üç harfin indeksi, `Y` sıradaki harf.
- İngilizcede 211.328 örnek çıktı.
- `C` tablosu 27×2. `C[X]` tek satırda bütün veriyi (211.328, 3, 2) şekline getiriyor.
- `C[5]` ile `one_hot(5) @ C` aynı: embedding aslında one-hot girişli bir katman.

## 3. MLP ve loss (`02_mlp_forward.py`)

- Üç harfin 2'lik vektörlerini `view(-1, 6)` ile yan yana koydum. `torch.cat` da aynı sonucu veriyor ama yeni bellek açıyor.
- `tanh(emb @ W1 + b1)` → `h @ W2 + b2` → logits.
- Elle loss 17.8694, `F.cross_entropy` aynı sonucu verdi.
- Neden `cross_entropy`: daha hızlı ve logit 100 olduğunda elle yazdığım softmax `nan` verdi, `cross_entropy` vermedi. **(ekranda çıktıyı göster)**

## 4. Eğitim (`03_training.py`)

- Önce 36 örneği overfit ettim, loss 0.23'e indi. Sıfır olmamasının sebebi `...` bağlamından sonra farklı harflerin gelmesi.
- Learning rate taraması grafiği: **(`03_lr_taramasi.png` göster)** loss 10⁻¹ civarında en düşük, 0.1 seçtim.
- Veriyi %80 / %10 / %10 böldüm. Dev seti hiperparametre seçmek için, test en sonda bir kez bakmak için.
- Train 2.338, dev 2.336: yakın olmaları modelin küçük olduğunu gösteriyor.

## 5. Modeli büyütmek (`04_bigger_model.py`)

- Hidden 100 → 300: dev 2.265 → 2.241, az kazanç.
- Embedding 2 → 10: dev 2.188. Darboğaz embedding boyutuymuş.
- **(`04_embedding_2d.png` göster)** Sesli harfler bir arada, ünsüzler ortada küme, q uzakta.
- Bigram dev loss 2.457. Örnek isimleri yan yana göster: MLP'nin isimleri çok daha okunabilir.

## 6. Init sorunları (`05_init_fix.py`)

- Rastgele bir model 27 harfe eşit olasılık vermeli, loss 3.30 olmalı. Bende 27.9 çıktı çünkü logits çok büyük.
- `W2`'yi küçültünce ilk loss 3.32'ye indi. **(`05_init_loss.png` göster, hokey sopası)**
- **(`05_tanh_histogram.png` göster)** tanh değerlerinin çoğu -1 ve 1'de. Orada türev `1 - t²` sıfır, gradient geçmiyor.
- Kaiming: `W1 * (5/3) / sqrt(fan_in)`. Doymuş oran %63'ten %10'a indi, dev loss 2.188 → 2.116.

## 7. BatchNorm (`06_batchnorm.py`)

- Gizli katmandan sonra, tanh'tan önce: batch ortalamasını çıkar, std'ye böl, `bngain` ile çarp, `bnbias` ekle.
- `b1`'i kaldırdım çünkü ortalama çıkarılınca etkisi sıfırlanıyor.
- Eğitimde batch istatistiği, tahminde running mean/std. Tek isim üretirken batch yok, o yüzden running değerler şart.
- **BN neyi düzeltti:** tanh'a giren değerler init'ten bağımsız olarak normalize oluyor. Ağırlıkları hiç ölçeklemediğimde bile doymuş oran %0.7. **(`06_bn_tanh_histogram.png` göster)**
- Dev loss: BN'siz 2.116, BN'li 2.121. Tek katmanlı ağda Kaiming zaten yetiyor; BN'in asıl faydası derin ağlarda.

## 8. Türkçe (`07_turkish.py`)

- Bigram dev 2.499, MLP BN'siz 2.144, BN'li 2.112, test 2.098.
- Bigram: `devrş`, `kterşeşkül`, `gülgöiğözilp`
- MLP: `devriyar`, `köyüksoy`, `fulay`, `nazakın`, `cemali`, `duhan`
- Türkçe veri az olduğu için train–dev farkı büyük; burada BN regularization etkisiyle dev loss'u düşürdü.

## 9. Ek: BN katlama (`08_bn_fold.py`)

Tahmin anında BN sabit bir doğrusal işlem olduğu için `W1` ve `b1` içine katladım. Aynı dev loss'u (2.181625) verdi.


