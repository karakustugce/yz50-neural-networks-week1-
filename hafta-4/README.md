# Hafta 4 — MLP Language Model, Aktivasyonlar ve BatchNorm

Bu hafta bigram tablosunu bırakıp Bengio (2003) tarzı ilk sinir ağı language
modelimi kurdum. Her harfi öğrenilen küçük bir vektöre (embedding) çevirdim,
önceki üç harfin vektörlerini yan yana koyup bir MLP'ye verdim ve model
sıradaki harfi tahmin etti. İkinci kısımda aynı modelin eğitimini içeriden
inceledim: başlangıç loss'u, tanh doyması, Kaiming init ve BatchNorm.

Kaynaklar:

- Karpathy, Building makemore Part 2: MLP
- Karpathy, Building makemore Part 3: Activations & Gradients, BatchNorm
- Bengio vd. (2003), A Neural Probabilistic Language Model
- https://github.com/karpathy/makemore

## Dosyalar

| Dosya | Görev |
|---|---|
| `ortak.py` | Veri temizliği, 3 harflik veri seti, train/dev/test bölme, model, eğitim ve örnekleme |
| `01_dataset_embedding.py` | 1 — X/Y veri seti, 27×2 embedding tablosu, indeksleme |
| `02_mlp_forward.py` | 2 — Gizli katman, logits, elle loss ve `F.cross_entropy` |
| `03_training.py` | 3 — Tek batch overfit, learning rate taraması, minibatch eğitim |
| `04_bigger_model.py` | 4 — Model büyütme, 2 boyutlu embedding çizimi, bigram karşılaştırması |
| `05_init_fix.py` | 5 — Başlangıç loss'u, tanh histogramı, Kaiming init |
| `06_batchnorm.py` | 6 — BatchNorm'lu ve BatchNorm'suz model |
| `07_turkish.py` | 7 — Türkçe isimlerle aynı model |
| `08_bn_fold.py` | 8 (ek) — BatchNorm'u Linear katmana katlama (Part 3, E02) |
| `results.json` | Bütün scriptlerin sayısal sonuçları |
| `grafikler/` | Çizimler |
| `data/` | Hafta 3'teki İngilizce ve Türkçe isim listeleri |

## Çalıştırma

```bash
python3 -m pip install -r requirements.txt
python3 01_dataset_embedding.py
python3 02_mlp_forward.py
# ... 08'e kadar sırayla
```

Train / dev / test bölmesi isim düzeyinde %80 / %10 / %10 ve sabit seed ile
yapıldı. Aynı isim iki farklı split'e düşmüyor. 200.000 adımlık eğitimlerde
learning rate ilk yarıda 0.1, ikinci yarıda 0.01.

## Sonuçlar

### Görev 1–2

- İngilizce veri 29.494 temiz isimden 211.328 örnek (`X: 3 harf`, `Y: sonraki harf`) üretti.
- `C[5]` ile `one_hot(5) @ C` aynı sonucu verdi; embedding aslında one-hot
  girişli, bias'sız ilk katman.
- Elle hesaplanan loss `17.869396`, `F.cross_entropy` `17.869398`.
- `cross_entropy`yi tercih etme sebebi: ara tensor oluşturmuyor ve büyük
  logit'te taşmıyor. Logit 100 olduğunda elle yazdığım softmax `nan` verdi,
  `cross_entropy` içerde en büyük logit'i çıkardığı için sorunsuz çalıştı.

### Görev 3

- 36 örnekli tek batch, 3.481 parametreyle 1.000 adımda **0.2254**'e indi.
  Sıfıra inmiyor çünkü `...` bağlamından sonra her isimde farklı harf geliyor.
- Learning rate taraması (`grafikler/03_lr_taramasi.png`) loss'un 10⁻¹
  civarında en düşük olduğunu gösterdi, **0.1** seçtim.
- 40.000 adım sonunda train **2.3379**, dev **2.3355**. İkisi çok yakın, yani
  model overfit değil, küçük kalıyor.

### Görev 4 — model boyutu (200.000 adım)

| Model | Parametre | Train | Dev |
|---|---:|---:|---:|
| emb 2, hidden 100 | 3.481 | 2.2566 | 2.2645 |
| emb 2, hidden 300 | 10.281 | 2.2301 | 2.2406 |
| emb 10, hidden 200 | 11.897 | 2.1473 | **2.1881** |
| Bigram (aynı bölme) | — | — | 2.4567 |

Gizli katmanı büyütmek az kazandırdı; asıl darboğaz 2 boyutlu embedding'di.
Embedding'i 10'a çıkarınca dev loss belirgin düştü.

`grafikler/04_embedding_2d.png`: sesli harfler (a, e, i, o, u) bir köşede
toplandı; b, c, d, f, k, p, t gibi ünsüzler ortada küme oluşturdu. `q` ve `g`
diğerlerinden uzakta kaldı. Model harflerin nerede kullanıldığına bakarak
benzer harfleri kendiliğinden yakına koydu.

Örnekler:

- Bigram: `careah, hlelikimri, deliyrt, chaiivin, g, d, br`
- MLP: `carmah, ambell, milia, halaysa, jazhuel, deliah, kaleigh, quint`

MLP üç harfe baktığı için isimler çok daha telaffuz edilebilir.

### Görev 5 — init

| | İlk loss | \|h\| > 0.99 | Train | Dev |
|---|---:|---:|---:|---:|
| Beklenen | 3.2958 | | | |
| Naive init | 27.9192 | %63.3 | 2.1473 | 2.1881 |
| Kaiming init | 3.3206 | %9.6 | 2.0416 | **2.1155** |

- Naive init'te logits 44'e kadar çıkıyor, model yanlış cevaba çok emin
  başlıyor. İlk adımlar loss'u sadece küçültmeye harcanıyor (hokey sopası,
  `grafikler/05_init_loss.png`). `W2 * 0.01`, `b2 * 0` ile ilk loss beklenen
  değere indi.
- `hpreact` std'si 5.50 idi, tanh'ın büyük kısmı ±1'e yığıldı
  (`grafikler/05_tanh_histogram.png`). Bu bölgede `1 - tanh²` sıfıra yakın,
  gradient geçmiyor.
- Kaiming: `W1 * (5/3) / sqrt(30)`. `hpreact` std'si 1.63'e indi, doymuş
  oran %63'ten %10'a düştü. Eğitim adımları boşa gitmediği için dev loss
  2.1881'den 2.1155'e indi.

### Görev 6 — BatchNorm

| Model | Başta doymuş | Train | Dev |
|---|---:|---:|---:|
| Kaiming, BN yok | %10.4 | 2.0416 | 2.1155 |
| Kaiming, BN var | %0.7 | 2.0705 | 2.1208 |
| Naive init, BN var | %0.7 | 2.1926 | 2.2029 |

- BN, `hpreact`'i her nöron için batch üzerinde ortalama 0, std 1 yapıyor;
  `bngain` ve `bnbias` ile model istediği ölçeğe geri dönebiliyor. BN kendi
  bias'ını eklediği için `b1`'i kaldırdım.
- Eğitimde batch istatistiği kullanıldı, aynı anda momentum 0.001 ile running
  mean/std tutuldu. Tahminde running değerler kullanıldı; tek bir isim
  üretirken 1 örneğin std'si hesaplanamayacağı için bu gerekli.
- Running mean ile bütün train setinden hesaplanan mean arasındaki en büyük
  fark 0.0286; dev loss 2.1208'e karşı 2.1205.
- **BN'in düzelttiği şey:** tanh girdisi init'ten bağımsız olarak normalize
  oluyor. Ağırlıkları hiç ölçeklemesem bile doymuş oran %0.7'de kaldı
  (`grafikler/06_bn_tanh_histogram.png`).
- Bu tek gizli katmanlı ağda Kaiming zaten iyi ayarlı olduğu için BN dev
  loss'u iyileştirmedi (2.1155 → 2.1208). Karpathy'nin de söylediği gibi
  BN'in asıl faydası derin ağlarda, her katmanın ölçeğini elle ayarlamak
  imkânsızlaştığında ortaya çıkıyor. Naive + BN modelinin geride kalmasının
  sebebi `W2`'nin hâlâ ölçeklenmemiş olması; BN sadece gizli katmanı düzeltiyor.

### Görev 7 — Türkçe

12.804 isim, 30 karakterlik vocab, 73.184 train örneği.

| Model | Train | Dev |
|---|---:|---:|
| Bigram (hafta 3, aynı bölme) | — | 2.4986 |
| MLP, BN yok | 1.9057 | 2.1444 |
| MLP, BN var | 1.9572 | **2.1115** |

En iyi modelin test loss'u **2.0978**.

- Bigram: `calakorat, devrş, alfencahursan, gülgöiğözilp, kterşeşkül, hsensuki`
- MLP: `devriyar, köyüksoy, sayyap, fulay, demirim, nazakın, cemali, duhan`

Bigram `devrş`, `kterşeşkül` gibi Türkçede olmayan ünsüz yığınları
üretirken MLP'nin isimleri Türkçe hece yapısına uyuyor. `tekin`, `cemali`,
`duhan` gerçek listede de var.

Türkçe veri İngilizcenin yarısından az olduğu için train ile dev arasındaki
fark daha büyük (BN'siz modelde 0.24). BN, batch'ten gelen gürültü sayesinde
biraz regularization etkisi yaptı ve dev loss'u 2.1444'ten 2.1115'e indirdi.

### Görev 8 (ek) — BatchNorm katlama

Tahmin sırasında BN sabit sayılarla yapılan doğrusal bir işlem, bu yüzden
önceki Linear katmana katlanabiliyor:

```
W1' = W1 * gain / std
b1' = bias - gain * mean / std
```

BN'li model ve BN katmanı silinmiş model dev setinde aynı loss'u verdi
(2.181625); logits arasındaki en büyük fark `1.9e-06`.
