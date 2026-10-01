# YZ50 Hafta 6 — Katmanlar ve sekiz harflik hiyerarşi

Bu çalışmada hafta 4'teki İngilizce ve Türkçe isim listelerini, aynı isim
düzeyindeki sabit `%80/%10/%10` ayrımını ve aynı temizleme kurallarını kullandım.
Katmanları `torch.nn` kullanmadan `layers.py` içinde yazdım. Kayıp için
`torch.nn.functional.cross_entropy` ve türev için `loss.backward()` kullanıyorum;
bu hafta amaç, beşinci haftadaki elle türevden sonra ileri yayılımın katman
soyutlamasını ve şekillerini kurmak.

## Dosyalar ve çalıştırma

- `layers.py`: `Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `Flatten`, `Sequential`.
- `experiment.py`: konfigürasyon listesinden beş deney, tam dev seti üzerinde
  ölçüm, örnek üretimi, `results.json` ve `loss.png` çıktısı.
- `turkish_context.py`: Türkçe'de aynı düz MLP'nin 3 ve 8 harf bağlamlarını
  aynı eğitim bütçesiyle karşılaştıran ek iki koşu.
- `test_layers.py`: eşleme şekli ve BatchNorm eksenleri için küçük kontroller.
- `results.json`: **bu çalıştırmada ölçülmüş** parametre sayıları ve loss'lar.
- `VIDEO_NOTES.md`: kendi sesimle kaydedeceğim kısa anlatımın akışı.

```bash
python -m pip install -r hafta-6/requirements.txt
python hafta-6/test_layers.py
python hafta-6/experiment.py --steps 50000
python hafta-6/turkish_context.py --steps 50000
```

Eğitim CPU'da, seed `2147483647`, minibatch 32 ve tüm koşular için 50.000
adım. Öğrenme oranı ilk yarıda 0.1, ikinci yarıda 0.01. Aynı İngilizce isim
ayrımı ve aynı eğitim adımı kullanıldı. Her dev loss, model `eval()` kipindeyken
**bütün dev örneklerinin** çapraz entropisi olarak hesaplandı. Son minibatch'in
loss'u dev loss değildir. Grafik önce minibatch loss'unu blok içinde aritmetik
ortalar, sonra `log10` alır. Ayrı minibatch loglarının ortalaması değildir.

## Şekilleri nasıl okudum?

İngilizce kelime haznesi 27 sembol (`.` ve 26 harf). İki örneklik bir batch
üzerinde, embedding boyutu 24 ve gizli genişlik 128 iken:

| Aşama | Çıktı şekli | Gerekçe |
|---|---|---|
| Giriş | `(2, 8)` | Her örnekte önceki sekiz harfin indeksleri var. |
| Embedding | `(2, 8, 24)` | Her indeks 24 sayılık öğrenilen bir vektöre dönüşür. |
| İlk eşleme | `(2, 4, 48)` | Sekiz komşu harf dört çifte iner; her çift `24+24` boyutunda. |
| İlk Linear/BN/Tanh | `(2, 4, 128)` | Dört konum korunur, çift başına 128 özellik üretilir. |
| İkinci eşleme | `(2, 2, 256)` | Dört konum iki çifte iner; `128+128`. |
| İkinci Linear/BN/Tanh | `(2, 2, 128)` | İki konumun her biri 128 özellik taşır. |
| Üçüncü eşleme | `(2, 256)` | Son iki konum tek 256 boyutlu vektör olur. |
| Üçüncü Linear/BN/Tanh | `(2, 128)` | Her isim bağlamı tek gizli vektörle temsil edilir. |
| Çıkış Linear | `(2, 27)` | Sonraki harfin 27 olası logiti. |

`Flatten(2)` bir çiftin yan yana gelen özelliklerini birleştirir; harflerin
sırası değişmez. Aynı küçük dönüşüm, o seviyedeki tüm konumlara ağırlık
paylaşımıyla uygulanır. Bu, videodaki WaveNet tarzı hiyerarşik uygulamadır;
özgün ses WaveNet modelindeki tüm dilated causal convolution ayrıntılarını
yeniden gerçekleştirdiğim anlamına gelmez.

## BatchNorm hatası

Üç boyutlu girdi `(B, T, C)` iken ilk denemede `mean(0)` kullandım. Bu,
`(1, T, C)` istatistiği çıkarır: her konum farklı normalize edilir ve batch
boyutu boyunca ortalama alınır. Oysa bu modelde her kanalın tüm batch ve
konumlar üzerinden ortak ortalaması gerekir: `mean((0, 1), keepdim=True)`;
varyans da aynı `(0, 1)` eksenlerinde hesaplanır. Çıktı `(1, 1, C)` olur.

`test_layers.py` bunu küçük sayılarla gösterir: iki örnekte her konumun
değerleri sırasıyla 0 ve 10 ise doğru ortak ortalama 5, hatalı konumsal
ortalamalar 0 ve 10'dur. İlkinde normalize değerler yaklaşık -1/+1, ikincide
her konum yaklaşık sıfır olur. Hata, eğitimde 3B tensörle çalışırken sessizce
kalabilir. `BatchNorm1d` tahmin kipinde eğitim sırasında biriktirdiği running
istatistiklerini kullanır; eğitim ve tek isim üretimi aynı normalizasyon
mantığını izlemelidir.

## Karşılaştırma

| İngilizce model | Parametre | Dev loss |
|---|---:|---:|
| Bağlam 3, düz MLP | 12.097 | 2.1479 |
| Bağlam 8, düz MLP | 22.097 | 2.0718 |
| Bağlam 8, hiyerarşik model | 76.579 | **2.0198** |

Üçten sekize geçince düz MLP'ye 10.000 parametre eklendi ve dev loss
`2.1479 - 2.0718 = 0.0761` azaldı. Hatalı konumsal BatchNorm'lu
hiyerarşik model aynı 76.579 parametreyle **2.0275**, düzeltilmiş model
**2.0198** dev loss verdi; fark 0.0077. Hata dramatik bir çökme yaratmadı,
ancak aynı model ve seed'de ölçülebilir bir fark yarattı.

İlk iki satırda yalnızca bağlam uzunluğu 3'ten 8'e değişir; embedding 10,
gizli genişlik 200 ve eğitim ayarları aynıdır. Düz modelde girdi boyutu 30'dan
80'e çıktığı için ilk Linear ağırlık matrisi ve parametre sayısı büyür.
Üçüncü satırda ise **hem mimari hem kapasite** değişir: embedding 24, genişlik
128 ve üç hiyerarşik katman. O satırla ikinci satır arasındaki loss farkını
yalnızca hiyerarşiye bağlayamam. BN hata deneyi, üçüncü satırla tamamen aynı
ayarlar ve seed üzerinde yalnızca normalizasyon eksenini değiştirir.

Türkçe için sekiz bağlamlı modeli aynı kodla, hafta 4'teki 12.804 isimlik
listede yeniden eğittim. Dev loss **2.0144**. Üretilen örneklerden bazıları:
`ogan`, `devrinör`, `cankos`, `aytaş`, `gökben`, `ilbey`. Bunların hepsinin
gerçek kişi adı veya eğitim verisinde olmayan ad olduğunu iddia etmiyorum;
yalnızca modelin örneklem çıktılarıdır.

| Türkçe model, 50.000 adım | Parametre | Dev loss |
|---|---:|---:|
| Bağlam 3, düz MLP | 12.730 | 2.1230 |
| Bağlam 8, düz MLP | 22.730 | 2.0555 |
| Bağlam 8, hiyerarşik model | 77.038 | **2.0144** |

Türkçe'de yalnızca bağlamı 3'ten 8'e uzatınca düz MLP'nin dev loss'u
`2.1230 - 2.0555 = 0.0675` azaldı; 10.000 parametre eklendi. Son satırdaki
ek `0.0411` düşüşte mimari ve kapasite birlikte değişti. Hafta 4'ün üç
bağlamlı BN'li Türkçe MLP'si dev loss **2.1115** idi; o çalışma **200.000
adım** sürmüştü. Bu haftaki 50.000 adımlık hiyerarşik sonuçla tarihsel sonuç
arasındaki `0.0971` farkı yalnızca bağlam artışının etkisi değildir.

## Ek deney düzeneği

`experiment.py` konfigürasyonları bir listeden okur, her koşunun tam dev loss'unu
ve parametre sayısını kaydeder. `results.json` ara deneylerde de güncellenir;
uzun bir çalışma yarıda kesilirse tamamlanan koşuların sonuçları korunur.
