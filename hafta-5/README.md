# Hafta 5 — Autograd'siz Backpropagation

Bu hafta, hafta 4'te kurduğum `embedding → Linear → BatchNorm → tanh → Linear`
isim modelinin geriye yayılımını PyTorch'a bırakmadan elle çıkardım. Önce ileri
yayılımı küçük ara değişkenlere böldüm; sonra her gradient'i zincir kuralıyla
hesaplayıp PyTorch autograd sonucu ile `cmp` mantığında karşılaştırdım.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `ortak.py` | Veri, parçalanmış forward pass, adım adım ve kapalı form backward fonksiyonları |
| `01_manual_backprop.py` | Egzersiz 1: bütün ara değişkenler ve parametreler için elle gradient |
| `02_fused_gradients.py` | Egzersiz 2–3: cross entropy ve BatchNorm'un tek ifadeli gradient'leri |
| `03_manual_training.py` | Egzersiz 4: autograd kullanmadan 200.000 adımlık eğitim |
| `test_gradients.py` | Tüm gradient karşılaştırmalarını otomatik kontrol eden testler |
| `VIDEO_NOTES.md` | Video anlatım akışı ve en zor türevin açıklaması |
| `results.json` | Çalıştırılmış deneylerin sayısal sonuçları |

Veri, tekrar yaratılmadan `../hafta-4/data/names_en.txt` dosyasından okunur.
Train/dev/test bölmesi ve seed, hafta 4 ile aynıdır.

## Çalıştırma

```bash
python3 -m pip install -r requirements.txt
python3 01_manual_backprop.py
python3 02_fused_gradients.py
python3 test_gradients.py
python3 03_manual_training.py --steps 200000
```

Hızlı bir eğitim kontrolü için:

```bash
python3 03_manual_training.py --steps 1000 --hidden 64
```

İlk iki script, herhangi bir gradient `approximate=True` olmazsa `AssertionError`
ile durur. Böylece yalnızca sonuçları yazdırmakla kalmıyor, doğruluğu da test
ediyorum.

## Egzersiz 1 — zinciri tek tek geri yürümek

Cross entropy'yi şu ara adımlara ayırdım:

```text
logits → norm_logits → counts → counts_sum → probs → logprobs → loss
```

BatchNorm'u da şu şekilde açtım:

```text
hprebn → mean → diff → diff² → variance → inverse_std → normalized → hpreact
```

Özellikle broadcasting sırasında forward'da kopyalanan boyut, backward'da
toplanır. Örneğin `b2` bütün batch'e eklendiği için:

```python
db2 = dlogits.sum(0)
```

Aynı nedenle BatchNorm `mean` gradient'i batch boyutu boyunca toplanır. Embedding
tablosunda ise aynı harf birden fazla kez kullanıldığı için bu katkılar
`index_add_` ile aynı satırda biriktirilir.

## Egzersiz 2 — cross entropy tek satırda

Softmax + negatif log likelihood zincirindeki terimler sadeleşince:

```python
dlogits = softmax(logits)
dlogits[range(n), targets] -= 1
dlogits /= n
```

Yani doğru sınıfın olasılığından 1 çıkarılıyor ve batch ortalaması alınıyor.
Her satırın gradient toplamı sıfır; bütün logit'leri aynı sabitle kaydırmanın
softmax sonucunu değiştirmemesi de burada görülüyor.

## Egzersiz 3 — BatchNorm tek ifadede

`unbiased=True` varyans için sadeleşmiş ifade:

```python
dhprebn = (bngain * bnvar_inv / n) * (
    n * dhpreact
    - dhpreact.sum(0)
    - (n / (n - 1)) * bnraw * (dhpreact * bnraw).sum(0)
)
```

Bu türev üç etkiyi aynı anda içeriyor: her örneğin doğrudan gradient'i,
ortalama üzerinden bütün batch'e yayılan etki ve varyans üzerinden yayılan
etki. En çok zorlandığım bölüm son terimdeki `n/(n-1)` katsayısıydı; sebebi
forward pass'te Bessel düzeltmeli örnek varyansı kullanılması.

## Egzersiz 4 — autograd olmadan eğitim

`03_manual_training.py` içinde parametreler `requires_grad=False` başlatılır,
eğitim `torch.no_grad()` altında çalışır ve güncellemeler yalnızca kendi
`fused_parameter_grads` fonksiyonumdan gelir. Autograd yalnızca ilk iki scriptte
doğrulama hakemi olarak kullanılır; eğitimde gradient üretmez.

## Sonuçlar

| Kontrol | Sonuç |
|---|---:|
| Adım adım doğrulanan gradient | 26 / 26 |
| Bit düzeyinde `exact` gradient | 11 / 26 |
| En büyük adım adım fark | 7.45e-09 |
| Fused cross entropy en büyük fark | 4.66e-09 |
| Fused BatchNorm en büyük fark | 1.40e-09 |
| Manuel eğitim train loss (200.000 adım) | 2.0751 |
| Manuel eğitim dev loss (200.000 adım) | 2.1212 |

Kayan nokta işlemlerinin sırası nedeniyle bazı sonuçlar bit düzeyinde `exact`
değil; tamamı `torch.allclose` ile `approximate=True`. Tam eğitim sonucu, hafta
4'te autograd ile eğitilen BatchNorm modelinin dev loss'una (2.1208) çok yakın.
Bu da yalnızca tek minibatch'teki türevleri değil, uzun eğitim boyunca yapılan
parametre güncellemelerini de doğruluyor.

## Kaynak

- Andrej Karpathy — [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [Karpathy egzersiz notebook'u](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)
