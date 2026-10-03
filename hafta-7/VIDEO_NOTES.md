# Hafta 7 — video hazırlık notları

Bu notlar Codex tarafından hazırlanmış bir çalışma kılavuzudur. Anlatımı kendi
cümlelerinle yap; anlamadığın bölümü çalışma tamamlandı diye anlatma. Kod ve
deneylerdeki AI katkısını açıklamak gerekir; yönerge bağımsız kod yazımı istiyor.

## 1. Kurulum ve veri (20–30 saniye)

Ekranda `README.md` ve `01_data_preview.py`:

- Karakter düzeyinde 65 token, 1.115.394 karakter.
- İlk %90 train, son %10 val. Pencere ayrım sınırını aşmıyor.
- `Let's he` → `et's hea`: her konum bir sonraki karakteri tahmin ediyor.
- Bigram `(B,T,65)` logits çıkarıyor; yalnız mevcut karakteri kullanıyor.

## 2. Attention satırını wei üzerinden anlat (60–90 saniye)

Ekranda `results/attention_heatmap.png` ve `results/attention_row.txt`:

- Satır query, sütun key konumu. Matris `(T,T)`; batch ile `(B,T,T)`.
- `To be, or not to be.` içindeki indeks 15 `o`; sonraki hedef boşluk.
- İndeks 14 `t`: **0,503401**; indeks 15 `o`: **0,433479**.
- İndeks 12 `t`: **0,032109**; indeks 13 boşluk: **0,022910**.
- Geri kalan izinli sütunların ağırlıkları küçük; satır toplamı yaklaşık 1.
- 16–19 konumları gelecekte; maske nedeniyle ağırlıklarının hepsi 0.
- Çıkış, bu ağırlıklarla **value vektörlerini** topluyor. Harf indekslerini veya
  key vektörlerini doğrudan ortalamıyor.
- Kendi token'ına bakmak yasak değil: model mevcut token'dan sonraki token'ı tahmin ediyor.

Kendi cümlelerinle açıklanması gereken soru: “Bu o, önceki t'ye yarım ağırlık
verdiğinde nasıl bir bilgi karışımı oluşuyor?” Ağırlıkları modelin anlamı
kesin olarak çözdüğünün kanıtı gibi anlatma.

## 3. Ölçekleme ve pozisyon (30–45 saniye)

Ekranda README'nin sayısal ölçekleme tablosu:

- H=64 → √H=8. `[1,2,4,8]` skorlarında en büyük softmax ağırlığı bölmeden
  **0,978755**, bölünce **0,400680**.
- Bağımsız normal q/k örneğinde skor varyansı **63,877 → 0,998**.
- Amaç skor büyüklüğünü kontrol etmek; attention öğrenmekten vazgeçmiyor.
- Token embedding içerik, position embedding pencere içindeki konum bilgisini taşıyor.

## 4. Loss karşılaştırması (30–45 saniye)

Ekranda `results/loss.png` ve `results/results.json`:

- Aynı veri ayrımı, aynı batch başlangıçları, 5.000 adım.
- Bigram val: **2,4864**; tek head val: **2,3467**.
- Fark: **0,1397 nat/token**, göreli azalma **%5,62**.
- Bu val loss sabit 100 batch'in ortalaması; eğitim minibatch loss'u değil.
- Attention önceki 32 karakterden ağırlıklı bilgi alabiliyor; bigram yalnız
  mevcut karakterden sonraki karakteri tahmin ediyor.
- Parametre sayısı da 4.225'ten 8.321'e çıkıyor; farkın tamamını yalnız
  attention'a bağlamak doğru değil. Bir seed ile ölçülmüş sonuç.

## 5. Üretim ve kapanış (20–30 saniye)

Ekranda iki `*_sample.txt` dosyası ve `models.py` içindeki `generate`:

- 400 yeni karakter üretildi. Hâlâ bozuk kelimeler var; tam GPT değil.
- Model her adımda yalnız son **block_size=32** token'ı alıyor; konum tablosu
  ve maske bu uzunluğu destekliyor. Üretilen toplam metin kaybolmuyor.
- Görev 2'nin üç yönteminin `allclose=True` olduğunu ve bonus ısı haritasını belirt.

## Kayıttan önce kendi kendine yanıtla

1. Neden target, input'tan bir karakter kaymış?
2. Maske softmax'tan önce uygulanmazsa ne bozulur?
3. `wei @ v` çıktısının şekli neden `(B,T,H)`?
4. Aynı `o` harfinin farklı konumlardaki embedding toplamları neden farklı?
5. Daha düşük loss, üretilen her cümlenin daha iyi olduğu anlamına gelir mi?

## Teslim

Repo klasörü: https://github.com/karakustugce/yz50-neural-networks-week1-/tree/main/hafta-7

Videoyu kullanıcı kaydedip ders mailine repo linkiyle birlikte gönderecek.
Son teslim ve toplantı: **4 Ekim 2026 Pazar, 19.00 (Türkiye saati)**.
Bu klasör için e-posta gönderimi yapılmadı; hafta 6'ya yeni ekleme yapılmadı.
