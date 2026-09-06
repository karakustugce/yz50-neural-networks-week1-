# Hafta 3 — İlk Language Model

Bu çalışmada bir sonraki karakteri tahmin eden bigram modelini iki farklı yolla
kurdum: önce karakter çiftlerini sayarak, sonra aynı problemi tek katmanlı bir
sinir ağı ve gradient descent ile çözerek.

## Veri

- İngilizce: Karpathy `makemore/names.txt` — 32.032 satır
- Türkçe: `ozymaxx/turkce_dil_verisi/tum_isimler.txt` — 12.923 ham satır
- Türkçe veri `I → ı`, `İ → i` dönüşümüyle küçültüldü; boş, tekrar eden ve
  29 harfli Türk alfabesine uymayan parçalar temizlendi. 12.804 benzersiz isim kaldı.

Kaynaklar:

- https://github.com/karpathy/makemore
- https://github.com/ozymaxx/turkce_dil_verisi

## Çalıştırma

```bash
python3 -m pip install -r requirements.txt
python3 bigram_language_model.py
```

Kod aşağıdaki çıktıları üretir:

- `results.json`: loss değerleri ve örnek isimler
- `bigram_english_counts.png`, `bigram_turkish_counts.png`: sayım tabloları
- `bigram_english_loss.png`, `bigram_turkish_loss.png`: eğitim loss grafikleri

## Sonuç

| Model | İngilizce NLL | Türkçe NLL |
|---|---:|---:|
| Sayım + smoothing | 2.4543 | 2.4982 |
| Sinir ağı | 2.4619 | 2.5067 |

Sinir ağının loss'u İngilizcede 3.7623'ten 2.4619'a, Türkçede 3.9438'den
2.5068'e düştü ve sayım modelinin loss'una yaklaştı.

## NLL'yi nasıl anlıyorum?

Model doğru karaktere yüksek olasılık veriyorsa log olasılığı 0'a yaklaşır.
Yanlış veya düşük olasılıklı tahminlerde log değeri daha negatif olur. Başına
eksi koyduğumuzda iyi tahmin küçük, kötü tahmin büyük loss üretir. Bütün
bigramların ortalamasını almak da farklı uzunluktaki veri setlerini karşılaştırmayı
kolaylaştırır. Smoothing ile daha önce görülmeyen bir bigramın olasılığının sıfır,
dolayısıyla log değerinin eksi sonsuz olması engellenir.

## Broadcasting notu

`counts.sum(dim=1, keepdim=True)` her satırın toplamını ayrı bir sütun olarak
tutar. Böylece her bigram satırı kendi toplamına bölünür ve satırların toplamı
1 olur. `keepdim=True` kullanılmazsa boyutlar yanlış hizalanıp kod hata vermeden
yanlış olasılıklar üretebilir.
