# Hafta 9 çalıştırma sonuçları

Çalışma zamanı: Python 3.12; `regex` 2026.9.29; `tiktoken` 0.14.0. Veri kaynağı: hafta 8 klasöründe `data/turkish.txt` yoktu; `main.py` içindeki kısa Türkçe örnek metin 120 defa tekrarlandı. Bu nedenle sayılar yöntem gösterimidir, önceki hafta korpusuna ait değildir.

| Deney | Byte | Token | Byte/token |
|---|---:|---:|---:|
| BPE, regex yok | 47,880 | 24,240 | 1.975 |
| BPE, GPT-4 regex ön bölümlendirmesi | 47,880 | 24,120 | 1.985 |

Her deneyde 64 merge, yani 256 başlangıç byte'ı + 64 yeni token ile 320'lik hedef vocab kullanıldı. İlk merge'ler (regex yok) `ı`, `n `, `er`, `ni`, `ö`, `ler`, `ü`, `ar`, `de`, `eni` oldu. Regex'li ilk 10 ise `ı`, `en`, `er`, `ö`, `ler`, `ü`, ` b`, `an`, `ar`, `in`. Türkçe sık harf grupları erken birleşiyor; UTF-8'de çok byte'lı harflerin byte parçaları da merge edilebiliyor.

## Örnek encode/decode

`Çağrı İstanbul'da 🎉` → token ID'leri → `Çağrı İstanbul'da 🎉` (birebir round-trip). Decode'a tek başına geçersiz başlangıç byte'ı `0xC3` verildiğinde `�` üretildi; `errors="replace"` hatalı veriyi görünür tutup çökme riskini kaldırır.

## GPT-4 regex parçalama örneği

`Ankara'dan İstanbul'da` → `Ankara` | `'d` | `an` | ` İstanbul` | `'d` | `a`. Bu desen İngilizce contraction parçalarını tanıdığı için `'d`'yi ayırıyor; Türkçe kesme eki ve kök ilişkisini çözümlemiyor. Regex ile eğitimde merge sırası değişti çünkü çiftler artık parça sınırlarını aşamıyor.

## cl100k_base sayımı

| Metin | Kelime | Token | Token/kelime |
|---|---:|---:|---:|
| Türkçe paragraf | 10 | 33 | 3.300 |
| İngilizce paragraf | 13 | 21 | 1.615 |
| Türkçe, yerel BPE | 10 | 17 | 1.700 |

Bu sayım küçük örnek ve örnek üzerinde eğitilmiş yerel BPE içindir. Genel Türkçe/İngilizce kıyası olarak genellenmemelidir.
