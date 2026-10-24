# Hafta 10 — GPT-2'yi sıfırdan kurmak

Bu çalışma GPT-2 Small (124M) mimarisini PyTorch modülleriyle kurar; Hugging Face ağırlıklarını eşler, forward çıktısını karşılaştırır, top-k sampling yapar ve rastgele başlatılmış modelde ilk loss ile tek batch overfit kontrolü çalıştırır.

## Kurulum

Python 3.10+ önerilir. PyTorch'u cihazınıza uygun biçimde kurun. Sonra:

```bash
pip install -r requirements.txt
python hafta10.py --mode all
```

İlk çalışma Hugging Face'ten GPT-2 ağırlıklarını indirir. CPU'da model çıktısı/karşılaştırma zaman alabilir. GPU ve Apple Silicon MPS otomatik seçilir. İsterseniz `--device cpu`, `--device cuda` veya `--device mps` kullanın.

- `--mode inspect`: ağırlık şekilleri ve wpe grafiği
- `--mode compare`: HF ve kendi modelimin logits karşılaştırması
- `--mode sample`: İngilizce/Türkçe örnekler ve Türkçe token sayısı
- `--mode train --steps 300`: rastgele başlangıç loss'u ve sabit batch overfit

Grafik `results/wpe.png` dosyasına yazılır. Büyük ağırlık dosyaları repoya eklenmez.

## Mimari ve hafta 8'den farklar

GPT-2 yine decoder-only Transformer: masked self-attention, residual bağlantılar ve feed-forward katmanları. GPT-2 Small konfigürasyonu 12 blok, 12 head, 768 hidden size, 1024 context ve 50.257 token vocab'dır.

- Hafta 8'deki küçük modelden çok daha büyük katman/embedding/vocab boyutları.
- Q, K, V tek `c_attn` doğrusal katmanında birleştirilmiş hesaplanır, sonra üç parçaya ayrılır.
- Aktivasyon GELU (tanh yaklaşıklaştırması); ReLU/Tanh değil.
- GPT-2 bloklarında LayerNorm attention ve MLP'den önce (pre-norm) kullanılır.
- HF Conv1D matrisleri [girdi, çıktı] saklar; PyTorch Linear [çıktı, girdi] kullanır. Bu nedenle lineer katman ağırlıkları aktarılırken transpose edilir; bias aktarılmaz.
- Token embedding ile lm_head ağırlığı paylaşılır (weight tying); parametre sayımında aynı tensor bir kez sayılır.

Pozisyon embedding grafiği çalışınca oluşturulur. Renk örüntüsü, konum başına öğrenilmiş vektör bileşenlerini gösterir; komşu konumlar arasında yapısal benzerlik olup olmadığını grafikten gözlemleyin.

## Beklenen yorumlar

Rastgele başlangıçta her vocab token'ı yaklaşık eşit olasılıklıysa cross-entropy yaklaşık `ln(50257) = 10.8249` olur. Tek batch overfit kontrolünde model sabit örnekleri tekrar tekrar görür; loss'un keskin düşmesi, gradient ve optimizer'ın veri hedeflerini öğrenebildiğini, modelin ileri/geri geçişinin çalıştığını gösterir. Bu kontrol genel dil yeteneğini ölçmez.

Türkçe üretimin zayıf olması, GPT-2'nin eğitim verisinin ağırlıkla İngilizce olması ve tokenizer/kelime dağılımının İngilizceye daha uygun olmasıyla ilgilidir. Tokenizer ölçümü çalıştırıldığında aynı cümle için token sayısı görünür; bu, Türkçe metnin aynı bağlam penceresinde daha az karaktere sığabilmesine yol açabilir.

## İsteğe bağlı görev (b)

Ağırlık yükleme adımından sonra birkaç Türkçe/İngilizce paralel cümlenin loss'unu ölçmek için `GPT2(..., targets)` forward'ını kullanın. Türkçe loss'unun daha yüksek çıkması dil kapsama ve token parçalanmasıyla ilişkili olabilir; cümle çiftlerinin anlam ve uzunlukça benzer olmasına dikkat edin.
