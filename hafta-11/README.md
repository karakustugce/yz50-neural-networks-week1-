# Hafta 11 — GPT-2 eğitimini hızlandırmak

Bu klasör GPT-2 Small (12 katman, 12 head, 768 gizli boyut) modelini eğitmek için hafta 10'daki mimariyi kullanır. Modelin ağırlıkları bu çalışmada sıfırdan başlatılır; önceki haftadaki Hugging Face ağırlıkları yüklenmez.

## Hızlı başlatma

T4 GPU için Colab defteri: [Hafta11_Colab.ipynb](Hafta11_Colab.ipynb). Colab'da **Runtime → Change runtime type → T4 GPU** seçin.

Yerel kurulum:

```bash
pip install -r requirements.txt
python hafta11.py benchmark --device auto
python hafta11.py train --device auto --steps 300 --batch-size 4 --block-size 128 --grad-accum 4 --amp auto --compile
```

Mac'te `auto` MPS'yi seçer. T4'te `auto` CUDA'yı seçer; T4'ün FP16 desteği kullanılır. `--data kendi_metin_dosyan.txt` ile başka UTF-8 metin verilebilir. Dosya verilmezse örnek olarak Tiny Shakespeare indirilir. Metin bir defa GPT-2 tokenizer'ıyla token'lara çevrilir; her batch tüm token dizisinin farklı konumlarından rastgele pencereler alır.

## 1. Data loader, weight tying ve başlangıç değerleri

`get_batch` giriş dizisi `x` ve bir token sağa kaydırılmış hedef `y` üretir. Batch başlangıçları token akışının tamamından seçildiğinden eğitim sadece metnin başındaki bir parçaya bağlı değildir.

Token embedding'i ile çıktı katmanının ağırlığı aynı parametredir. Varsayılan `block_size=128` ve yuvarlanmış 50.304 vocab ile model **123.787.776** parametre taşır. `lm_head` ayrı bir matris olsaydı **162.421.248** parametre olurdu. Weight tying **38.633.472 parametre** kazandırır.

Linear ve embedding ağırlıkları `std=0.02` ile başlatılır. Her Transformer bloğundaki attention ve MLP residual çıkış projeksiyonlarının std değeri `0.02 / sqrt(2 * n_layer)` olur. Her blokta iki residual kolu ana akışa ekleniyor; katman sayısı büyüdükçe bu toplamların aşırı büyümemesi için residual projeksiyonlarını küçültüyoruz. Böylece derin modelin başlangıç aktivasyonları ve gradient'leri daha dengeli kalır.

## 2. Adım süresi ve hızlandırma karşılaştırması

```bash
python hafta11.py benchmark --device auto --batch-size 2 --block-size 128 --bench-warmup 3 --bench-steps 10
```

Benchmark aynı model boyutunda sırayla temel FP32, TF32, mixed precision, `torch.compile`, zorlanmış Flash Attention ve 50.304 vocab aşamalarını ölçer. Isınma/derleme adımları süre ortalamasına katılmaz. Her ölçümde bir ileri-geri geçiş, clipping ve AdamW güncellemesi bulunur. Tablo terminale; ham ölçümler `results/benchmark_results.csv` dosyasına yazılır. Her aşama için adım saniyesi ve token/s hesaplanır; en hızlı desteklenen aşama temel FP32 ile karşılaştırılır.

Donanım adımı desteklemiyorsa sonuç satırında `skipped` ve nedeni yazılır. Örneğin T4 (compute capability 7.5) TF32 ve bfloat16 desteklemez; `--amp auto` FP16'yı seçer. Flash kernel seçimi PyTorch'un gerçek backend kontrolüyle denenir; desteklenmiyorsa ölçüm başarısızlık nedeni CSV'ye yazılır. MPS üzerinde CUDA TF32/Flash adımları atlanır. Ölçülmeyen süreler için sabit veya tahmini değer koymuyoruz.

| Aşama | Sonuç |
|---|---|
| Hedef cihazda gerçek süre ve token/s | `python hafta11.py benchmark` çalıştırılınca `results/benchmark_results.csv` içine yazılır |
| Bu kodu hazırladığım çalışma ortamı | Linux CPU; PyTorch ve GPU yok, bu yüzden gerçek adım ölçümü alınamadı |

## 3. GPT-3 tarzı optimizer ayarları

Eğitim varsayılanları: AdamW `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.1`; gradient clipping limiti `1.0`; learning rate `6e-4`'ten warmup sonrası cosine ile `6e-5`'e iner. Varsayılan 300 optimizer adımının ilk 30'u warmup'tır.

Weight decay sadece `ndim >= 2` parametrelere verilir. Varsayılan padded vocab ve 128 token context'te bu **123.666.432 parametre** demektir. Bias ve LayerNorm'un tek boyutlu ağırlıkları decay almayan gruptadır. Çalışma başında gerçek model sayıları ayrıca yazdırılır.

LR eğrisi: [learning_rate_schedule.svg](learning_rate_schedule.svg). Eğitim çalışınca aynı ayarlara göre `results/learning_rate_schedule.png` de üretilir. Her optimizer adımında loss, gradient norm ve LR hem terminale yazılır hem `results/training_log.csv` içine kaydedilir.

### Gradient accumulation

Varsayılan mikro batch 4, accumulation 4'tür; optimizer her güncellemede 16 örneklik etkin batch'in gradient ortalamasını kullanır. Her mikro batch'in ortalama loss'unu accumulation sayısına bölüyoruz. Bölmezsek gradient'ler toplanır, büyüklük accumulation sayısı kadar artar; bu, aynı batch ortalama gradient'ini ve aynı etkin learning rate'i elde etmemizi engeller. Clipping accumulation tamamlandıktan sonra bir kez uygulanır.

## 4. Çıktılar ve sınırlar

```bash
# Ölçüm
python hafta11.py benchmark --device cuda

# 300 adımlık eğitim
python hafta11.py train --device cuda --steps 300 --batch-size 4 --block-size 128 --grad-accum 4 --amp auto --compile
```

Eğitim sonucu birkaç yüz adımlık bir optimizasyon alıştırmasıdır; GPT-2'nin genel dil yeteneğini ölçmez ve model checkpoint'i kaydetmez. Bu çalışma alanında CUDA/MPS erişimi olmadığı için DDP ek görevi ve Türkçe tokenizer karşılaştırması çalıştırılmadı. Performans sayıları, Colab T4 veya MPS cihazında kod çalıştırılınca oluşur.

## Kaynaklar

- Andrej Karpathy, [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- Karpathy, [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
