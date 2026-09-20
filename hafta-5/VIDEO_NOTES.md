# Hafta 5 Video Anlatım Notları

## Önerilen süre: 2–3 dakika

### 0:00–0:20 — giriş

“Bu hafta, geçen hafta kurduğum MLP ve BatchNorm modelinin gradient'lerini
`loss.backward()` olmadan elle hesapladım. Autograd'ı sadece kendi sonucumu
kontrol eden bir hakem olarak kullandım.”

Ekranda `01_manual_backprop.py` ve terminaldeki `exact / approximate` tablosunu
göster.

### 0:20–0:55 — forward pass'i parçalamak

“Cross entropy'yi logits, normalize edilmiş logits, counts, probabilities ve
log probabilities adımlarına; BatchNorm'u ise mean, difference, variance,
inverse standard deviation ve normalized activation adımlarına böldüm. Böylece
zincir kuralında her okun gradient'ini ayrı ayrı görebildim.”

Ekranda `ortak.py` içindeki `detailed_forward` fonksiyonunu göster.

### 0:55–1:40 — en zor türev: BatchNorm

“En çok BatchNorm türevinde zorlandım. Çünkü tek bir örneğin girdisi yalnızca
kendi çıktısını etkilemiyor; batch ortalamasını ve varyansını değiştirerek diğer
bütün örnekleri de etkiliyor. Ayrıca forward pass'te `unbiased=True` örnek
varyansı kullandığım için sade formda `n/(n-1)` düzeltmesi geliyor. Bu katsayıyı
atınca sonuç yakın görünse bile `cmp` testi geçmiyor.”

Ekranda `fused_parameter_grads` içindeki `dhprebn` ifadesini göster.

### 1:40–2:10 — nasıl doğruladım

“Aynı minibatch ve aynı parametrelerle önce PyTorch backward çalıştırdım. Sonra
elle bulduğum her tensorü ilgili `.grad` ile karşılaştırdım. `torch.equal` bit
düzeyinde eşitliği, `torch.allclose` ise kayan nokta işlem sırası yüzünden oluşan
çok küçük farklarla yaklaşık eşitliği ölçüyor. Script, tek bir sonuç bile
yaklaşık eşit değilse hata veriyor.”

Terminalde `python3 01_manual_backprop.py` çıktısını göster.

### 2:10–2:35 — sonuç

“Son bölümde cross entropy ve BatchNorm türevlerini kapalı forma indirdim ve
modeli autograd grafiği kurmadan, yalnızca kendi gradient'lerimle eğittim. Benim
için en önemli çıkarım şu oldu: broadcasting forward'da görünmez bir kopyalama
gibi çalışıyor; backward'da ise o kopyalanan boyut mutlaka `sum` ile geri
toplanıyor.”

Ekranda `results.json` ve `03_manual_training.py` dosyalarını göster.
