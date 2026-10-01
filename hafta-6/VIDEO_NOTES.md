# Video akışı (yaklaşık 3 dakika)

1. Ekranda `layers.py`yi aç: "Hafta 4'teki matrisleri şimdi çağrılabilir
   katmanlara topladım. Eğitim döngüsü katman adlarını bilmiyor; yalnızca
   `model(X)` ve `model.parameters()` çağırıyor. `train()`/`eval()` BatchNorm'un
   minibatch ve running istatistikleri arasında geçişini sağlıyor."
2. `results.json` içindeki `en_8_wavenet.shapes_batch_2` bölümünü göster:
   "Giriş `(2,8)`, embedding `(2,8,24)`. Sekiz komşu harfi ikişer birleştirince
   `(2,4,48)`, dönüşümden sonra `(2,4,128)`. Bir daha birleştirince
   `(2,2,256)` ve `(2,2,128)`; son çift `(2,256)` oluyor. Son Linear 27
   harf için logit üretiyor."
3. `test_layers.py`deki 0/10 örneğini ve `layers.py`deki `axes` satırını göster:
   "Önce yalnız batch üzerinde `mean(0)` aldım, `(1,T,C)` çıktı. Konuma özel
   ortalamalar 0 ve 10 olduğu için bilgi sıfırlanıyor. `(0,1)` ile batch ve
   konumu beraber indirince kanal ortalaması 5 oluyor."
4. README'deki üç satırlık İngilizce tabloyu, ardından hatalı/düzeltilmiş BN
   ölçümlerini oku. "Düz 3 ve düz 8 yalnız bağlam farkı. WaveNet satırında
   kapasite de büyüdü; sonuçta tek başına mimari etkisini izole etmiyorum."
5. Türkçe örneklerden birkaçını oku. Aynı 50 bin adımda Türkçe düz MLP'nin
   3→8 bağlam değişiminde 2.1230→2.0555 sonucunu göster. Hiyerarşik model
   2.0144; hafta 4'ün 200 bin adımlık tarihsel MLP sonucu 2.1115 idi.

Kayıt sırasında gerçek `results.json` değerlerini oku; bu not bir video dosyası
veya ses kaydı değildir.
