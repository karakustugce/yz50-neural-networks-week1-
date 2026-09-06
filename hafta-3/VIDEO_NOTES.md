# Teslim Videosu Konuşma Akışı

## 1. Giriş

Bu hafta bir harfe bakarak sonraki harfi tahmin eden bigram karakter modeli kurdum. Aynı modeli önce sayım yöntemiyle, ardından tek katmanlı bir sinir ağıyla çalıştırdım. Son olarak Türkçe karakterleri ekleyerek Türkçe isim verisi üzerinde tekrar eğittim.

## 2. Veri ve temizlik

- İngilizce veri Karpathy'nin makemore reposundaki isim listesidir.
- Türkçe veri açık kaynak `ozymaxx/turkce_dil_verisi` listesidir.
- Türkçede `I → ı` ve `İ → i` dönüşümü yaptım.
- Yalnızca 29 Türk alfabesi harfini kabul ettim; boş ve tekrar eden kayıtları çıkardım.
- 12.923 ham satırdan 12.804 benzersiz temiz isim kaldı.

## 3. Sayım modeli

`bigram_counts` fonksiyonunda her ismin başına ve sonuna nokta ekledim. Nokta başlangıç ve bitiş karakterini temsil ediyor. Yan yana gelen her iki karakteri önce dictionary içinde, sonra tensor matrisinde saydım.

İngilizcede matris 27×27, Türkçede başlangıç/bitiş işaretiyle birlikte 30×30 oldu.

## 4. Broadcasting

Her satırı kendi toplamına bölerek sayıları olasılığa çevirdim. Buradaki önemli satır:

`counts.sum(dim=1, keepdim=True)`

`keepdim=True`, satır toplamlarını sütun biçiminde tutuyor. Böylece her satır kendi toplamına bölünüyor. Kullanılmazsa tensor boyutları yanlış hizalanabilir ve kod hata vermeden yanlış model üretebilir.

## 5. NLL ve smoothing

Doğru karakterin olasılığının logaritmasını aldım. İyi tahminlerde bu değer sıfıra yaklaşır; düşük olasılıklarda daha negatif olur. Başına eksi koyunca iyi model küçük, kötü model büyük loss üretir. Ortalama almak farklı uzunluktaki verileri karşılaştırmayı sağlar.

Sayım matrisine bir sahte sayım ekledim. Bu smoothing işlemi, hiç görülmeyen bigramların olasılığının sıfır ve log değerinin eksi sonsuz olmasını engelledi.

## 6. Sinir ağı modeli

Karakterleri one-hot vektörlere çevirdim. Bu vektörleri weight matrisiyle çarpınca logits oluştu. Exponential ve satır toplamına bölme işlemi softmax görevi görerek logits değerlerini olasılığa çevirdi.

`loss.backward()` ile gradientleri hesapladım ve weight matrisini gradientin ters yönünde güncelledim. Bu, geçen hafta kendi micrograd uygulamamda yazdığım `backward()` mekanizmasının PyTorch tensorlarıyla çalışan hâlidir.

## 7. Sonuçlar

| Model | İngilizce NLL | Türkçe NLL |
|---|---:|---:|
| Sayım modeli | 2.4543 | 2.4982 |
| Sinir ağı | 2.4619 | 2.5067 |

İngilizce sinir ağı loss'u 3.7623'ten 2.4619'a; Türkçe loss ise 3.9438'den 2.5068'e düştü. İki sinir ağı da sayım modelinin loss'una yaklaştı. Bu sonuç, iki farklı yöntemle aslında aynı bigram ilişkisini öğrendiğimizi gösteriyor.

Türkçe örnek çıktılar arasında `ağdin`, `mineha`, `tun`, `me` ve `özoğbe` yer aldı. Bazı sonuçların anlamsız veya çok uzun olması beklenen bir durum; çünkü bigram modeli yalnızca bir önceki harfe bakıyor ve kelimenin daha geniş bağlamını bilmiyor.

## 8. Kapanış

En çok broadcasting boyutlarını ve Türkçe karakter dönüşümünü doğru kurarken zorlandım. Bu çalışmayla language model fikrinin temelinde bir sonraki tokenın olasılığını tahmin etmenin bulunduğunu daha net anladım.
