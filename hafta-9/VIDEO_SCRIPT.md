# Hafta 9 video anlatım taslağı

> Bunu kendi sesinle anlat; ölçümlerde örnek korpus kullandığını özellikle belirt.

Merhaba, bu hafta harf tabanlı modelin önüne byte-level BPE tokenizer ekledim. Önce Unicode kod noktası ile UTF-8 byte'ını ayırdım: ASCII karakterler bir byte, ç, ğ, ı, ö, ş, ü ve İ gibi Türkçe harfler iki byte tutuyor. Byte düzeyinde başlamak her metni 256 olası byte ile temsil edebiliyor.

BPE'de ardışık byte çiftlerini sayıyorum. En sık çifti tek token yapıp aynı işlemi tekrar ediyorum. Regex olmadan örneğimde ilk birleşmelerden bazıları `ı`, `n `, `er`, `ni`, `ö`, `ler`; GPT-4 tarzı regex'i ekleyince sıra `ı`, `en`, `er`, `ö`, `ler` şeklinde değişti. Regex sözcük ve noktalama parçalarını önce ayırdığı için BPE bu parçaların sınırını geçemiyor.

Bir metni encode edip tekrar decode ettiğimde `Çağrı İstanbul'da 🎉` aynen geri geliyor. Decoder'a geçersiz UTF-8 byte'ı verirsem hata ile durmak yerine değiştirme karakteri gösteriyorum.

Regex örneğinde `Ankara'dan` üç parçaya ayrılıyor: `Ankara`, `'d`, `an`. Desen İngilizce kısaltmaları tanımak üzere yazıldığından Türkçe kesme ekini dilbilgisel olarak analiz etmiyor.

cl100k_base ile bu kısa örnekte Türkçe paragraf 10 kelimede 33 token, İngilizcesi 13 kelimede 21 token oldu. Türkçe paragraf üzerinde eğittiğim küçük tokenizer 17 token verdi. Bu örnek genel sonuç değildir: korpus, tokenizer eğitimi ve kelime seçimi sayıları değiştirir. Ayrıca hafta 8 veri dosyası repoda bulunmadığı için BPE deneyinde tekrarlanan örnek metin kullandım; gerçek derlemeyle yeniden ölçmem gerekiyor.

Bu çalışmadan çıkardığım sonuç şu: tokenizer yalnızca metni bölmüyor; modelin her adımda hangi birimi gördüğünü, bağlam penceresine ne kadar metin sığdığını ve diller arasında token maliyetinin nasıl değiştiğini belirliyor.
