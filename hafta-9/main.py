from pathlib import Path
import json, unicodedata
from tokenizer import ByteBPE

ROOT = Path(__file__).parent
DATA = ROOT.parent / 'hafta-8' / 'data' / 'turkish.txt'
# Fallback corpus is explicitly small and illustrative; replace with hafta-8 corpus when available.
SAMPLE = ("Türkçe, eklemeli yapısı sayesinde köklere farklı anlam ve görevler katan zengin bir dildir. "
          "İstanbul'da öğrenciler yapay zekâ modellerinin metni nasıl parçaladığını araştırıyor. "
          "Ankara'dan başlayan bu çalışma; Unicode, UTF-8 ve tokenizasyon kavramlarını inceliyor. "
          "Bir dil modeli, gördüğü örneklerden örüntüler öğrenir; bugün yeni bir tokenizer yazıyoruz! 😊\n")
text = DATA.read_text(encoding='utf-8') if DATA.exists() else SAMPLE * 120

sentence = "Çağrı, İstanbul'da öğrendiği Türkçe metni çözümlüyor."
print('UNICODE / UTF-8')
for ch in dict.fromkeys(sentence):
    if ch.isspace(): continue
    print(f"{ch!r}: U+{ord(ch):04X}, UTF-8={list(ch.encode('utf-8'))}")
print('Kod noktalarını vocab yapmak neden kötü? Unicode kod noktaları byte temsiliyle aynı değildir; farklı kodlamalar/byte akışlarıyla uyum sağlamaz ve çok geniş, seyrek bir sözlük doğurur. Byte tabanı her girdiyi kayıpsız temsil eder; BPE sık dizileri sonradan birleştirir.')

for use_regex in (False, True):
    tok = ByteBPE(use_regex=use_regex).train(text, target_vocab_size=320)
    ids = tok.encode(text)
    print('\nREGEX=',use_regex,'merges=',len(tok.merges),'bytes=',len(text.encode()),'tokens=',len(ids),'ratio=',round(len(text.encode())/len(ids),3))
    print('İlk 20 merge:')
    for n,(a,b) in enumerate(tok.merges[:20],1):
        print(f'{n:02d}. {a!r} + {b!r} -> {(a+b).decode("utf-8", "replace")!r}')
    print("Ankara'dan / İstanbul'da pieces:")
    print([(p, list(p.encode())) for p in tok.pattern.findall("Ankara'dan İstanbul'da")] if use_regex else ["regex kullanılmadı"])
    sample="Çağrı İstanbul'da 🎉"
    assert tok.decode(tok.encode(sample)) == sample
    print('round-trip:',tok.decode(tok.encode(sample)))
    malformed=tok.decode([0xC3]); print('invalid UTF-8 example:',repr(malformed))

# Save compact reproducibility report
report = {
 'corpus_source': str(DATA) if DATA.exists() else 'fallback sample repeated 120 times (hafta-8 data/turkish.txt not present)',
 'corpus_characters': len(text), 'corpus_bytes': len(text.encode()),
 'sample_round_trip': 'Çağrı İstanbul\'da 🎉',
 'unicode_normalization': unicodedata.is_normalized('NFC', text),
}
(ROOT/'run_metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
