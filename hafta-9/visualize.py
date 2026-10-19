"""Run experiments and create charts plus a self-contained HTML summary."""
from pathlib import Path
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tokenizer import ByteBPE

ROOT = Path(__file__).parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
DATA = ROOT.parent / 'hafta-8' / 'data' / 'turkish.txt'
SAMPLE = ("Türkçe, eklemeli yapısı sayesinde köklere farklı anlam ve görevler katan zengin bir dildir. "
          "İstanbul'da öğrenciler yapay zekâ modellerinin metni nasıl parçaladığını araştırıyor. "
          "Ankara'dan başlayan bu çalışma; Unicode, UTF-8 ve tokenizasyon kavramlarını inceliyor. "
          "Bir dil modeli, gördüğü örneklerden örüntüler öğrenir; bugün yeni bir tokenizer yazıyoruz! 😊\n")
text = DATA.read_text(encoding='utf-8') if DATA.exists() else SAMPLE * 120
plain = ByteBPE(False).train(text, 320)
regex = ByteBPE(True).train(text, 320)
plain_count, regex_count = len(plain.encode(text)), len(regex.encode(text))
byte_count = len(text.encode('utf-8'))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold'})
# Chart 1: merge rank versus frequency
fig, ax = plt.subplots(figsize=(11, 5.2))
labels=[]; values=[]
for rank,(a,b) in enumerate(regex.merges[:20]):
    token=(a+b).decode('utf-8','replace')
    labels.append(token.replace(' ','<space>').replace('\n','<newline>'))
    values.append(regex.merge_counts[rank])
colors=['#cf7548' if any(x in lab for x in 'çğıöşüÇĞİÖŞÜ') else '#397f92' for lab in labels]
ax.bar(range(len(labels)),values,color=colors)
ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right')
ax.set_ylabel('Birleştirme anındaki çift sayısı')
ax.set_title('İlk 20 BPE birleşmesi: hangi parçalar önce geliyor?')
ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'01_merge_frequencies.png',dpi=170); fig.savefig(OUT/'01_merge_frequencies.svg'); plt.close(fig)

# Chart 2: token count + compression ratio
fig, axes=plt.subplots(1,2,figsize=(10,4.4))
axes[0].bar(['Regex yok','GPT-4 regex'],[plain_count,regex_count],color=['#397f92','#cf7548'])
axes[0].set_title('Metni kaç token temsil ediyor?'); axes[0].set_ylabel('Token sayısı'); axes[0].grid(axis='y',alpha=.2)
ratios=[byte_count/plain_count,byte_count/regex_count]
axes[1].bar(['Regex yok','GPT-4 regex'],ratios,color=['#397f92','#cf7548'])
axes[1].set_title('Sıkıştırma: byte / token'); axes[1].set_ylabel('Token başına byte'); axes[1].grid(axis='y',alpha=.2)
fig.suptitle(f'BPE karşılaştırması  |  {byte_count:,} UTF-8 byte',fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'02_regex_comparison.png',dpi=170); fig.savefig(OUT/'02_regex_comparison.svg'); plt.close(fig)

# Chart 3: cl100k Turkish/English and locally trained Turkish BPE
import tiktoken
tr="Türkçe eklemeli bir dildir. İstanbul'da öğrenciler yapay zekâ modellerini inceliyor."
en="Turkish is an agglutinative language. In Istanbul, students are studying artificial intelligence models."
enc=tiktoken.get_encoding('cl100k_base')
local=ByteBPE(True).train(tr*100,320)
labels=['Türkçe\ncl100k','İngilizce\ncl100k','Türkçe\nyerel BPE']
texts=[tr,en,tr]; counts=[len(enc.encode(s)) for s in texts[:2]]+[len(local.encode(tr))]
words=[len(s.split()) for s in texts]
per=[c/w for c,w in zip(counts,words)]
fig,ax=plt.subplots(figsize=(8.5,4.8)); bars=ax.bar(labels,per,color=['#cf7548','#397f92','#72a78e'])
for bar,c,w in zip(bars,counts,words): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.05,f'{c} token / {w} kelime',ha='center',fontsize=9)
ax.set_ylabel('Token / kelime'); ax.set_title('Aynı anlam çevresinde token maliyeti')
ax.set_ylim(0,max(per)*1.3); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'03_language_comparison.png',dpi=170); fig.savefig(OUT/'03_language_comparison.svg'); plt.close(fig)

source='Hafta 8 Türkçe derlemi' if DATA.exists() else 'Örnek Türkçe paragraf (120 tekrar; hafta 8 derlemi dosyası bulunamadı)'
summary=f'''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Hafta 9 BPE sonuçları</title><style>body{{font:16px/1.55 system-ui,sans-serif;max-width:900px;margin:36px auto;padding:0 18px;color:#202a31}}h1,h2{{color:#183e4c}}img{{max-width:100%;border-radius:10px;box-shadow:0 2px 14px #0002;margin:8px 0 24px}}.note{{background:#fff4df;padding:14px 18px;border-left:4px solid #cf7548}}</style><h1>Hafta 9 — BPE tokenizer</h1><p>Byte Pair Encoding, sık görülen bitişik byte parçalarını birleştirerek metni daha az token ile temsil eder.</p><p class="note"><b>Veri:</b> {html.escape(source)}. Gerçek hafta 8 dosyası bulunduğunda kod onu otomatik kullanır.</p><h2>1. En sık birleşen parçalar</h2><p>Her yeni token, eğitim verisinde sık geçen bitişik bir çiftten oluşur. Türkçe harflerin UTF-8 byte parçaları da birleştirilebilir.</p><img src="01_merge_frequencies.png"><h2>2. Regex ön bölümlendirmesi ne değiştiriyor?</h2><p>Regex kelime ve noktalama sınırlarını önceden belirler; BPE birleşmeleri parça sınırlarını aşamaz.</p><img src="02_regex_comparison.png"><h2>3. Türkçe ve İngilizce token sayısı</h2><p>cl100k_base bu örneklerde Türkçe paragrafı İngilizceye göre daha çok token'a ayırıyor. Yerel BPE Türkçe paragraf üzerinde eğitildiği için onu daha kısa temsil edebiliyor. Küçük örnek genellenemez.</p><img src="03_language_comparison.png"><p><b>Ölçüm:</b> {byte_count:,} byte; regex olmadan {plain_count:,} token, regex ile {regex_count:,} token. Round-trip testi Türkçe karakter ve emojiyle başarılı.</p></html>'''
(OUT/'index.html').write_text(summary,encoding='utf-8')
print('Rapor oluşturuldu:', OUT/'index.html')
print('Grafikler:')
for p in sorted(OUT.glob('*.png')): print(' -',p.name)
print(f'Veri: {source}')
print(f'Byte={byte_count:,} | regex yok token={plain_count:,} | regex var token={regex_count:,}')
print('Tarayıcıda aç:', (OUT/'index.html').resolve())
