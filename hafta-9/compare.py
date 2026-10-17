"""Compare Turkish/English token counts with OpenAI cl100k_base and local BPE."""
import tiktoken
from tokenizer import ByteBPE

tr = "Türkçe eklemeli bir dildir. İstanbul'da öğrenciler yapay zekâ modellerini inceliyor."
en = "Turkish is an agglutinative language. In Istanbul, students are studying artificial intelligence models."
enc=tiktoken.get_encoding('cl100k_base')
local=ByteBPE(use_regex=True).train(tr*100, target_vocab_size=320)
for label,s in [('TR',tr),('EN',en)]:
    words=len(s.split()); ids=enc.encode(s)
    print(f'{label}: words={words}, cl100k tokens={len(ids)}, tokens/word={len(ids)/words:.3f}')
print(f'TR local BPE: words={len(tr.split())}, tokens={len(local.encode(tr))}, tokens/word={len(local.encode(tr))/len(tr.split()):.3f}')
print('Note: local tokenizer is trained only on repeated Turkish paragraph; comparison is illustrative, not representative.')
