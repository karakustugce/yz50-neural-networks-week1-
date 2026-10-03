"""Görev 3: head şekilleri ve ölçeklemenin sayısal etkisi."""
import json
import torch
from torch.nn import functional as F
from data import ROOT, load_data
from models import SingleHeadLanguageModel


def entropy(probs):
    return float(-(probs * probs.clamp_min(1e-30).log()).sum())


def run():
    torch.manual_seed(1337)
    tokenizer, _, _ = load_data()
    model = SingleHeadLanguageModel(len(tokenizer.chars))
    text = 'To be, or not to be.'
    idx = torch.tensor([tokenizer.encode(text)])
    wei = model.attention_weights(idx)[0]
    print('Eğitilmemiş head; yalnızca mekanizmayı gösterir.\nwei:\n', wei)
    print('Şekiller: idx', tuple(idx.shape), 'features', tuple(model.features(idx).shape), 'wei', tuple(wei.shape))
    print('5. satır (indeks 4):', [(repr(text[j]), round(float(wei[4, j]), 4)) for j in range(5)])
    # Bağımsız N(0,1) q ve k: dot product varyansı H, ölçekli varyans ~1.
    H = 64
    g = torch.Generator().manual_seed(42)
    q = torch.randn(10000, H, generator=g)
    k = torch.randn(10000, H, generator=g)
    dots = (q * k).sum(dim=-1)
    raw_scores = torch.tensor([1., 2., 4., 8.])
    raw_probs = F.softmax(raw_scores, dim=-1)
    scaled_probs = F.softmax(raw_scores / H ** 0.5, dim=-1)
    result = {
        'untrained_text': text, 'untrained_wei': wei.tolist(),
        'head_size': H, 'sqrt_head_size': H ** 0.5,
        'dot_variance_raw': float(dots.var()),
        'dot_variance_scaled': float((dots / H ** 0.5).var()),
        'illustrative_scores': raw_scores.tolist(),
        'softmax_without_scaling': raw_probs.tolist(),
        'softmax_with_scaling': scaled_probs.tolist(),
        'entropy_without_scaling': entropy(raw_probs),
        'entropy_with_scaling': entropy(scaled_probs),
    }
    print('                    bölmeden       sqrt(64)=8 ile bölerek')
    for raw, scaled in zip(raw_probs.tolist(), scaled_probs.tolist()):
        print(f'                    {raw:.6f}       {scaled:.6f}')
    print('Dot-product varyansı:', result['dot_variance_raw'], '->', result['dot_variance_scaled'])
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (out / 'attention_demo.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
