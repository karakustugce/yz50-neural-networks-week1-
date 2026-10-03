"""Bonus (a): eğitilmiş modelin attention ısı haritası ve bir satırın dökümü."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from data import ROOT, CharacterTokenizer
from models import SingleHeadLanguageModel


def run():
    checkpoint = torch.load(ROOT / 'checkpoints' / 'single_head.pt', map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    tokenizer = CharacterTokenizer(''.join(checkpoint['chars']))
    text = 'To be, or not to be.'
    model = SingleHeadLanguageModel(len(tokenizer.chars), cfg['block_size'], cfg['n_embd'], cfg['head_size'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    idx = torch.tensor([tokenizer.encode(text)])
    wei = model.attention_weights(idx)[0]
    row = 15  # ikinci 'to' ifadesindeki 'o'; sonraki boşluğu tahmin eder.
    dump = {
        'text': text, 'row_index': row, 'query_character': text[row],
        'next_character': text[row + 1], 'weights': wei.tolist(),
        'row': [{'index': j, 'character': ch, 'weight': float(wei[row, j])} for j, ch in enumerate(text)],
        'row_sum': float(wei[row].sum()),
        'future_max': float(wei[row, row + 1:].max()),
    }
    out = ROOT / 'results'
    (out / 'attention_weights.json').write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['Eğitilmiş head — wei matrisi', f'Girdi: {text!r}', str(wei), '',
             f'Satır {row}: {text[row]!r}, sonraki hedef {text[row+1]!r}']
    lines.extend(f"j={item['index']:2d} token={item['character']!r:5s} weight={item['weight']:.6f}" for item in dump['row'])
    lines.append(f"Satır toplamı={dump['row_sum']:.6f}; gelecek ağırlıkları=0")
    (out / 'attention_row.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    labels = [f'{i}:{ch if ch != " " else "␣"}' for i, ch in enumerate(text)]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(wei.numpy(), cmap='magma', vmin=0, vmax=1)
    ax.set_xticks(range(len(text)), labels=labels, rotation=90)
    ax.set_yticks(range(len(text)), labels=labels)
    ax.set(xlabel='Key position (attended token)', ylabel='Query position (current token)',
           title='Trained single-head causal attention — “To be, or not to be.”')
    ax.axhline(row - 0.5, color='cyan', linewidth=0.8)
    ax.axhline(row + 0.5, color='cyan', linewidth=0.8)
    fig.colorbar(im, ax=ax, label='Attention weight')
    fig.tight_layout()
    fig.savefig(out / 'attention_heatmap.png', dpi=160)
    plt.close(fig)
    return dump


if __name__ == '__main__':
    run()
