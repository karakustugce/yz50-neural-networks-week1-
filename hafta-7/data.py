"""Karakter tokenizer ve birbirinden ayrı train/val pencereleri."""
from pathlib import Path
import hashlib
import torch

ROOT = Path(__file__).resolve().parent


class CharacterTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, indices):
        return ''.join(self.itos[int(i)] for i in indices)


def load_data(path=ROOT / 'data' / 'input.txt'):
    text = Path(path).read_text(encoding='utf-8')
    tokenizer = CharacterTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    cut = int(0.9 * len(tokens))
    # Bir eğitim penceresi ayrım sınırını geçmez.
    splits = {'train': tokens[:cut], 'val': tokens[cut:]}
    metadata = {
        'characters': len(text), 'vocab_size': len(tokenizer.chars),
        'train_tokens': cut, 'val_tokens': len(tokens) - cut,
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    }
    return tokenizer, splits, metadata


def get_batch(splits, split, block_size, batch_size, generator=None):
    data = splits[split]
    if block_size < 1 or batch_size < 1 or len(data) <= block_size:
        raise ValueError('Pozitif boyutlar ve block_size + 1 kadar veri gerekli.')
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x, y
