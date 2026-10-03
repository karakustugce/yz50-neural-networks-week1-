"""Görev 1: tokenizer roundtrip, veri ayrımı ve x/y hizasını göster."""
import torch
from data import load_data, get_batch


def run():
    tokenizer, splits, metadata = load_data()
    text = 'Hello, world!'
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded) == text
    x, y = get_batch(splits, 'train', 8, 2, torch.Generator().manual_seed(1337))
    assert torch.equal(x[:, 1:], y[:, :-1])
    print(metadata)
    print('encode:', encoded, '\ndecode:', tokenizer.decode(encoded))
    print('x şekli:', tuple(x.shape), '\ny şekli:', tuple(y.shape))
    print('x:\n', x, '\ny:\n', y)
    print('x[0]:', repr(tokenizer.decode(x[0])), '\ny[0]:', repr(tokenizer.decode(y[0])))


if __name__ == '__main__':
    run()
