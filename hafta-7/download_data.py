"""Tiny Shakespeare'in sabit sürümünü indir ve SHA-256 ile doğrula."""
import hashlib
from urllib.request import urlopen
from data import ROOT

SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt'
EXPECTED_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'


def download():
    path = ROOT / 'data' / 'input.txt'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256:
        print('Veri hazır; SHA-256 doğru.')
        return path
    with urlopen(SOURCE, timeout=30) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Veri SHA-256 eşleşmiyor; mevcut dosyaya yazılmadı.')
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    print('Tiny Shakespeare indirildi ve doğrulandı:', path)
    return path


if __name__ == '__main__':
    download()
