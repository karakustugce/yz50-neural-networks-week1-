"""Hafta 4 scriptlerinin ortak kullandigi veri ve yardimci fonksiyonlar."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GRAFIK = ROOT / "grafikler"
SONUCLAR = ROOT / "results.json"

EN_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
TR_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
BLOCK_SIZE = 3  # kac onceki harfe bakiyoruz
SEED = 2147483647


def turkish_lower(text: str) -> str:
    return text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()


def load_names(language: str = "en") -> list[str]:
    """Hafta 3'teki temizligin aynisi: bos, tekrarli, alfabe disi kayitlar atilir."""
    if language == "en":
        path, alphabet = DATA / "names_en.txt", EN_ALPHABET
        raw = path.read_text(encoding="utf-8-sig").lower()
    else:
        path, alphabet = DATA / "names_tr_raw.txt", TR_ALPHABET
        raw = turkish_lower(path.read_text(encoding="utf-8-sig"))
    splitter = re.compile(r"[^" + re.escape(alphabet) + r"]+")
    names = {
        token
        for line in raw.splitlines()
        for token in splitter.split(line.strip())
        if token
    }
    return sorted(names)


def vocabulary(language: str = "en") -> tuple[dict[str, int], dict[int, str]]:
    alphabet = EN_ALPHABET if language == "en" else TR_ALPHABET
    chars = [".", *alphabet]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(
    words: list[str], stoi: dict[str, int], block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """X: onceki block_size harfin indeksi, Y: siradaki harf."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size  # basta "..." ile basliyoruz
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # pencereyi bir kaydir
    return torch.tensor(X), torch.tensor(Y)


def split_words(words: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    """%80 train, %10 dev, %10 test."""
    words = words[:]
    random.seed(seed)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


def get_splits(language: str = "en", block_size: int = BLOCK_SIZE):
    words = load_names(language)
    stoi, itos = vocabulary(language)
    tr, dev, te = split_words(words)
    return {
        "words": words,
        "stoi": stoi,
        "itos": itos,
        "train": build_dataset(tr, stoi, block_size),
        "dev": build_dataset(dev, stoi, block_size),
        "test": build_dataset(te, stoi, block_size),
        "train_words": tr,
        "dev_words": dev,
    }


def bigram_split_loss(train_words, eval_words, stoi) -> float:
    """Hafta 3 sayim modeli: train'de say, baska split'te NLL hesapla."""
    V = len(stoi)
    N = torch.ones((V, V))  # +1 smoothing
    for w in train_words:
        chs = [".", *w, "."]
        for a, b in zip(chs, chs[1:]):
            N[stoi[a], stoi[b]] += 1
    P = N / N.sum(1, keepdim=True)
    total, n = 0.0, 0
    for w in eval_words:
        chs = [".", *w, "."]
        for a, b in zip(chs, chs[1:]):
            total += -torch.log(P[stoi[a], stoi[b]]).item()
            n += 1
    return total / n, P


def sample_bigram(P, itos, count, seed=SEED + 10) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        ix, name = 0, []
        while True:
            ix = torch.multinomial(P[ix], 1, generator=g).item()
            if ix == 0:
                break
            name.append(itos[ix])
        out.append("".join(name))
    return out


def save_result(key: str, value) -> None:
    """Her script kendi sonucunu results.json icine kendi anahtariyla yazar."""
    data = {}
    if SONUCLAR.exists():
        data = json.loads(SONUCLAR.read_text(encoding="utf-8"))
    data[key] = value
    SONUCLAR.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Part 3'te kullandigim model: embedding -> Linear -> (BatchNorm) -> tanh -> Linear
# Parametreleri acik acik tutuyorum ki init ve BN'in etkisi gorunsun.
# ---------------------------------------------------------------------------


def init_params(
    vocab_size: int,
    n_embd: int = 10,
    n_hidden: int = 200,
    block_size: int = BLOCK_SIZE,
    init: str = "kaiming",
    batchnorm: bool = False,
    seed: int = SEED,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    fan_in = n_embd * block_size
    p = {
        "C": torch.randn((vocab_size, n_embd), generator=g),
        "W1": torch.randn((fan_in, n_hidden), generator=g),
        "b1": torch.randn(n_hidden, generator=g),
        "W2": torch.randn((n_hidden, vocab_size), generator=g),
        "b2": torch.randn(vocab_size, generator=g),
    }
    if init == "kaiming":
        # tanh icin gain = 5/3, std = gain / sqrt(fan_in)
        p["W1"] *= (5 / 3) / fan_in**0.5
        p["b1"] *= 0.01
        # cikis katmanini kucultuyoruz ki baslangicta tahminler esit dagilsin
        p["W2"] *= 0.01
        p["b2"] *= 0
    elif init == "zero":
        for k in ("W1", "b1", "W2", "b2"):
            p[k] = torch.zeros_like(p[k])
    # init == "naive" ise hic dokunmuyoruz (videonun basindaki durum)

    if batchnorm:
        p["bngain"] = torch.ones((1, n_hidden))
        p["bnbias"] = torch.zeros((1, n_hidden))
        del p["b1"]  # BN zaten bias ekliyor, b1 bosa gidiyor
    for t in p.values():
        t.requires_grad = True
    return p


def forward(p, X, bn_stats=None, training=True, momentum=0.001):
    """Logits'i ve arada olusan ara degerleri dondurur.

    bn_stats: {"mean": ..., "std": ...} running istatistikleri. Egitimde
    guncellenir, tahminde batch istatistigi yerine bunlar kullanilir.
    """
    emb = p["C"][X]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ p["W1"]
    if "b1" in p:
        hpreact = hpreact + p["b1"]
    if "bngain" in p:
        if training:
            mean = hpreact.mean(0, keepdim=True)
            std = hpreact.std(0, keepdim=True)
            with torch.no_grad():
                bn_stats["mean"] = (1 - momentum) * bn_stats["mean"] + momentum * mean
                bn_stats["std"] = (1 - momentum) * bn_stats["std"] + momentum * std
        else:
            mean, std = bn_stats["mean"], bn_stats["std"]
        hpreact = p["bngain"] * (hpreact - mean) / std + p["bnbias"]
    h = torch.tanh(hpreact)
    logits = h @ p["W2"] + p["b2"]
    return logits, {"hpreact": hpreact, "h": h}


def train(
    p,
    Xtr,
    Ytr,
    steps: int = 200_000,
    batch_size: int = 32,
    lr_schedule=None,
    seed: int = SEED,
    log_every: int = 10_000,
    bn_stats=None,
):
    g = torch.Generator().manual_seed(seed)
    if lr_schedule is None:
        lr_schedule = lambda i: 0.1 if i < steps // 2 else 0.01  # noqa: E731
    params = list(p.values())
    lossi = []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        logits, _ = forward(p, Xtr[ix], bn_stats=bn_stats, training=True)
        loss = F.cross_entropy(logits, Ytr[ix])
        for t in params:
            t.grad = None
        loss.backward()
        lr = lr_schedule(i)
        for t in params:
            t.data += -lr * t.grad
        if log_every and i % log_every == 0:
            print(f"  {i:7d}/{steps}: {loss.item():.4f}")
        lossi.append(loss.log10().item())
    return lossi


@torch.no_grad()
def split_loss(p, X, Y, bn_stats=None) -> float:
    logits, _ = forward(p, X, bn_stats=bn_stats, training=False)
    return F.cross_entropy(logits, Y).item()


@torch.no_grad()
def sample(p, itos, count=20, block_size=BLOCK_SIZE, bn_stats=None, seed=SEED + 10):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        context = [0] * block_size
        name = []
        while True:
            logits, _ = forward(p, torch.tensor([context]), bn_stats=bn_stats, training=False)
            ix = torch.multinomial(F.softmax(logits, dim=1), 1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            name.append(itos[ix])
        out.append("".join(name))
    return out


def new_bn_stats(n_hidden: int = 200):
    return {"mean": torch.zeros((1, n_hidden)), "std": torch.ones((1, n_hidden))}


def smooth(lossi, k=1000):
    t = torch.tensor(lossi)
    n = len(t) // k * k
    return t[:n].view(-1, k).mean(1)
