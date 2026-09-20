"""YZ50 Hafta 5 icin veri, model ve gradient yardimcilari.

Model: 3 karakter -> embedding -> Linear -> BatchNorm -> tanh -> Linear.
Hafta 4 verisini kullanir; veri kopyalamak yerine tek kaynagi korur.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "hafta-4" / "data" / "names_en.txt"
RESULTS_FILE = ROOT / "results.json"
SEED = 2147483647
BLOCK_SIZE = 3
VOCAB_SIZE = 27


def load_names() -> list[str]:
    """Hafta 4 ile ayni temizlenmis Ingilizce isim listesini yukler."""
    raw = DATA_FILE.read_text(encoding="utf-8-sig").lower()
    names = {
        token
        for line in raw.splitlines()
        for token in re.split(r"[^a-z]+", line.strip())
        if token
    }
    return sorted(names)


def build_dataset(words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    stoi = {".": 0, **{ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}}
    X, Y = [], []
    for word in words:
        context = [0] * BLOCK_SIZE
        for char in word + ".":
            target = stoi[char]
            X.append(context)
            Y.append(target)
            context = context[1:] + [target]
    return torch.tensor(X), torch.tensor(Y)


def get_splits() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    words = load_names()
    random.Random(42).shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    return {
        "train": build_dataset(words[:n1]),
        "dev": build_dataset(words[n1:n2]),
        "test": build_dataset(words[n2:]),
    }


def init_parameters(n_embd: int = 10, n_hidden: int = 64, requires_grad: bool = True):
    """Yanlis turevleri saklamayacak sekilde parametreleri sifirdan farkli baslatir."""
    generator = torch.Generator().manual_seed(SEED)
    parameters = {
        "C": torch.randn((VOCAB_SIZE, n_embd), generator=generator),
        "W1": torch.randn((n_embd * BLOCK_SIZE, n_hidden), generator=generator)
        * (5 / 3)
        / (n_embd * BLOCK_SIZE) ** 0.5,
        # BatchNorm nedeniyle gereksiz olsa da turev zincirini gormek icin tutuluyor.
        "b1": torch.randn(n_hidden, generator=generator) * 0.1,
        "W2": torch.randn((n_hidden, VOCAB_SIZE), generator=generator) * 0.1,
        "b2": torch.randn(VOCAB_SIZE, generator=generator) * 0.1,
        "bngain": torch.randn((1, n_hidden), generator=generator) * 0.1 + 1.0,
        "bnbias": torch.randn((1, n_hidden), generator=generator) * 0.1,
    }
    for value in parameters.values():
        value.requires_grad = requires_grad
    return parameters


def make_batch(Xtr: torch.Tensor, Ytr: torch.Tensor, batch_size: int = 32):
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(0, Xtr.shape[0], (batch_size,), generator=generator)
    return Xtr[indices], Ytr[indices]


def detailed_forward(parameters, Xb: torch.Tensor, Yb: torch.Tensor):
    """Her turev dugumunu ayri isimle tutan ileri yayilim."""
    C, W1, b1 = parameters["C"], parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    bngain, bnbias = parameters["bngain"], parameters["bnbias"]
    n = Xb.shape[0]

    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1

    # BatchNorm'u atomik islemlere ayiriyoruz.
    bnmeani = (1 / n) * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = (1 / (n - 1)) * bndiff2.sum(0, keepdim=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2

    # Cross entropy'yi de log-sum-exp'in acik adimlarina ayiriyoruz.
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum**-1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    values = {
        "emb": emb,
        "embcat": embcat,
        "hprebn": hprebn,
        "bnmeani": bnmeani,
        "bndiff": bndiff,
        "bndiff2": bndiff2,
        "bnvar": bnvar,
        "bnvar_inv": bnvar_inv,
        "bnraw": bnraw,
        "hpreact": hpreact,
        "h": h,
        "logits": logits,
        "logit_maxes": logit_maxes,
        "norm_logits": norm_logits,
        "counts": counts,
        "counts_sum": counts_sum,
        "counts_sum_inv": counts_sum_inv,
        "probs": probs,
        "logprobs": logprobs,
    }
    return loss, values


def retain_and_backward(loss: torch.Tensor, values: dict[str, torch.Tensor], parameters):
    for parameter in parameters.values():
        parameter.grad = None
    for value in values.values():
        value.retain_grad()
    loss.backward()


def manual_backward(parameters, values, Xb: torch.Tensor, Yb: torch.Tensor):
    """Exercise 1: zincirin her dugumunden tek tek geri yayilim."""
    W1, W2 = parameters["W1"], parameters["W2"]
    bngain = parameters["bngain"]
    n = Xb.shape[0]

    logprobs, probs = values["logprobs"], values["probs"]
    counts = values["counts"]
    counts_sum, counts_sum_inv = values["counts_sum"], values["counts_sum_inv"]
    norm_logits, logits = values["norm_logits"], values["logits"]
    h = values["h"]
    bnraw, bnvar_inv, bnvar = values["bnraw"], values["bnvar_inv"], values["bnvar"]
    bndiff2, bndiff = values["bndiff2"], values["bndiff"]
    hprebn, embcat, emb = values["hprebn"], values["embcat"], values["emb"]

    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n), Yb] = -1.0 / n
    dprobs = (1.0 / probs) * dlogprobs
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
    dcounts = counts_sum_inv * dprobs
    dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
    # counts_sum = counts.sum(dim=1): broadcast'in geri donusu her satira dagilir.
    dcounts += torch.ones_like(counts) * dcounts_sum
    dnorm_logits = counts * dcounts
    dlogits = dnorm_logits.clone()
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    max_mask = F.one_hot(logits.max(1).indices, num_classes=logits.shape[1])
    dlogits += max_mask * dlogit_maxes

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dhpreact = (1.0 - h**2) * dh

    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0, keepdim=True)
    dbndiff = bnvar_inv * dbnraw
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
    dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
    dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(bndiff2) * dbnvar
    dbndiff += (2 * bndiff) * dbndiff2
    dhprebn = dbndiff.clone()
    dbnmeani = (-dbndiff).sum(0)
    # mean batch boyunca broadcast edildi; geride batch boyutu boyunca sum alinir.
    dhprebn += (1.0 / n) * torch.ones_like(hprebn) * dbnmeani

    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(parameters["C"])
    dC.index_add_(0, Xb.reshape(-1), demb.reshape(-1, demb.shape[-1]))

    return {
        "logprobs": dlogprobs,
        "probs": dprobs,
        "counts_sum_inv": dcounts_sum_inv,
        "counts_sum": dcounts_sum,
        "counts": dcounts,
        "norm_logits": dnorm_logits,
        "logit_maxes": dlogit_maxes,
        "logits": dlogits,
        "h": dh,
        "W2": dW2,
        "b2": db2,
        "hpreact": dhpreact,
        "bngain": dbngain,
        "bnbias": dbnbias,
        "bnraw": dbnraw,
        "bnvar_inv": dbnvar_inv,
        "bnvar": dbnvar,
        "bndiff2": dbndiff2,
        "bndiff": dbndiff,
        "bnmeani": dbnmeani,
        "hprebn": dhprebn,
        "embcat": dembcat,
        "W1": dW1,
        "b1": db1,
        "emb": demb,
        "C": dC,
    }


def fused_parameter_grads(parameters, values, Xb: torch.Tensor, Yb: torch.Tensor):
    """Exercise 2-4: cross entropy ve BatchNorm'u kapali formla geri yayar."""
    W1, W2 = parameters["W1"], parameters["W2"]
    bngain = parameters["bngain"]
    logits, h = values["logits"], values["h"]
    bnraw, bnvar_inv = values["bnraw"], values["bnvar_inv"]
    embcat, emb = values["embcat"], values["emb"]
    n = Xb.shape[0]

    dlogits = F.softmax(logits, dim=1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dhpreact = (1.0 - h**2) * dh
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)

    # unbiased=True oldugu icin son terimde n/(n-1) duzeltmesi vardir.
    dhprebn = (bngain * bnvar_inv / n) * (
        n * dhpreact
        - dhpreact.sum(0)
        - (n / (n - 1)) * bnraw * (dhpreact * bnraw).sum(0)
    )
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(parameters["C"])
    dC.index_add_(0, Xb.reshape(-1), demb.reshape(-1, demb.shape[-1]))

    return {
        "C": dC,
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "bngain": dbngain,
        "bnbias": dbnbias,
        "logits": dlogits,
        "hprebn": dhprebn,
    }


def comparison(name: str, manual: torch.Tensor, target: torch.Tensor) -> dict:
    autograd = target.grad
    exact = torch.equal(manual, autograd)
    approximate = torch.allclose(manual, autograd)
    maxdiff = (manual - autograd).abs().max().item()
    return {"name": name, "exact": exact, "approximate": approximate, "maxdiff": maxdiff}


def print_comparisons(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"{row['name']:15s} | exact: {str(row['exact']):5s} | "
            f"approximate: {str(row['approximate']):5s} | maxdiff: {row['maxdiff']:.3e}"
        )


def save_result(key: str, value) -> None:
    results = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    results[key] = value
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
