"""Gorev 8 (ek, Part 3 E02): BatchNorm'u onceki Linear katmanin W ve b'sine katla.

Tahmin sirasinda BN sabit sayilarla yapilan bir dogrusal islem:
    y = gain * (x @ W1 - mean) / std + bias
      = x @ (W1 * gain / std) + (bias - gain * mean / std)
Yani yeni W1' ve b1' ile BN katmanini tamamen silebiliriz.
"""

import torch

from ortak import forward, get_splits, init_params, new_bn_stats, save_result, split_loss, train

data = get_splits("en")
Xtr, Ytr = data["train"]
Xdev, Ydev = data["dev"]

p = init_params(27, init="kaiming", batchnorm=True)
bn = new_bn_stats()
train(p, Xtr, Ytr, steps=30_000, log_every=10_000, bn_stats=bn)
# kisa egitimde running istatistik tam oturmadigi icin tum train'den hesaplayalim
with torch.no_grad():
    emb = p["C"][Xtr]
    hpre = emb.view(emb.shape[0], -1) @ p["W1"]
    bn = {"mean": hpre.mean(0, keepdim=True), "std": hpre.std(0, keepdim=True)}

with torch.no_grad():
    scale = p["bngain"] / bn["std"]  # (1, 200)
    folded = {
        "C": p["C"],
        "W1": p["W1"] * scale,  # (30, 200) * (1, 200): her noronun sutunu olceklenir
        "b1": (p["bnbias"] - p["bngain"] * bn["mean"] / bn["std"]).view(-1),
        "W2": p["W2"],
        "b2": p["b2"],
    }

    logits_bn, _ = forward(p, Xdev, bn_stats=bn, training=False)
    logits_fold, _ = forward(folded, Xdev)  # BN katmani yok
    max_diff = (logits_bn - logits_fold).abs().max().item()

dev_bn = split_loss(p, Xdev, Ydev, bn)
dev_fold = split_loss(folded, Xdev, Ydev)
print(f"logits max fark: {max_diff:.2e}")
print(f"dev loss  BN'li: {dev_bn:.6f} | katlanmis: {dev_fold:.6f}")
print("ayni mi?", torch.allclose(logits_bn, logits_fold, atol=1e-4))

save_result(
    "gorev8_bn_katlama",
    {"logits_max_fark": max_diff, "dev_bn": round(dev_bn, 6), "dev_katlanmis": round(dev_fold, 6)},
)
