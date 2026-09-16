"""Gorev 7: ayni MLP'yi hafta 3'teki Turkce isim listesiyle egit, bigram ile karsilastir."""

import math

import matplotlib.pyplot as plt

from ortak import (
    GRAFIK,
    bigram_split_loss,
    get_splits,
    init_params,
    new_bn_stats,
    sample,
    sample_bigram,
    save_result,
    smooth,
    split_loss,
    train,
)

data = get_splits("tr")
Xtr, Ytr = data["train"]
Xdev, Ydev = data["dev"]
Xte, Yte = data["test"]
itos, stoi = data["itos"], data["stoi"]
V = len(stoi)  # 29 harf + "."
print(f"Turkce isim: {len(data['words'])} | vocab {V}")
print(f"train {Xtr.shape[0]} | dev {Xdev.shape[0]} | test {Xte.shape[0]}")

bigram_dev, P = bigram_split_loss(data["train_words"], data["dev_words"], stoi)
print(f"bigram dev loss: {bigram_dev:.4f}")

STEPS = 200_000
out, curves, models = {}, {}, {}
for name, bn_on in [("mlp_bn_yok", False), ("mlp_bn_var", True)]:
    print(f"\n== {name} ==")
    p = init_params(V, init="kaiming", batchnorm=bn_on)
    bn = new_bn_stats() if bn_on else None
    curves[name] = train(p, Xtr, Ytr, steps=STEPS, log_every=50_000, bn_stats=bn)
    tr, dev = split_loss(p, Xtr, Ytr, bn), split_loss(p, Xdev, Ydev, bn)
    print(f"  train {tr:.4f} | dev {dev:.4f}")
    out[name] = {"train_loss": round(tr, 4), "dev_loss": round(dev, 4)}
    models[name] = (p, bn)

# en iyi dev loss'lu modelle test'e bir kez bakiyoruz
best = min(out, key=lambda k: out[k]["dev_loss"])
p, bn = models[best]
test = split_loss(p, Xte, Yte, bn)
print(f"\nen iyi model {best} | test loss {test:.4f}")

mlp_samples = sample(p, itos, 20, bn_stats=bn)
bigram_samples = sample_bigram(P, itos, 20)
# egitim setinde birebir olan isimler gercek isim, digerleri yeni uretilmis
known = set(data["words"])
print("\nbigram:", bigram_samples)
print("MLP   :", mlp_samples)
print("MLP orneklerinden gercek listede olanlar:", [s for s in mlp_samples if s in known])

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, c in curves.items():
    ax.plot(smooth(c), label=name)
ax.axhline(math.log10(bigram_dev), color="gray", linestyle="--", label="bigram dev")
ax.set_xlabel("adim (x1000)")
ax.set_ylabel("log10(loss)")
ax.set_title("Turkce isimler: MLP egitim loss'u")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAFIK / "07_turkce_loss.png", dpi=150)
plt.close(fig)

save_result(
    "gorev7",
    {
        "isim_sayisi": len(data["words"]),
        "vocab": V,
        "bigram_dev_loss": round(bigram_dev, 4),
        "modeller": out,
        "en_iyi_model": best,
        "test_loss": round(test, 4),
        "bigram_ornekler": bigram_samples,
        "mlp_ornekler": mlp_samples,
        "listede_olanlar": [s for s in mlp_samples if s in known],
    },
)
