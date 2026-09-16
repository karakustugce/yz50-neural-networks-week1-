"""Gorev 4: modeli buyut, embedding'leri ciz, isim ornekle, bigram ile karsilastir.

Part 2'deki gibi basit (naive) init kullaniyorum; init duzeltmesi 05'te.
"""

import matplotlib.pyplot as plt

from ortak import (
    GRAFIK,
    bigram_split_loss,
    get_splits,
    init_params,
    sample,
    sample_bigram,
    save_result,
    smooth,
    split_loss,
    train,
)

data = get_splits("en")
Xtr, Ytr = data["train"]
Xdev, Ydev = data["dev"]
itos = data["itos"]

STEPS = 200_000
configs = [
    # (embedding, hidden)
    (2, 100),
    (2, 300),
    (10, 200),
]

results = []
models = {}
curves = {}
for n_embd, n_hidden in configs:
    name = f"emb{n_embd}_h{n_hidden}"
    print(f"\n== {name} ==")
    p = init_params(27, n_embd=n_embd, n_hidden=n_hidden, init="naive")
    n_params = sum(t.nelement() for t in p.values())
    curves[name] = train(p, Xtr, Ytr, steps=STEPS, log_every=50_000)
    tr, dev = split_loss(p, Xtr, Ytr), split_loss(p, Xdev, Ydev)
    print(f"  parametre {n_params} | train {tr:.4f} | dev {dev:.4f}")
    results.append(
        {"model": name, "parametre": n_params, "train_loss": round(tr, 4), "dev_loss": round(dev, 4)}
    )
    models[name] = p

# loss egrileri
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, lossi in curves.items():
    ax.plot(smooth(lossi), label=name)
ax.set_xlabel("adim (x1000)")
ax.set_ylabel("log10(loss), 1000 adim ortalama")
ax.set_title("Model boyutu ve egitim loss'u")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAFIK / "04_model_boyutu_loss.png", dpi=150)
plt.close(fig)

# 2 boyutlu embedding cizimi (emb2_h300 modeli)
C = models["emb2_h300"]["C"].detach()
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(C[:, 0], C[:, 1], s=200)
for i in range(C.shape[0]):
    ax.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="white")
ax.set_title("Ogrenilen 2 boyutlu harf embedding'leri")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAFIK / "04_embedding_2d.png", dpi=150)
plt.close(fig)

# hangi harfler birbirine yakin?
dist = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1).sqrt()
dist.fill_diagonal_(float("inf"))
neighbours = {itos[i]: itos[dist[i].argmin().item()] for i in range(27)}
print("\nen yakin komsular:", neighbours)

# bigram ile karsilastirma (ayni train/dev bolmesi)
bigram_dev, P = bigram_split_loss(data["train_words"], data["dev_words"], data["stoi"])
bigram_samples = sample_bigram(P, itos, 15)
best = models["emb10_h200"]
mlp_samples = sample(best, itos, 15)
print(f"\nbigram dev loss: {bigram_dev:.4f}")
print("bigram  :", bigram_samples)
print("MLP     :", mlp_samples)

save_result(
    "gorev4",
    {
        "adim": STEPS,
        "modeller": results,
        "bigram_dev_loss": round(bigram_dev, 4),
        "en_yakin_harfler": neighbours,
        "bigram_ornekler": bigram_samples,
        "mlp_ornekler": mlp_samples,
    },
)
