"""Gorev 5: baslangic loss'u, tanh doymasi ve Kaiming init."""

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ortak import GRAFIK, SEED, forward, get_splits, init_params, save_result, smooth, split_loss, train

data = get_splits("en")
Xtr, Ytr = data["train"]
Xdev, Ydev = data["dev"]

# Hic bir sey bilmeyen model 27 harfe esit olasilik vermeli:
expected = -math.log(1 / 27)
print(f"beklenen baslangic loss'u: -log(1/27) = {expected:.4f}")

g = torch.Generator().manual_seed(SEED)
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]


def inspect(p, label):
    with torch.no_grad():
        logits, acts = forward(p, Xb)
        loss = F.cross_entropy(logits, Yb).item()
        h = acts["h"]
        sat = (h.abs() > 0.99).float().mean().item()
        dead = (h.abs() > 0.99).all(0).sum().item()  # her ornekte doymus noron
    print(f"{label:>8}: ilk loss {loss:.4f} | logits max {logits.abs().max():.1f} "
          f"| |h|>0.99 orani %{sat * 100:.1f} | tamamen olu noron {dead}")
    return loss, sat, dead, acts


naive = init_params(27, init="naive")
kaiming = init_params(27, init="kaiming")
naive_loss, naive_sat, naive_dead, naive_acts = inspect(naive, "naive")
kai_loss, kai_sat, kai_dead, kai_acts = inspect(kaiming, "kaiming")

# Neden bu kadar yuksek? logits cok buyuk -> softmax kendinden emin ama yanlis.
# W2'yi 0.01 ile, b2'yi 0 ile carpinca logits ~0 oluyor ve loss 3.29'a iniyor.
#
# tanh neden doyuyor? hpreact cok genis dagiliyor (std ~ sqrt(30) civari).
# |x| buyukse tanh 1 veya -1 veriyor, turevi 1 - t^2 = 0 oluyor ve
# o norondan geriye gradient akmiyor.
# Kaiming: std = gain / sqrt(fan_in), tanh icin gain = 5/3.
print(f"\nhpreact std  naive: {naive_acts['hpreact'].std():.2f}  kaiming: {kai_acts['hpreact'].std():.2f}")
print(f"Kaiming W1 std hedefi: (5/3)/sqrt(30) = {(5/3)/30**0.5:.4f}")

# --- histogramlar -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for row, (label, acts) in enumerate([("naive", naive_acts), ("kaiming", kai_acts)]):
    axes[row, 0].hist(acts["h"].view(-1).tolist(), 50)
    axes[row, 0].set_title(f"{label}: h = tanh(hpreact)")
    axes[row, 1].hist(acts["hpreact"].view(-1).tolist(), 50)
    axes[row, 1].set_title(f"{label}: hpreact")
fig.tight_layout()
fig.savefig(GRAFIK / "05_tanh_histogram.png", dpi=150)
plt.close(fig)

# videodaki beyaz/siyah tablo: satir = ornek, sutun = noron, beyaz = |h| > 0.99
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
for ax, (label, acts) in zip(axes, [("naive", naive_acts), ("kaiming", kai_acts)]):
    ax.imshow(acts["h"].abs() > 0.99, cmap="gray", interpolation="nearest")
    ax.set_title(f"{label}: beyaz = doymus (|h| > 0.99)")
    ax.set_xlabel("noron")
    ax.set_ylabel("ornek")
fig.tight_layout()
fig.savefig(GRAFIK / "05_doymus_noronlar.png", dpi=150)
plt.close(fig)

# --- iki modeli de egitelim -----------------------------------------------
STEPS = 200_000
print("\nnaive egitim")
naive_curve = train(naive, Xtr, Ytr, steps=STEPS, log_every=50_000)
print("kaiming egitim")
kai_curve = train(kaiming, Xtr, Ytr, steps=STEPS, log_every=50_000)

out = {}
for label, p in [("naive", naive), ("kaiming", kaiming)]:
    tr, dev = split_loss(p, Xtr, Ytr), split_loss(p, Xdev, Ydev)
    out[label] = {"train_loss": round(tr, 4), "dev_loss": round(dev, 4)}
    print(f"{label:>8}: train {tr:.4f} | dev {dev:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(naive_curve[:2000], label="naive", linewidth=0.7)
axes[0].plot(kai_curve[:2000], label="kaiming", linewidth=0.7)
axes[0].set_title("ilk 2000 adim: naive'de hokey sopasi")
axes[1].plot(smooth(naive_curve), label="naive")
axes[1].plot(smooth(kai_curve), label="kaiming")
axes[1].set_title("tum egitim (1000 adim ortalama)")
for ax in axes:
    ax.set_ylabel("log10(loss)")
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAFIK / "05_init_loss.png", dpi=150)
plt.close(fig)

save_result(
    "gorev5",
    {
        "beklenen_ilk_loss": round(expected, 4),
        "naive": {
            "ilk_loss": round(naive_loss, 4),
            "doymus_oran": round(naive_sat, 4),
            "olu_noron": naive_dead,
            **out["naive"],
        },
        "kaiming": {
            "ilk_loss": round(kai_loss, 4),
            "doymus_oran": round(kai_sat, 4),
            "olu_noron": kai_dead,
            **out["kaiming"],
        },
    },
)
