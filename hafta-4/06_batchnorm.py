"""Gorev 6: gizli katmandan sonra BatchNorm, BN'li ve BN'siz karsilastirma.

BatchNorm: hpreact'i her noron icin batch uzerinden ortalama 0, std 1 yapiyor,
sonra ogrenilen bngain ile olcekleyip bnbias ile kaydiriyor.
Egitimde batch istatistigi, tahminde running mean/std kullaniliyor.
"""

import matplotlib.pyplot as plt
import torch

from ortak import (
    GRAFIK,
    forward,
    get_splits,
    init_params,
    new_bn_stats,
    sample,
    save_result,
    smooth,
    split_loss,
    train,
)

data = get_splits("en")
Xtr, Ytr = data["train"]
Xdev, Ydev = data["dev"]
STEPS = 200_000

runs = {
    "kaiming_bn_yok": dict(init="kaiming", batchnorm=False),
    "kaiming_bn_var": dict(init="kaiming", batchnorm=True),
    # BN'in init'i ne kadar tolere ettigini gormek icin: hic olceklenmemis agirliklar
    "naive_bn_var": dict(init="naive", batchnorm=True),
}

models, stats, curves, out = {}, {}, {}, {}
for name, cfg in runs.items():
    print(f"\n== {name} ==")
    p = init_params(27, **cfg)
    bn = new_bn_stats() if cfg["batchnorm"] else None
    # egitim oncesi doymus noron orani
    with torch.no_grad():
        _, acts = forward(p, Xtr[:32], bn_stats=new_bn_stats() if bn else None, training=True)
    sat0 = (acts["h"].abs() > 0.99).float().mean().item()
    curves[name] = train(p, Xtr, Ytr, steps=STEPS, log_every=50_000, bn_stats=bn)
    tr, dev = split_loss(p, Xtr, Ytr, bn), split_loss(p, Xdev, Ydev, bn)
    print(f"  baslangicta doymus %{sat0 * 100:.1f} | train {tr:.4f} | dev {dev:.4f}")
    models[name], stats[name] = p, bn
    out[name] = {"baslangic_doymus_oran": round(sat0, 4), "train_loss": round(tr, 4), "dev_loss": round(dev, 4)}

# running mean, tum train setinden hesaplanan gercek istatistige yakin mi?
p = models["kaiming_bn_var"]
with torch.no_grad():
    emb = p["C"][Xtr]
    hpre = emb.view(emb.shape[0], -1) @ p["W1"]
    calib = {"mean": hpre.mean(0, keepdim=True), "std": hpre.std(0, keepdim=True)}
running = stats["kaiming_bn_var"]
mean_diff = (calib["mean"] - running["mean"]).abs().max().item()
calib_dev = split_loss(p, Xdev, Ydev, calib)
print(f"\nrunning vs kalibre mean max fark: {mean_diff:.4f}")
print(f"kalibre istatistikle dev: {calib_dev:.4f} | running ile: {out['kaiming_bn_var']['dev_loss']:.4f}")

# tek ornekle tahmin: batch istatistigi olsaydi 1 ornegin std'si hesaplanamazdi,
# running istatistik sayesinde tek tek isim uretebiliyoruz
bn_samples = sample(p, data["itos"], 10, bn_stats=running)
print("BN'li modelden ornekler:", bn_samples)

# --- grafikler ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, c in curves.items():
    ax.plot(smooth(c), label=name)
ax.set_xlabel("adim (x1000)")
ax.set_ylabel("log10(loss), 1000 adim ortalama")
ax.set_title("BatchNorm karsilastirmasi")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAFIK / "06_batchnorm_loss.png", dpi=150)
plt.close(fig)

# naive init + BN: agirliklar buyuk olsa da tanh girdisi normalize oluyor
naive_plain = init_params(27, init="naive")
naive_bn = init_params(27, init="naive", batchnorm=True)
g = torch.Generator().manual_seed(1)
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
with torch.no_grad():
    _, a1 = forward(naive_plain, Xtr[ix])
    _, a2 = forward(naive_bn, Xtr[ix], bn_stats=new_bn_stats(), training=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(a1["h"].view(-1).tolist(), 50)
axes[0].set_title("naive init, BN yok: tanh doymus")
axes[1].hist(a2["h"].view(-1).tolist(), 50)
axes[1].set_title("naive init, BN var: tanh girdisi normalize")
fig.tight_layout()
fig.savefig(GRAFIK / "06_bn_tanh_histogram.png", dpi=150)
plt.close(fig)

save_result(
    "gorev6",
    {
        "adim": STEPS,
        "modeller": out,
        "running_vs_kalibre_mean_max_fark": round(mean_diff, 4),
        "kalibre_dev_loss": round(calib_dev, 4),
        "bn_ornekler": bn_samples,
    },
)
