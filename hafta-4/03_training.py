"""Gorev 3: tek batch'i overfit et, learning rate tara, train/dev/test ile egit."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ortak import GRAFIK, SEED, build_dataset, get_splits, load_names, save_result, vocabulary


def make_params(n_embd=2, n_hidden=100, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    ps = [
        torch.randn((27, n_embd), generator=g),
        torch.randn((3 * n_embd, n_hidden), generator=g),
        torch.randn(n_hidden, generator=g),
        torch.randn((n_hidden, 27), generator=g),
        torch.randn(27, generator=g),
    ]
    for p in ps:
        p.requires_grad = True
    return ps


def loss_of(ps, X, Y):
    C, W1, b1, W2, b2 = ps
    emb = C[X]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    return F.cross_entropy(h @ W2 + b2, Y)


def step(ps, loss, lr):
    for p in ps:
        p.grad = None
    loss.backward()
    for p in ps:
        p.data += -lr * p.grad


# --- 1) tek batch overfit ---------------------------------------------------
stoi, _ = vocabulary("en")
Xs, Ys = build_dataset(load_names("en")[:5], stoi)
ps = make_params()
for i in range(1000):
    loss = loss_of(ps, Xs, Ys)
    step(ps, loss, 0.1)
print(f"tek batch ({Xs.shape[0]} ornek, 3481 parametre) 1000 adim sonra: {loss.item():.4f}")
# Neden 0'a inmiyor? "..." baglaminin arkasindan her isimde farkli harf geliyor.
# Ayni girdiye farkli cevap varsa model ancak ortalamasini ogrenebilir.
C, W1, b1, W2, b2 = ps
with torch.no_grad():
    pred = (torch.tanh(C[Xs].view(-1, 6) @ W1 + b1) @ W2 + b2).argmax(1)
print("dogru tahmin:", (pred == Ys).sum().item(), "/", len(Ys))
overfit_loss = loss.item()

# --- 2) veri bolme ---------------------------------------------------------
data = get_splits("en")
Xtr, Ytr = data["train"]
Xdev, Ydev = data["dev"]
Xte, Yte = data["test"]
print(f"\ntrain {Xtr.shape[0]} | dev {Xdev.shape[0]} | test {Xte.shape[0]}")

# --- 3) learning rate taramasi ---------------------------------------------
# 1000 adimda lr'yi 10^-3'ten 10^0'a kadar yavas yavas buyutuyoruz,
# her adimdaki loss'u o lr'nin karsisina yaziyoruz.
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
ps = make_params()
g = torch.Generator().manual_seed(SEED)
lri, lossi = [], []
for i in range(1000):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    loss = loss_of(ps, Xtr[ix], Ytr[ix])
    step(ps, loss, lrs[i].item())
    lri.append(lre[i].item())
    lossi.append(loss.item())

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(lri, lossi)
ax.set_xlabel("log10(learning rate)")
ax.set_ylabel("minibatch loss")
ax.set_title("Learning rate taramasi")
ax.axvline(-1, color="red", linestyle="--", label="lr = 0.1")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAFIK / "03_lr_taramasi.png", dpi=150)
plt.close(fig)
# grafikte loss 10^-1 civarinda en dusuk, sonrasinda zigzag yapip patliyor -> 0.1 sectim

# --- 4) tum veriyle minibatch egitimi --------------------------------------
# minibatch: gradient tam dogru degil ama cok hizli, bol adim atmak daha iyi
ps = make_params()
g = torch.Generator().manual_seed(SEED)
steps = 40_000
losses = []
for i in range(steps):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    loss = loss_of(ps, Xtr[ix], Ytr[ix])
    lr = 0.1 if i < 30_000 else 0.01  # sonda learning rate decay
    step(ps, loss, lr)
    losses.append(loss.log10().item())
    if i % 10_000 == 0:
        print(f"  {i:6d}: {loss.item():.4f}")

with torch.no_grad():
    tr = loss_of(ps, Xtr, Ytr).item()
    dev = loss_of(ps, Xdev, Ydev).item()
print(f"train loss {tr:.4f} | dev loss {dev:.4f}")
# train ile dev birbirine yakin -> overfit yok, model kucuk kaliyor (underfit)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(losses, linewidth=0.5)
ax.set_xlabel("adim")
ax.set_ylabel("log10(minibatch loss)")
ax.set_title("Minibatch egitimi (emb=2, hidden=100)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GRAFIK / "03_egitim_loss.png", dpi=150)
plt.close(fig)

save_result(
    "gorev3",
    {
        "tek_batch_overfit_loss": round(overfit_loss, 4),
        "split_boyutlari": [Xtr.shape[0], Xdev.shape[0], Xte.shape[0]],
        "secilen_lr": 0.1,
        "adim": steps,
        "train_loss": round(tr, 4),
        "dev_loss": round(dev, 4),
    },
)
