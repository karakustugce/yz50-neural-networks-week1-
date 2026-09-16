"""Gorev 2: gizli katman, cikis katmani, elle loss ve F.cross_entropy."""

import torch
import torch.nn.functional as F

from ortak import SEED, build_dataset, load_names, save_result, vocabulary

words = load_names("en")
stoi, itos = vocabulary("en")
X, Y = build_dataset(words[:5], stoi)  # videodaki gibi ilk 5 isim (bizde 36 ornek)

g = torch.Generator().manual_seed(SEED)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

emb = C[X]  # (36, 3, 2)
# 3 harfin 2'lik vektorlerini yan yana koyup 6'lik tek vektor yapiyoruz.
# torch.cat(torch.unbind(emb, 1), 1) de ayni seyi yapar ama yeni bellek aciyor,
# view sadece ayni veriye farkli bakiyor.
cat_way = torch.cat(torch.unbind(emb, 1), 1)
view_way = emb.view(-1, 6)
print("cat ile view ayni mi?", torch.equal(cat_way, view_way))

h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # (36, 100)  b1 broadcast oluyor
logits = h @ W2 + b2  # (36, 27)
print("h:", h.shape, "logits:", logits.shape)

# gecen haftaki gibi elle
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
manual_loss = -prob[torch.arange(len(Y)), Y].log().mean()

ce_loss = F.cross_entropy(logits, Y)
print(f"elle loss      : {manual_loss.item():.6f}")
print(f"F.cross_entropy: {ce_loss.item():.6f}")

# Neden cross_entropy?
# 1) ara tensor olusturmuyor, forward/backward daha hizli
# 2) sayisal olarak guvenli: buyuk logit'te exp() inf'e tasiyor
big = torch.tensor([[-100.0, -3.0, 0.0, 100.0]])
c = big.exp()
print("\nbuyuk logit ile elle softmax:", (c / c.sum(1, keepdim=True)).tolist())
print("cross_entropy ile:", F.cross_entropy(big, torch.tensor([3])).item())
# cross_entropy icerde max logit'i cikariyor, sonuc degismiyor ama tasma olmuyor
shifted = big - big.max()
c = shifted.exp()
print("max'i cikarinca elle softmax:", (c / c.sum(1, keepdim=True)).tolist())

params = [C, W1, b1, W2, b2]
print("\nparametre sayisi:", sum(p.nelement() for p in params))

save_result(
    "gorev2",
    {
        "elle_loss": round(manual_loss.item(), 6),
        "cross_entropy_loss": round(ce_loss.item(), 6),
        "parametre_sayisi": sum(p.nelement() for p in params),
    },
)
