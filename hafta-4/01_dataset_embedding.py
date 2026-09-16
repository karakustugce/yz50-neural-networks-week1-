"""Gorev 1: 3 harflik baglam veri seti ve embedding tablosu."""

import torch
import torch.nn.functional as F

from ortak import SEED, build_dataset, load_names, save_result, vocabulary

words = load_names("en")
stoi, itos = vocabulary("en")
print(f"isim sayisi: {len(words)}, vocab: {len(stoi)}")

# ilk isimden hangi orneklerin ciktigina bakalim
for w in words[:1]:
    context = [0, 0, 0]
    for ch in w + ".":
        ix = stoi[ch]
        print("".join(itos[i] for i in context), "--->", itos[ix])
        context = context[1:] + [ix]

X, Y = build_dataset(words[:5], stoi)
print("\nilk 5 isim icin X:", X.shape, X.dtype, "| Y:", Y.shape)

X, Y = build_dataset(words, stoi)
print("tum veri icin X:", X.shape, "Y:", Y.shape)

# 27 harfin her biri 2 boyutlu bir vektor
g = torch.Generator().manual_seed(SEED)
C = torch.randn((27, 2), generator=g)

# tek harf: C[5] ile one-hot @ C ayni sey
one_hot_way = F.one_hot(torch.tensor(5), num_classes=27).float() @ C
print("\nC[5]          :", C[5])
print("one_hot(5) @ C:", one_hot_way)
print("esit mi?", torch.allclose(C[5], one_hot_way))

# pytorch indexleme tum tensoru tek seferde cekebiliyor
emb = C[X]
print("\nC[X].shape:", emb.shape, " -> (ornek sayisi, 3 harf, 2 boyut)")
print("X[13, 2] =", X[13, 2].item(), "| C[X][13, 2] =", emb[13, 2], "| C[X[13,2]] =", C[X[13, 2]])

save_result(
    "gorev1",
    {
        "isim_sayisi": len(words),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "emb_shape": list(emb.shape),
    },
)
