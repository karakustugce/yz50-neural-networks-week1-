"""Train the Week 6 comparisons on the exact Week 4 name-level splits.

Run from the repository root: python hafta-6/experiment.py --steps 30000
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from layers import make_model

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("week4", ROOT.parent / "hafta-4" / "ortak.py")
week4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(week4)


@torch.no_grad()
def evaluate(model, split, batch=4096):
    model.eval()
    X, Y = split
    total = 0.0
    for i in range(0, len(X), batch):
        logits = model(X[i:i+batch])
        total += F.cross_entropy(logits, Y[i:i+batch], reduction="sum").item()
    return total / len(X)


@torch.no_grad()
def sample(model, itos, context, count=15):
    model.eval()
    g = torch.Generator().manual_seed(2147483657)
    samples = []
    for _ in range(count):
        state, chars = [0] * context, []
        for _ in range(32):
            logits = model(torch.tensor([state]))
            ix = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g).item()
            if ix == 0:
                break
            chars.append(itos[ix])
            state = state[1:] + [ix]
        samples.append("".join(chars))
    return samples


def train_one(name, language, context, emb, hidden, hierarchical, legacy, steps):
    data = week4.get_splits(language, context)
    Xtr, Ytr = data["train"]
    model = make_model(len(data["stoi"]), context, emb, hidden,
                       hierarchical, legacy)
    count = sum(p.numel() for p in model.parameters())
    model.train()
    _, shapes = model(torch.zeros((2, context), dtype=torch.long), trace=True)
    generator = torch.Generator().manual_seed(2147483647)
    losses = []
    for i in range(steps):
        model.train()
        ix = torch.randint(len(Xtr), (32,), generator=generator)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for p in model.parameters():
            p.grad = None
        loss.backward()
        lr = 0.1 if i < steps // 2 else 0.01
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
        if i % 100 == 0:
            losses.append(loss.item())
        if i % max(1, steps // 4) == 0:
            print(f"{name} {i}/{steps} minibatch={loss.item():.4f}", flush=True)
    result = {"language": language, "context": context, "embedding": emb,
              "hidden": hidden, "steps": steps, "parameters": count,
              "train_loss": round(evaluate(model, data["train"]), 4),
              "dev_loss": round(evaluate(model, data["dev"]), 4),
              "shapes_batch_2": shapes, "samples": sample(model, data["itos"], context)}
    return result, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    # The first two models differ only by context length. The two hierarchical
    # English models differ only by their BatchNorm reduction axes.
    configs = [
        ("en_3_flat", "en", 3, 10, 200, False, False),
        ("en_8_flat", "en", 8, 10, 200, False, False),
        ("en_8_wavenet", "en", 8, 24, 128, True, False),
        ("en_8_legacy_bn", "en", 8, 24, 128, True, True),
        ("tr_8_wavenet", "tr", 8, 24, 128, True, False),
    ]
    results, curves = {}, {}
    for name, *cfg in configs:
        results[name], curves[name] = train_one(name, *cfg, args.steps)
        (ROOT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in curves.items():
        k = max(1, len(curve) // 100)
        # Average raw losses first, then take the logarithm. Averaging logs
        # would turn the graph into a geometric mean and understate spikes.
        y = [math.log10(sum(curve[j:j+k]) / len(curve[j:j+k]))
             for j in range(0, len(curve), k)]
        ax.plot([100 * j * k for j in range(len(y))], y, label=name)
    ax.set(xlabel="step", ylabel="log10(mean minibatch loss)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "loss.png", dpi=140)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
