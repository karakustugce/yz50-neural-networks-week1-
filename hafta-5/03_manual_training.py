"""Exercise 4: modeli yalnizca elle yazilan gradient'lerle egit.

Ornekler:
    python3 03_manual_training.py --steps 1000   # hizli kontrol
    python3 03_manual_training.py --steps 200000 # tam deney
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Bu modelde matrisler kucuk; tek thread, thread havuzu ek maliyetini onler.
torch.set_num_threads(1)

from ortak import (
    SEED,
    detailed_forward,
    fused_parameter_grads,
    get_splits,
    init_parameters,
    save_result,
)


def evaluate(parameters, X, Y, bnmean, bnvar, chunk_size=50_000):
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(X), chunk_size):
            xb, yb = X[start : start + chunk_size], Y[start : start + chunk_size]
            emb = parameters["C"][xb]
            embcat = emb.view(emb.shape[0], -1)
            hprebn = embcat @ parameters["W1"] + parameters["b1"]
            bnraw = (hprebn - bnmean) * (bnvar + 1e-5) ** -0.5
            h = torch.tanh(parameters["bngain"] * bnraw + parameters["bnbias"])
            logits = h @ parameters["W2"] + parameters["b2"]
            total += F.cross_entropy(logits, yb, reduction="sum").item()
    return total / len(X)


def calibrate(parameters, X):
    with torch.no_grad():
        emb = parameters["C"][X]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ parameters["W1"] + parameters["b1"]
        return (
            hprebn.mean(0, keepdim=True),
            hprebn.var(0, keepdim=True, unbiased=True),
        )


def train(steps: int, batch_size: int, n_hidden: int):
    data = get_splits()
    Xtr, Ytr = data["train"]
    Xdev, Ydev = data["dev"]
    parameters = init_parameters(n_hidden=n_hidden, requires_grad=False)
    generator = torch.Generator().manual_seed(SEED)
    last_loss = None
    history_steps = []
    history_losses = []

    # Autograd grafigi kurulmaz: tum gradient'ler fused_parameter_grads'ten gelir.
    with torch.no_grad():
        for step in range(steps):
            indices = torch.randint(0, len(Xtr), (batch_size,), generator=generator)
            Xb, Yb = Xtr[indices], Ytr[indices]
            loss, values = detailed_forward(parameters, Xb, Yb)
            grads = fused_parameter_grads(parameters, values, Xb, Yb)
            learning_rate = 0.1 if step < steps // 2 else 0.01
            for name, parameter in parameters.items():
                parameter += -learning_rate * grads[name]
            last_loss = loss.item()
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                print(f"{step:7d}/{steps:7d} | minibatch loss {last_loss:.4f}")
                history_steps.append(step + 1)
                history_losses.append(last_loss)

    bnmean, bnvar = calibrate(parameters, Xtr)
    train_loss = evaluate(parameters, Xtr, Ytr, bnmean, bnvar)
    dev_loss = evaluate(parameters, Xdev, Ydev, bnmean, bnvar)
    print(f"\ntrain loss: {train_loss:.4f} | dev loss: {dev_loss:.4f}")

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_steps, history_losses, marker="o", label="Minibatch loss")
    ax.axhline(train_loss, color="#397f92", linestyle="--", label=f"Train loss = {train_loss:.4f}")
    ax.axhline(dev_loss, color="#cf7548", linestyle="--", label=f"Dev loss = {dev_loss:.4f}")
    ax.set(title="Manual backpropagation training", xlabel="Training step", ylabel="Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    chart_path = output_dir / "manual_training_loss.png"
    fig.savefig(chart_path, dpi=160)
    fig.savefig(output_dir / "manual_training_loss.svg", format="svg")
    plt.close(fig)
    print(f"Loss grafiği kaydedildi: {chart_path}")
    return {
        "steps": steps,
        "batch_size": batch_size,
        "n_hidden": n_hidden,
        "son_minibatch_loss": round(last_loss, 6),
        "train_loss": round(train_loss, 6),
        "dev_loss": round(dev_loss, 6),
        "autograd_kullanildi": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=200)
    args = parser.parse_args()
    result = train(args.steps, args.batch_size, args.hidden)
    save_result("exercise_4", result)
