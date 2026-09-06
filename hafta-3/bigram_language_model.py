"""YZ50 Hafta 3: sayimla ve sinir agiyla bigram dil modeli."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
TR_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"


def turkish_lower(text: str) -> str:
    return text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()


def load_names(path: Path, alphabet: str, turkish: bool = False) -> list[str]:
    """Dosyayi temizler; bos, tekrarli ve alfabe disi kayitlari atar."""
    raw = path.read_text(encoding="utf-8-sig")
    raw = turkish_lower(raw) if turkish else raw.lower()
    splitter = re.compile(r"[^" + re.escape(alphabet) + r"]+")
    names = {
        token
        for line in raw.splitlines()
        for token in splitter.split(line.strip())
        if token and all(ch in alphabet for ch in token)
    }
    return sorted(names)


def vocabulary(alphabet: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = [".", *alphabet]
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    return stoi, itos


def bigram_counts(names: list[str], stoi: dict[str, int]) -> tuple[Counter, torch.Tensor]:
    pairs: Counter[tuple[str, str]] = Counter()
    matrix = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
    for name in names:
        chars = [".", *name, "."]
        for first, second in zip(chars, chars[1:]):
            pairs[(first, second)] += 1
            matrix[stoi[first], stoi[second]] += 1
    return pairs, matrix


def probabilities(counts: torch.Tensor, smoothing: float = 1.0) -> torch.Tensor:
    smoothed = counts.float() + smoothing
    # keepdim=True sayesinde her satir kendi toplamina bolunur.
    return smoothed / smoothed.sum(dim=1, keepdim=True)


def nll_from_probabilities(
    names: list[str], probabilities_: torch.Tensor, stoi: dict[str, int]
) -> float:
    log_likelihood = 0.0
    pair_count = 0
    for name in names:
        chars = [".", *name, "."]
        for first, second in zip(chars, chars[1:]):
            probability = probabilities_[stoi[first], stoi[second]]
            log_likelihood += torch.log(probability).item()
            pair_count += 1
    return -log_likelihood / pair_count


def sample_names(
    probabilities_: torch.Tensor,
    itos: dict[int, str],
    count: int,
    seed: int,
) -> list[str]:
    generator = torch.Generator().manual_seed(seed)
    generated = []
    for _ in range(count):
        name = []
        index = 0
        for _ in range(30):
            index = torch.multinomial(
                probabilities_[index], 1, replacement=True, generator=generator
            ).item()
            if index == 0:
                break
            name.append(itos[index])
        generated.append("".join(name))
    return generated


def make_training_examples(
    names: list[str], stoi: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[int] = []
    targets: list[int] = []
    for name in names:
        chars = [".", *name, "."]
        for first, second in zip(chars, chars[1:]):
            inputs.append(stoi[first])
            targets.append(stoi[second])
    return torch.tensor(inputs), torch.tensor(targets)


def train_neural_bigram(
    names: list[str],
    stoi: dict[str, int],
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    xs, ys = make_training_examples(names, stoi)
    vocab_size = len(stoi)
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn((vocab_size, vocab_size), generator=generator, requires_grad=True)
    losses = []

    for epoch in range(epochs):
        one_hot = F.one_hot(xs, num_classes=vocab_size).float()
        logits = one_hot @ weights
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)
        loss = -probs[torch.arange(len(ys)), ys].log().mean()

        weights.grad = None
        loss.backward()
        weights.data += -learning_rate * weights.grad
        losses.append(loss.item())

        if epoch in {0, epochs - 1} or (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:>3}/{epochs}: loss={loss.item():.4f}")

    return weights.detach().softmax(dim=1), losses


def save_count_plot(
    counts: torch.Tensor, itos: dict[int, str], title: str, output: Path
) -> None:
    size = len(itos)
    fig, axis = plt.subplots(figsize=(12, 12))
    axis.imshow(counts, cmap="Blues")
    for row in range(size):
        for column in range(size):
            pair = itos[row] + itos[column]
            axis.text(column, row, pair, ha="center", va="bottom", fontsize=6)
            axis.text(
                column,
                row,
                str(counts[row, column].item()),
                ha="center",
                va="top",
                fontsize=5,
            )
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_loss_plot(losses: list[float], label: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(range(1, len(losses) + 1), losses)
    axis.set_title(f"{label} Sinir Agi Egitim Loss'u")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Negative log likelihood")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_language(
    label: str,
    data_path: Path,
    alphabet: str,
    turkish: bool,
    epochs: int,
    samples: int,
    seed: int,
) -> dict:
    print(f"\n=== {label} ===")
    names = load_names(data_path, alphabet, turkish)
    stoi, itos = vocabulary(alphabet)
    pairs, counts = bigram_counts(names, stoi)
    count_probs = probabilities(counts)
    count_loss = nll_from_probabilities(names, count_probs, stoi)

    print(f"Temiz isim sayisi: {len(names)}")
    print("En sik 10 bigram:", pairs.most_common(10))
    print(f"Sayim modeli NLL: {count_loss:.4f}")
    count_samples = sample_names(count_probs, itos, samples, seed)
    print("Sayim modeli ornekleri:", count_samples)

    neural_probs, losses = train_neural_bigram(
        names, stoi, epochs=epochs, learning_rate=50.0, seed=seed
    )
    neural_loss = nll_from_probabilities(names, neural_probs, stoi)
    neural_samples = sample_names(neural_probs, itos, samples, seed + 1)
    print(f"Sinir agi NLL: {neural_loss:.4f}")
    print("Sinir agi ornekleri:", neural_samples)

    plot_path = ROOT / f"bigram_{label.lower()}_counts.png"
    save_count_plot(counts, itos, f"{label} Bigram Sayim Tablosu", plot_path)
    loss_plot_path = ROOT / f"bigram_{label.lower()}_loss.png"
    save_loss_plot(losses, label, loss_plot_path)

    return {
        "language": label,
        "name_count": len(names),
        "vocabulary_size": len(stoi),
        "count_model_nll": round(count_loss, 4),
        "neural_model_nll": round(neural_loss, 4),
        "count_samples": count_samples,
        "neural_samples": neural_samples,
        "first_neural_loss": round(losses[0], 4),
        "last_neural_loss": round(losses[-1], 4),
        "plot": plot_path.name,
        "loss_plot": loss_plot_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YZ50 Hafta 3 bigram modelleri")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2147483647)
    args = parser.parse_args()

    results = [
        run_language(
            "English",
            ROOT / "names_en.txt",
            "abcdefghijklmnopqrstuvwxyz",
            False,
            args.epochs,
            args.samples,
            args.seed,
        ),
        run_language(
            "Turkish",
            ROOT / "names_tr_raw.txt",
            TR_ALPHABET,
            True,
            args.epochs,
            args.samples,
            args.seed,
        ),
    ]
    (ROOT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSonuclar results.json dosyasina yazildi.")


if __name__ == "__main__":
    main()
