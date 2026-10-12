import argparse
import csv
import json
import platform
import random
import time
import urllib.request
from pathlib import Path

import torch
from models import BigramLanguageModel, GPTLanguageModel

ROOT = Path(__file__).resolve().parent
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt"
STAGES = ["bigram", "single_head", "multi_head", "feedforward", "residual", "layernorm", "gpt"]


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def load_text(language):
    path = ROOT / "data" / ("shakespeare.txt" if language == "shakespeare" else "turkish.txt")
    path.parent.mkdir(exist_ok=True)
    if language == "shakespeare" and not path.exists():
        req = urllib.request.Request(SHAKESPEARE_URL, headers={"User-Agent": "YZ50-week8 educational"})
        with urllib.request.urlopen(req, timeout=60) as response:
            path.write_bytes(response.read())
    if not path.exists():
        raise FileNotFoundError(f"{path} bulunamadı; önce collect_turkish.py çalıştırın.")
    return path.read_text(encoding="utf-8")


def get_batch(data, split, block_size, batch_size, rng, device):
    cut = int(0.9 * len(data))
    source = data[:cut] if split == "train" else data[cut:]
    starts = torch.randint(len(source) - block_size - 1, (batch_size,), generator=rng)
    x = torch.stack([source[int(i):int(i)+block_size] for i in starts])
    y = torch.stack([source[int(i)+1:int(i)+block_size+1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate(model, data, cfg, device):
    was_training = model.training
    model.eval()
    result = {}
    for split in ("train", "val"):
        rng = torch.Generator().manual_seed(cfg["eval_seed"] + (split == "val"))
        values = []
        for _ in range(cfg["eval_batches"]):
            x, y = get_batch(data, split, cfg["block_size"], cfg["batch_size"], rng, device)
            _, loss = model(x, y)
            values.append(float(loss))
        result[split] = sum(values) / len(values)
    model.train(was_training)
    return result


def build_model(stage, vocab, cfg):
    if stage == "bigram":
        return BigramLanguageModel(vocab)
    return GPTLanguageModel(vocab, cfg["block_size"], cfg["n_embd"], cfg["n_head"],
                            cfg["n_layer"], cfg["dropout"] if stage == "gpt" else 0.0, stage)


def fit(stage, text, cfg, language, device):
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    model = build_model(stage, len(tokenizer.chars), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    batch_rng = torch.Generator().manual_seed(cfg["batch_seed"])
    history = []
    start = time.perf_counter()
    for step in range(cfg["steps"] + 1):
        if step % cfg["eval_interval"] == 0 or step == cfg["steps"]:
            score = estimate(model, data, cfg, device)
            history.append({"step": step, **score})
            print(f"{language:12s} {stage:12s} {step:5d} train={score['train']:.4f} val={score['val']:.4f}", flush=True)
        if step == cfg["steps"]:
            break
        x, y = get_batch(data, "train", cfg["block_size"], cfg["batch_size"], batch_rng, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    seconds = time.perf_counter() - start
    prompt = cfg["prompt"] if all(c in tokenizer.stoi for c in cfg["prompt"]) else text[:1]
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    torch.manual_seed(cfg["sample_seed"])
    sample = tokenizer.decode(model.generate(context, cfg["sample_tokens"], cfg["temperature"])[0].cpu())
    out = ROOT / "results" / language
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stage}_sample.txt").write_text(sample, encoding="utf-8")
    return {"stage": stage, "parameters": sum(p.numel() for p in model.parameters()),
            "train_loss": history[-1]["train"], "val_loss": history[-1]["val"],
            "seconds": seconds, "history": history, "sample": sample}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["shakespeare", "turkish"], required=True)
    p.add_argument("--stage", choices=STAGES, default="gpt")
    p.add_argument("--all-stages", action="store_true")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch-seed", type=int, default=7331)
    p.add_argument("--eval-seed", type=int, default=4242)
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument("--sample-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--prompt", default="ROMEO:\n")
    a = p.parse_args()
    if a.n_embd % a.n_head:
        p.error("--n-embd değeri --n-head'e tam bölünmeli.")
    if min(a.steps, a.block_size, a.eval_batches) < 1:
        p.error("steps, block-size ve eval-batches pozitif olmalı.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA seçildi ancak GPU yok.")
    cfg = vars(a).copy()
    text = load_text(a.language)
    tokenizer = CharTokenizer(text)
    print(json.dumps({"language": a.language, "characters": len(text),
                      "unique_characters": len(tokenizer.chars), "device": device,
                      "config": cfg}, ensure_ascii=False, indent=2))
    stages = STAGES if a.all_stages else [a.stage]
    results = [fit(stage, text, cfg, a.language, device) for stage in stages]
    out = ROOT / "results" / a.language
    out.mkdir(parents=True, exist_ok=True)
    report = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": device},
              "data": {"characters": len(text), "unique_characters": len(tokenizer.chars)},
              "config": cfg, "models": results,
              "cross_language_warning": "Farklı dillerin karakter dağılımı ve sözlük boyutu değiştiği için val loss doğrudan dil/model kalite kıyası değildir."}
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "loss_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "parameters", "train_loss_nats_per_char", "val_loss_nats_per_char", "training_seconds"])
        for r in results:
            w.writerow([r["stage"], r["parameters"], r["train_loss"], r["val_loss"], r["seconds"]])
    print(f"Sonuçlar yazıldı: {out}", flush=True)


if __name__ == "__main__":
    main()
