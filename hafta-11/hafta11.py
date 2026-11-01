"""GPT-2 Small training and step-by-step performance benchmark for YZ50 week 11."""

import argparse
import csv
import math
import random
import statistics
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
GPT2_VOCAB = 50_257
PADDED_VOCAB = 50_304


@dataclass
class GPTConfig:
    vocab_size: int = PADDED_VOCAB
    block_size: int = 128
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class LayerNorm(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, use_sdpa: bool):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_sdpa = use_sdpa
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            weights = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
            mask = self.causal_mask[:, :, :length, :length]
            weights = weights.masked_fill(mask == 0, float("-inf"))
            weights = F.softmax(weights, dim=-1)
            y = weights @ v

        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, use_sdpa: bool):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, use_sdpa)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2Small(nn.Module):
    def __init__(self, config: GPTConfig, use_sdpa: bool = True):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList(
            [Block(config, use_sdpa=use_sdpa) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 scales the residual output projections so residual sums do not
        # grow too quickly as the number of Transformer blocks increases.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.c_proj.weight") or name.endswith("mlp.c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

        # The output classifier reuses the token embedding matrix.
        self.lm_head.weight = self.wte.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        batch, length = idx.shape
        if length > self.config.block_size:
            raise ValueError(
                f"Sequence length {length} exceeds block_size={self.config.block_size}"
            )
        positions = torch.arange(length, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_amp_dtype(device: torch.device, requested: str = "auto"):
    if requested == "none":
        return None, "off"
    if requested == "bfloat16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        if device.type == "cpu":
            return torch.bfloat16, "bfloat16"
        return None, "unsupported"
    if requested == "float16":
        if device.type in ("cuda", "mps"):
            return torch.float16, "float16"
        return None, "unsupported"
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    if device.type == "mps":
        return torch.float16, "float16"
    return None, "off"


def make_scaler(device: torch.device, amp_dtype):
    if device.type != "cuda" or amp_dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def train_step(model, optimizer, x, y, device, amp_dtype, scaler, clip_norm=1.0):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, amp_dtype):
        _, loss = model(x, y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return loss.detach(), float(grad_norm.detach().cpu())


def count_parameters(model: nn.Module):
    tied_total = sum(parameter.numel() for parameter in model.parameters())
    embedding_count = model.config.vocab_size * model.config.n_embd
    untied_total = tied_total + embedding_count
    return tied_total, untied_total, embedding_count


def build_optimizer(model: GPT2Small, learning_rate: float, weight_decay: float):
    decay, no_decay = [], []
    decay_count = 0
    no_decay_count = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
            decay_count += parameter.numel()
        else:
            no_decay.append(parameter)
            no_decay_count += parameter.numel()
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer, decay_count, no_decay_count


def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float):
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def plot_lr_schedule(path: Path, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float):
    import matplotlib.pyplot as plt

    steps = list(range(total_steps + 1))
    rates = [get_lr(step, total_steps, warmup_steps, max_lr, min_lr) for step in steps]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rates, color="#2855a6", linewidth=2.5)
    ax.axvline(warmup_steps, color="#d97706", linestyle="--", linewidth=1.2, label="warmup biter")
    ax.set(title="GPT-3 tarzı warmup + cosine learning rate", xlabel="Optimizer adımı", ylabel="Learning rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ensure_data(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Veri indiriliyor: {TINY_SHAKESPEARE_URL}")
    try:
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    except Exception as error:
        raise RuntimeError(
            f"{path} bulunamadı ve örnek veri indirilemedi. Kendi metin dosyanı --data ile ver."
        ) from error


def load_token_stream(path: Path, device: torch.device):
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError("GPT-2 tokenizer için `pip install tiktoken` çalıştır.") from error
    ensure_data(path)
    text = path.read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        raise ValueError("Eğitim metni en az iki token içermeli.")
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    print(f"Veri: {path} | {len(text):,} karakter | {len(token_ids):,} GPT-2 token")
    print("Batch'ler tüm token akışından rastgele pencerelerle seçiliyor.")
    return tokens


def get_batch(tokens, batch_size: int, block_size: int, device: torch.device):
    max_start = tokens.numel() - block_size - 1
    if max_start < 0:
        raise ValueError(
            f"Metinde {tokens.numel()} token var; block_size={block_size} için daha uzun metin gerekli."
        )
    starts = torch.randint(max_start + 1, (batch_size,), device=device)
    offsets = torch.arange(block_size + 1, device=device)
    windows = tokens[starts[:, None] + offsets[None, :]]
    return windows[:, :-1], windows[:, 1:]


def report_model(model: GPT2Small):
    tied, untied, saved = count_parameters(model)
    print(f"Parametre sayısı (weight tying ile): {tied:,}")
    print(f"lm_head bağımsız olsaydı:          {untied:,}")
    print(f"Weight tying ile kazanılan:        {saved:,} ({100 * saved / untied:.2f}%)")


def train(args):
    device = choose_device(args.device)
    amp_dtype, amp_name = get_amp_dtype(device, args.amp)
    print(f"Cihaz: {device} | AMP: {amp_name}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
    )
    model = GPT2Small(config, use_sdpa=True).to(device)
    report_model(model)
    optimizer, decay_count, no_decay_count = build_optimizer(
        model, args.max_lr, args.weight_decay
    )
    print(f"Weight decay uygulanan parametre: {decay_count:,} (ndim >= 2)")
    print(f"Weight decay uygulanmayan parametre: {no_decay_count:,} (bias/LayerNorm)")
    if args.compile:
        if not hasattr(torch, "compile"):
            print("torch.compile bulunamadı; eager modda devam ediliyor.")
        else:
            print("torch.compile hazırlanıyor; ilk derleme adımları yavaş olabilir.")
            model = torch.compile(model)

    tokens = load_token_stream(Path(args.data), device)
    scaler = make_scaler(device, amp_dtype)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "learning_rate_schedule.png"
    plot_lr_schedule(plot_path, args.steps, args.warmup_steps, args.max_lr, args.min_lr)
    print(f"Learning rate grafiği: {plot_path}")

    log_path = results_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["step", "learning_rate", "loss", "gradient_norm", "tokens_seen"]
        )
        writer.writeheader()
        tokens_seen = 0
        for step in range(args.steps):
            lr = get_lr(step, args.steps, args.warmup_steps, args.max_lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(args.grad_accum):
                x, y = get_batch(tokens, args.batch_size, args.block_size, device)
                with autocast_context(device, amp_dtype):
                    _, loss = model(x, y)
                    scaled_loss = loss / args.grad_accum
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                micro_losses.append(float(loss.detach().cpu()))
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            loss_value = statistics.mean(micro_losses)
            tokens_seen += args.batch_size * args.block_size * args.grad_accum
            grad_value = float(grad_norm.detach().cpu())
            writer.writerow(
                {
                    "step": step + 1,
                    "learning_rate": f"{lr:.8g}",
                    "loss": f"{loss_value:.6f}",
                    "gradient_norm": f"{grad_value:.6f}",
                    "tokens_seen": tokens_seen,
                }
            )
            csv_file.flush()
            print(
                f"step {step + 1:4d}/{args.steps} | loss {loss_value:.4f} "
                f"| grad_norm {grad_value:.4f} | lr {lr:.3e} | tokens {tokens_seen:,}"
            )
    print(f"Eğitim günlüğü: {log_path}")


def tf32_supported(device: torch.device):
    if device.type != "cuda":
        return False, "TF32 yalnızca CUDA GPU'larda kullanılabilir."
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        return False, f"GPU compute capability {capability[0]}.{capability[1]}; TF32 için Ampere (8.0+) gerekir."
    return True, ""


def flash_backend_context():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])


def benchmark_stage(
    name,
    device,
    vocab_size,
    use_sdpa,
    use_tf32,
    amp_dtype,
    compile_model,
    force_flash,
    batch_size,
    block_size,
    warmup_steps,
    measured_steps,
    seed,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size)
    model = GPT2Small(config, use_sdpa=use_sdpa).to(device)
    optimizer, _, _ = build_optimizer(model, 6e-4, 0.1)
    if compile_model:
        model = torch.compile(model)
    x = torch.randint(GPT2_VOCAB, (batch_size, block_size), device=device)
    y = torch.randint(GPT2_VOCAB, (batch_size, block_size), device=device)
    scaler = make_scaler(device, amp_dtype)
    attention_context = flash_backend_context if force_flash else nullcontext

    def update():
        with attention_context():
            train_step(model, optimizer, x, y, device, amp_dtype, scaler)

    model.train()
    for _ in range(warmup_steps):
        update()
    synchronize(device)
    times = []
    tokens_per_step = batch_size * block_size
    for _ in range(measured_steps):
        synchronize(device)
        start = time.perf_counter()
        update()
        synchronize(device)
        times.append(time.perf_counter() - start)
    average = statistics.mean(times)
    return {
        "stage": name,
        "status": "measured",
        "reason": "",
        "vocab_size": vocab_size,
        "step_seconds": average,
        "tokens_per_second": tokens_per_step / average,
    }


def skipped_row(name, reason, vocab_size):
    return {
        "stage": name,
        "status": "skipped",
        "reason": reason,
        "vocab_size": vocab_size,
        "step_seconds": "",
        "tokens_per_second": "",
    }


def benchmark(args):
    device = choose_device(args.device)
    amp_dtype, amp_name = get_amp_dtype(device, args.amp)
    print(f"Benchmark cihazı: {device} | AMP adımı: {amp_name}")
    results = []
    current_tf32 = False
    current_amp = None
    current_compile = False
    current_sdpa = False
    current_flash = False

    def run_stage(name, vocab_size):
        try:
            row = benchmark_stage(
                name=name,
                device=device,
                vocab_size=vocab_size,
                use_sdpa=current_sdpa,
                use_tf32=current_tf32,
                amp_dtype=current_amp,
                compile_model=current_compile,
                force_flash=current_flash,
                batch_size=args.batch_size,
                block_size=args.block_size,
                warmup_steps=args.bench_warmup,
                measured_steps=args.bench_steps,
                seed=args.seed,
            )
            results.append(row)
            return True
        except (RuntimeError, ImportError, AttributeError) as error:
            results.append(
                skipped_row(
                    name,
                    f"Çalıştırılamadı: {str(error).splitlines()[0]}",
                    vocab_size,
                )
            )
            return False
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    run_stage("Temel FP32", GPT2_VOCAB)

    supported, reason = tf32_supported(device)
    if supported:
        current_tf32 = True
        run_stage("+ TF32", GPT2_VOCAB)
    else:
        results.append(skipped_row("+ TF32", reason, GPT2_VOCAB))

    if amp_dtype is not None:
        current_amp = amp_dtype
        run_stage(f"+ AMP {amp_name}", GPT2_VOCAB)
    else:
        results.append(
            skipped_row(
                "+ AMP",
                "Bu cihaz/ayar için mixed precision desteklenmiyor.",
                GPT2_VOCAB,
            )
        )

    if device.type in ("cuda", "mps") and hasattr(torch, "compile"):
        current_compile = True
        if not run_stage("+ torch.compile", GPT2_VOCAB):
            current_compile = False
    else:
        results.append(
            skipped_row(
                "+ torch.compile",
                "Bu ölçümde compile yalnızca CUDA/MPS cihazında deneniyor.",
                GPT2_VOCAB,
            )
        )

    if device.type == "cuda":
        current_sdpa = True
        current_flash = True
        if not run_stage("+ Flash Attention", GPT2_VOCAB):
            current_sdpa = False
            current_flash = False
    else:
        results.append(
            skipped_row(
                "+ Flash Attention",
                "Bu görevde zorlanmış PyTorch Flash Attention yalnızca CUDA'da deneniyor.",
                GPT2_VOCAB,
            )
        )

    run_stage("+ vocab 50,304", PADDED_VOCAB)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "benchmark_results.csv"
    fields = ["stage", "status", "reason", "vocab_size", "step_seconds", "tokens_per_second"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n| Aşama | Durum | Adım (s) | Token/s | Not |")
    print("|---|---:|---:|---:|---|")
    for row in results:
        seconds = f"{row['step_seconds']:.4f}" if isinstance(row["step_seconds"], float) else "—"
        tps = f"{row['tokens_per_second']:.0f}" if isinstance(row["tokens_per_second"], float) else "—"
        print(f"| {row['stage']} | {row['status']} | {seconds} | {tps} | {row['reason']} |")
    measured = [row for row in results if row["status"] == "measured"]
    if measured:
        best = min(measured, key=lambda row: row["step_seconds"])
        base = next((row for row in measured if row["stage"] == "Temel FP32"), None)
        if base:
            speedup = base["step_seconds"] / best["step_seconds"]
            print(f"En hızlı ölçüm: {best['stage']} ({speedup:.2f}x temel FP32 hızında).")
    print(f"Ölçüm CSV dosyası: {csv_path}")


def make_parser():
    parser = argparse.ArgumentParser(description="YZ50 hafta 11: hızlı GPT-2 eğitimi")
    parser.add_argument("mode", choices=["benchmark", "train"])
    parser.add_argument("--device", default="auto", help="auto, cuda, mps veya cpu")
    parser.add_argument("--data", default="input.txt", help="UTF-8 eğitim metni; yoksa Tiny Shakespeare indirilir")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=PADDED_VOCAB)
    parser.add_argument("--amp", choices=["auto", "none", "bfloat16", "float16"], default="auto")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--bench-warmup", type=int, default=3)
    parser.add_argument("--bench-steps", type=int, default=10)
    return parser


def main():
    args = make_parser().parse_args()
    if args.steps <= 0 or args.grad_accum <= 0 or args.batch_size <= 0:
        raise ValueError("steps, grad-accum ve batch-size pozitif olmalı.")
    if args.mode == "train":
        train(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
