"""Görev 1 ve 4: eşit bütçe, sabit değerlendirme pencereleri, gerçek sonuçlar."""
import argparse
import json
import platform
import time
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from data import ROOT, get_batch, load_data
from models import BigramLanguageModel, SingleHeadLanguageModel


@dataclass
class Config:
    steps: int = 5000
    block_size: int = 32
    batch_size: int = 32
    n_embd: int = 32
    head_size: int = 32
    learning_rate: float = 0.003
    eval_batches: int = 100
    eval_interval: int = 500
    seed: int = 1337
    batch_seed: int = 7331
    eval_seed: int = 4242
    sample_seed: int = 2026
    threads: int = 2


@torch.no_grad()
def estimate_loss(model, splits, cfg):
    was_training = model.training
    model.eval()
    result = {}
    # Her model ve her ölçüm aynı train/val pencerelerini görür.
    for split in ('train', 'val'):
        g = torch.Generator().manual_seed(cfg.eval_seed + (split == 'val'))
        losses = []
        for _ in range(cfg.eval_batches):
            x, y = get_batch(splits, split, cfg.block_size, cfg.batch_size, g)
            _, loss = model(x, y)
            losses.append(float(loss))
        result[split] = sum(losses) / len(losses)
    model.train(was_training)
    return result


def run_experiment(cfg):
    if cfg.steps < 1 or cfg.eval_batches < 1 or cfg.eval_interval < 1 or cfg.threads < 1:
        raise ValueError('Adım, değerlendirme ve thread sayıları pozitif olmalı.')
    torch.set_num_threads(cfg.threads)
    tokenizer, splits, metadata = load_data()
    results_dir = ROOT / 'results'
    checkpoints = ROOT / 'checkpoints'
    results_dir.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    results = {
        'config': asdict(cfg), 'data': metadata,
        'environment': {'torch': torch.__version__, 'python': platform.python_version(),
                        'platform': platform.platform(), 'device': 'cpu'},
        'evaluation': 'mean cross entropy (nats/token) over fixed random windows; not full val set',
        'ai_contribution': 'Code, experiments and explanations prepared by OpenAI Codex.',
        'models': {},
    }
    for name in ('bigram', 'single_head'):
        torch.manual_seed(cfg.seed)
        if name == 'bigram':
            model = BigramLanguageModel(len(tokenizer.chars), cfg.block_size)
        else:
            model = SingleHeadLanguageModel(len(tokenizer.chars), cfg.block_size, cfg.n_embd, cfg.head_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        train_generator = torch.Generator().manual_seed(cfg.batch_seed)
        history = []
        start = time.perf_counter()
        for step in range(cfg.steps + 1):
            if step % cfg.eval_interval == 0 or step == cfg.steps:
                measured = estimate_loss(model, splits, cfg)
                history.append({'step': step, **measured})
                print(f'{name:12s} step {step:5d} train {measured["train"]:.4f} val {measured["val"]:.4f}', flush=True)
            if step == cfg.steps:
                break
            x, y = get_batch(splits, 'train', cfg.block_size, cfg.batch_size, train_generator)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        sample_generator = torch.Generator().manual_seed(cfg.sample_seed)
        prompt = 'ROMEO:\n'
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        sample = tokenizer.decode(model.generate(context, 400, sample_generator)[0])
        (results_dir / f'{name}_sample.txt').write_text(sample, encoding='utf-8')
        results['models'][name] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'train_loss': history[-1]['train'], 'val_loss': history[-1]['val'],
            'seconds': time.perf_counter() - start, 'history': history,
            'generated_text': sample,
        }
        # weights_only=True ile okunabilir; yeniden eğitim gerektirmeyen görsel.
        torch.save({'state_dict': model.state_dict(), 'config': asdict(cfg),
                    'chars': tokenizer.chars, 'model': name}, checkpoints / f'{name}.pt')
        (results_dir / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    base = results['models']['bigram']['val_loss']
    attention = results['models']['single_head']['val_loss']
    results['comparison'] = {'absolute_drop': base - attention, 'relative_drop_percent': 100 * (base - attention) / base}
    (results_dir / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, result in results['models'].items():
        ax.plot([h['step'] for h in result['history']], [h['val'] for h in result['history']], label=name)
    ax.set(xlabel='Optimizer steps', ylabel='Validation cross entropy (nats/token)', title='Bigram vs single causal attention head')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / 'loss.png', dpi=160)
    plt.close(fig)
    print('Karşılaştırma:', results['comparison'], flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run_experiment(Config(steps=args.steps, threads=args.threads))
