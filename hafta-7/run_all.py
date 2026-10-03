"""Tüm görevleri sırasıyla çalıştır; sonuçlar hafta-7/results içine yazılır."""
import runpy
from pathlib import Path
from experiment import Config, run_experiment
from download_data import download


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    download()
    for name in ('01_data_preview.py', '02_averages.py', '03_attention_demo.py'):
        runpy.run_path(str(root / name), run_name='__main__')
    run_experiment(Config(steps=args.steps, threads=args.threads))
    runpy.run_path(str(root / '05_heatmap.py'), run_name='__main__')
