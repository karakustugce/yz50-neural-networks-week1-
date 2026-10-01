"""Isolate the effect of context length for Turkish with a fixed flat MLP."""
import argparse
import json
from pathlib import Path

import torch
from experiment import train_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = Path(__file__).with_name("results.json")
    results = json.loads(path.read_text()) if path.exists() else {}
    for context in (3, 8):
        name = f"tr_{context}_flat"
        results[name], _ = train_one(name, "tr", context, 10, 200,
                                     False, False, args.steps)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
