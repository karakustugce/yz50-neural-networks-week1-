"""Kayıtlı modelden eğitimi tekrar etmeden örnek üret."""
import argparse
import torch
from data import ROOT, CharacterTokenizer
from models import BigramLanguageModel, SingleHeadLanguageModel


def generate(model_name, prompt, tokens, seed):
    if not prompt or tokens < 0:
        raise ValueError('Prompt boş olamaz; tokens negatif olamaz.')
    checkpoint = torch.load(ROOT / 'checkpoints' / f'{model_name}.pt', map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    torch.set_num_threads(cfg['threads'])
    tokenizer = CharacterTokenizer(''.join(checkpoint['chars']))
    if model_name == 'bigram':
        model = BigramLanguageModel(len(tokenizer.chars), cfg['block_size'])
    else:
        model = SingleHeadLanguageModel(len(tokenizer.chars), cfg['block_size'], cfg['n_embd'], cfg['head_size'])
    model.load_state_dict(checkpoint['state_dict'])
    x = torch.tensor([tokenizer.encode(prompt)])
    sample = model.generate(x, tokens, torch.Generator().manual_seed(seed))
    return tokenizer.decode(sample[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bigram', 'single_head'], default='single_head')
    parser.add_argument('--prompt', default='ROMEO:\n')
    parser.add_argument('--tokens', type=int, default=400)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print(generate(args.model, args.prompt, args.tokens, args.seed))
