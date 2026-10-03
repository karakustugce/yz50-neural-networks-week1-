"""Görev 2: aynı geçmiş ortalaması, üç ayrı hesaplama."""
import json
import torch
from torch.nn import functional as F
from data import ROOT


def run():
    generator = torch.Generator().manual_seed(1337)
    x = torch.randn(2, 8, 4, generator=generator)
    loop = torch.zeros_like(x)
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            loop[b, t] = x[b, :t + 1].mean(dim=0)
    tril = torch.tril(torch.ones(8, 8))
    weights = tril / tril.sum(dim=1, keepdim=True)
    matrix = weights @ x
    scores = torch.zeros(8, 8).masked_fill(tril == 0, float('-inf'))
    softmax_weights = F.softmax(scores, dim=-1)
    softmax = softmax_weights @ x
    checks = {
        'loop_vs_matrix': torch.allclose(loop, matrix),
        'loop_vs_softmax': torch.allclose(loop, softmax),
        'matrix_vs_softmax': torch.allclose(matrix, softmax),
        'max_absolute_error': float((loop - matrix).abs().max()),
        'weights': weights.tolist(),
    }
    assert all(checks[key] for key in ('loop_vs_matrix', 'loop_vs_softmax', 'matrix_vs_softmax'))
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (out / 'averages.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('x:', tuple(x.shape), '\nwei:\n', weights)
    print('Üç yöntem allclose:', checks['loop_vs_matrix'], checks['loop_vs_softmax'], checks['matrix_vs_softmax'])
    print('out[b,t,c] = toplam_j wei[t,j] * x[b,j,c]. '
          't satırında j<=t için 1/(t+1), gelecekte 0 vardır; bu geçmişin ortalamasıdır.')


if __name__ == '__main__':
    run()
