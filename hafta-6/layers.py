"""Makemore Part 5 layers, written with tensors rather than torch.nn layers."""

import torch


class Linear:
    def __init__(self, fan_in, fan_out, generator, bias=True):
        self.weight = torch.randn(fan_in, fan_out, generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        return x @ self.weight + (self.bias if self.bias is not None else 0)

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, momentum=0.01, eps=1e-5, legacy_3d=False):
        self.training = True
        self.momentum, self.eps, self.legacy_3d = momentum, eps, legacy_3d
        self.gamma, self.beta = torch.ones(dim), torch.zeros(dim)
        self.running_mean = self.running_var = None

    def __call__(self, x):
        assert x.ndim in (2, 3) and x.shape[-1] == self.gamma.numel()
        axes = (0,) if x.ndim == 2 or self.legacy_3d else (0, 1)
        if self.training:
            mean = x.mean(dim=axes, keepdim=True)
            var = x.var(dim=axes, keepdim=True, unbiased=False)
            with torch.no_grad():
                if self.running_mean is None:
                    self.running_mean = mean.detach().clone()
                    self.running_var = var.detach().clone()
                else:
                    self.running_mean.lerp_(mean, self.momentum)
                    self.running_var.lerp_(var, self.momentum)
        else:
            assert self.running_mean is not None, "Train before evaluation"
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        return torch.tanh(x)

    def parameters(self):
        return []


class Embedding:
    def __init__(self, vocab_size, dim, generator):
        self.weight = torch.randn(vocab_size, dim, generator=generator)

    def __call__(self, x):
        return self.weight[x]

    def parameters(self):
        return [self.weight]


class Flatten:
    def __init__(self, n=None):
        self.n = n

    def __call__(self, x):
        if self.n is None:
            return x.reshape(x.shape[0], -1)
        assert x.ndim == 3 and x.shape[1] % self.n == 0
        y = x.reshape(x.shape[0], x.shape[1] // self.n, self.n * x.shape[2])
        return y[:, 0, :] if y.shape[1] == 1 else y

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x, trace=False):
        shapes = []
        for layer in self.layers:
            x = layer(x)
            if trace:
                shapes.append([type(layer).__name__, list(x.shape)])
        return (x, shapes) if trace else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = False


def make_model(vocab, context, emb=10, hidden=200, hierarchical=False,
               legacy_3d=False, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    if hierarchical:
        assert context == 8
        layers = [Embedding(vocab, emb, g)]
        dim = emb
        for _ in range(3):
            layers.extend([Flatten(2), Linear(2 * dim, hidden, g, bias=False),
                           BatchNorm1d(hidden, legacy_3d=legacy_3d), Tanh()])
            dim = hidden
        layers.append(Linear(hidden, vocab, g))
    else:
        layers = [Embedding(vocab, emb, g), Flatten(),
                  Linear(context * emb, hidden, g, bias=False),
                  BatchNorm1d(hidden), Tanh(), Linear(hidden, vocab, g)]
    layers[-1].weight *= 0.1
    model = Sequential(layers)
    for p in model.parameters():
        p.requires_grad_(True)
    return model
