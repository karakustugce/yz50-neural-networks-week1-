"""Hafta 8: karakter düzeyi GPT."""
import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, count, temperature=1.0):
        for _ in range(count):
            logits, _ = self(idx[:, -1:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, _ = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return self.drop(F.softmax(scores, dim=-1)) @ v


class MultiHeadAttention(nn.Module):
    """Bütün head'lerin Q/K/V projeksiyonlarını tek matris çarpımıyla hesaplar."""
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd, n_head'e tam bölünmeli.")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def heads(t):
            return t.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                  nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, stage, dropout):
        super().__init__()
        self.has_ff = stage in {"feedforward", "residual", "layernorm", "gpt"}
        self.residual = stage in {"residual", "layernorm", "gpt"}
        self.has_ln = stage in {"layernorm", "gpt"}
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout) if self.has_ff else nn.Identity()
        self.ln1 = nn.LayerNorm(n_embd) if self.has_ln else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if self.has_ln and self.has_ff else nn.Identity()

    def forward(self, x):
        a = self.attn(self.ln1(x))
        x = x + a if self.residual else a
        if self.has_ff:
            f = self.ffwd(self.ln2(x))
            x = x + f if self.residual else f
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=128, n_head=4,
                 n_layer=4, dropout=0.1, stage="gpt"):
        super().__init__()
        valid = {"single_head", "multi_head", "feedforward", "residual", "layernorm", "gpt"}
        if stage not in valid:
            raise ValueError(f"Geçersiz mimari aşaması: {stage}")
        if n_embd % n_head:
            raise ValueError("n_embd, n_head'e tam bölünmeli.")
        self.block_size, self.stage = block_size, stage
        self.token = nn.Embedding(vocab_size, n_embd)
        self.position = nn.Embedding(block_size, n_embd)
        self.embedding_drop = nn.Dropout(dropout)
        if stage == "single_head":
            self.single_head = Head(n_embd, n_embd, block_size, dropout)
            self.blocks = nn.Identity()
            self.final_norm = nn.Identity()
        else:
            depth = n_layer if stage == "gpt" else 1
            self.blocks = nn.Sequential(*[
                TransformerBlock(n_embd, n_head, block_size, stage, dropout)
                for _ in range(depth)
            ])
            self.single_head = None
            self.final_norm = nn.LayerNorm(n_embd) if stage in {"layernorm", "gpt"} else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Girdi block_size'dan uzun.")
        pos = torch.arange(T, device=idx.device)
        x = self.embedding_drop(self.token(idx) + self.position(pos))
        x = self.single_head(x) if self.stage == "single_head" else self.blocks(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, count, temperature=1.0):
        training = self.training
        self.eval()
        try:
            for _ in range(count):
                logits, _ = self(idx[:, -self.block_size:])
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
            return idx
        finally:
            self.train(training)
