"""Hafta 7: bigram ve tek causal attention head. Tam Transformer değildir.

AI katkısı: Bu uygulama ve açıklamalar Codex tarafından hazırlanmıştır.
"""
import torch
from torch import nn
from torch.nn import functional as F


def language_loss(logits, targets):
    if targets is None:
        return None
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


class LanguageModel(nn.Module):
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, generator=None):
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Öğrenilen pozisyonlar yalnızca 0..block_size-1 aralığında.
                context = idx[:, -self.block_size:]
                logits, _ = self(context)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_idx = torch.multinomial(probs, 1, generator=generator)
                idx = torch.cat((idx, next_idx), dim=1)
            return idx
        finally:
            self.train(was_training)


class BigramLanguageModel(LanguageModel):
    def __init__(self, vocab_size, block_size=32):
        super().__init__()
        self.block_size = block_size
        # Her harf için doğrudan sonraki harfin V adet logiti.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,V)
        return logits, language_loss(logits, targets)


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x, return_weights=False):
        _, T, _ = x.shape
        if T > self.tril.shape[0]:
            raise ValueError('Bağlam block_size değerini aşamaz.')
        q = self.query(x)  # (B,T,H)
        k = self.key(x)    # (B,T,H)
        v = self.value(x)  # (B,T,H)
        scores = (q @ k.transpose(-2, -1)) * self.head_size ** -0.5
        scores = scores.masked_fill(~self.tril[:T, :T], float('-inf'))
        wei = F.softmax(scores, dim=-1)  # (B,T,T), satır toplamı 1
        out = wei @ v                  # (B,T,H)
        return (out, wei) if return_weights else out


class SingleHeadLanguageModel(LanguageModel):
    def __init__(self, vocab_size, block_size=32, n_embd=32, head_size=32):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd, head_size, block_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def features(self, idx):
        T = idx.shape[1]
        if not 0 < T <= self.block_size:
            raise ValueError('Giriş uzunluğu 1..block_size aralığında olmalı.')
        positions = torch.arange(T, device=idx.device)
        return self.token_embedding_table(idx) + self.position_embedding_table(positions)

    def forward(self, idx, targets=None):
        x = self.features(idx)          # (B,T,C)
        x = self.sa_head(x)             # (B,T,H)
        logits = self.lm_head(x)         # (B,T,V)
        return logits, language_loss(logits, targets)

    @torch.no_grad()
    def attention_weights(self, idx):
        _, wei = self.sa_head(self.features(idx), return_weights=True)
        return wei
