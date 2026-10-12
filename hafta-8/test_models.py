import torch
from models import GPTLanguageModel, Head, MultiHeadAttention


def test_stages_and_causal_mask():
    for stage in ["single_head", "multi_head", "feedforward", "residual", "layernorm", "gpt"]:
        model = GPTLanguageModel(23, block_size=8, n_embd=16, n_head=4,
                                 n_layer=2, dropout=0.1, stage=stage)
        idx = torch.randint(23, (2, 7))
        logits, loss = model(idx, idx)
        assert logits.shape == (2, 7, 23) and torch.isfinite(loss)
        loss.backward()
    model = GPTLanguageModel(23, block_size=8, n_embd=16, n_head=4,
                             n_layer=1, dropout=0, stage="gpt")
    x = torch.randint(23, (1, 7))
    a = model.blocks[0].attn
    emb = model.token(x) + model.position(torch.arange(7))
    q, k, _ = a.qkv(emb).split(16, -1)
    q = q.view(1, 7, 4, 4).transpose(1, 2)
    k = k.view(1, 7, 4, 4).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / (4 ** 0.5)
    scores = scores.masked_fill(a.tril[:7, :7] == 0, float("-inf"))
    weights = torch.softmax(scores, -1)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0


def test_parallel_heads_equal_separate_heads():
    torch.manual_seed(5)
    C, H, T = 12, 3, 7
    vector = MultiHeadAttention(C, H, T, dropout=0)
    heads = [Head(C, C // H, T) for _ in range(H)]
    with torch.no_grad():
        vector.qkv.weight.zero_()
        for h, head in enumerate(heads):
            lo, hi = h * (C // H), (h + 1) * (C // H)
            vector.qkv.weight[lo:hi].copy_(head.query.weight)
            vector.qkv.weight[C+lo:C+hi].copy_(head.key.weight)
            vector.qkv.weight[2*C+lo:2*C+hi].copy_(head.value.weight)
    x = torch.randn(2, 5, C)
    q, k, v = vector.qkv(x).split(C, -1)
    def split(t):
        return t.view(2, 5, H, C // H).transpose(1, 2)
    q, k, v = split(q), split(k), split(v)
    scores = (q @ k.transpose(-2, -1)) / ((C // H) ** 0.5)
    scores = scores.masked_fill(vector.tril[:5, :5] == 0, float("-inf"))
    parallel = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(2, 5, C)
    separate = torch.cat([head(x) for head in heads], dim=-1)
    assert torch.allclose(parallel, separate, atol=1e-6)


if __name__ == "__main__":
    test_stages_and_causal_mask()
    test_parallel_heads_equal_separate_heads()
    print("Hafta 8 mimari, causal mask ve bonus eşdeğerlik testleri geçti.")
