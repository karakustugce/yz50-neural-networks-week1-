"""
GPT-2 (124M) from-scratch architecture, HF weight mapping, sampling and sanity checks.
Run: python hafta10.py --mode all
"""
import argparse, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3*c.n_embd)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd)
        self.n_head, self.n_embd = c.n_head, c.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(c.block_size, c.block_size)).view(1,1,c.block_size,c.block_size))
    def forward(self, x):
        b,t,ch=x.shape
        q,k,v=self.c_attn(x).split(self.n_embd, dim=2)
        q=q.view(b,t,self.n_head,ch//self.n_head).transpose(1,2)
        k=k.view(b,t,self.n_head,ch//self.n_head).transpose(1,2)
        v=v.view(b,t,self.n_head,ch//self.n_head).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y=y.transpose(1,2).contiguous().view(b,t,ch)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c_fc=nn.Linear(c.n_embd,4*c.n_embd)
        self.c_proj=nn.Linear(4*c.n_embd,c.n_embd)
    def forward(self,x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ln_1=LayerNorm(c.n_embd); self.attn=CausalSelfAttention(c)
        self.ln_2=LayerNorm(c.n_embd); self.mlp=MLP(c)
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        return x+self.mlp(self.ln_2(x))

class Config:
    vocab_size=50257; block_size=1024; n_layer=12; n_head=12; n_embd=768

class GPT2(nn.Module):
    def __init__(self):
        super().__init__(); c=Config()
        self.wte=nn.Embedding(c.vocab_size,c.n_embd)
        self.wpe=nn.Embedding(c.block_size,c.n_embd)
        self.drop=nn.Dropout(0.0)
        self.h=nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f=LayerNorm(c.n_embd)
        self.lm_head=nn.Linear(c.n_embd,c.vocab_size,bias=False)
        self.lm_head.weight=self.wte.weight
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>Config.block_size: raise ValueError("sequence exceeds block_size")
        pos=torch.arange(t,device=idx.device)
        x=self.drop(self.wte(idx)+self.wpe(pos))
        for block in self.h: x=block(x)
        logits=self.lm_head(self.ln_f(x))
        loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) if targets is not None else None
        return logits,loss

@torch.no_grad()
def generate(model, idx, max_new_tokens=80, top_k=40, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        logits,_=model(idx[:,-Config.block_size:])
        logits=logits[:,-1,:]/temperature
        if top_k:
            values,_=torch.topk(logits,min(top_k,logits.size(-1)))
            logits[logits<values[:,-1,None]]=-float("inf")
        probs=F.softmax(logits,dim=-1)
        idx=torch.cat([idx,torch.multinomial(probs,1)],dim=1)
    return idx

def load_hf(model):
    from transformers import GPT2LMHeadModel
    hf=GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    sd=hf.state_dict()
    own=model.state_dict()
    # HF Conv1D stores [in,out]; torch Linear stores [out,in], so transpose those matrices.
    mapping={"wte.weight":"transformer.wte.weight","wpe.weight":"transformer.wpe.weight",
             "ln_f.weight":"transformer.ln_f.weight","ln_f.bias":"transformer.ln_f.bias"}
    for a,b in mapping.items(): own[a].copy_(sd[b])
    for i in range(Config.n_layer):
        for ours,hfs in [(f"h.{i}.ln_1","transformer.h.%d.ln_1"%i),
                         (f"h.{i}.ln_2","transformer.h.%d.ln_2"%i)]:
            own[ours+".weight"].copy_(sd[hfs+".weight"]); own[ours+".bias"].copy_(sd[hfs+".bias"])
        pairs=[("attn.c_attn","attn.c_attn"),("attn.c_proj","attn.c_proj"),
               ("mlp.c_fc","mlp.c_fc"),("mlp.c_proj","mlp.c_proj")]
        for o,h in pairs:
            for suffix in ("weight","bias"):
                key=f"h.{i}.{o}.{suffix}"
                val=sd[f"transformer.h.{i}.{h}.{suffix}"]
                own[key].copy_(val.T if suffix=="weight" else val)
    model.load_state_dict(own)
    return hf

def report(model):
    print("Model parameters (tied wte/lm_head counted once):")
    for n,p in model.named_parameters(): print(f"{n:42s} {str(tuple(p.shape)):22s} {p.numel():,}")
    print(f"TOTAL: {sum(p.numel() for p in model.parameters()):,}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["all","inspect","compare","sample","train"],default="all")
    ap.add_argument("--steps",type=int,default=300)
    ap.add_argument("--device",default="auto")
    a=ap.parse_args()
    device=("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if a.device=="auto" else a.device
    print("device:",device)
    model=GPT2().to(device)
    if a.mode in ("all","inspect","compare","sample"):
        hf=load_hf(model)
        report(model)
        try:
            from transformers import GPT2Tokenizer
            tok=GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        except Exception as e: print("Tokenizer unavailable:",e); return
        if a.mode in ("all","inspect"):
            print("\nPosition embedding plot: results/wpe.png")
            import matplotlib.pyplot as plt
            os.makedirs("results",exist_ok=True)
            plt.figure(figsize=(12,5)); plt.imshow(model.wpe.weight.detach().cpu().numpy().T,aspect="auto",cmap="coolwarm")
            plt.colorbar(); plt.xlabel("position"); plt.ylabel("embedding dimension"); plt.title("GPT-2 learned position embeddings")
            plt.tight_layout(); plt.savefig("results/wpe.png",dpi=160); plt.close()
        if a.mode in ("all","compare"):
            x=tok("The model learns patterns.",return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                y,_=model(x); yh=hf(x).logits
            print("HF/custom max |logit difference|:",(y-yh).abs().max().item())
            assert torch.allclose(y,yh,atol=2e-4,rtol=2e-4)
        if a.mode in ("all","sample"):
            for prompt in ["Once upon a time","Türkiye'de yapay zeka"]:
                ids=tok(prompt,return_tensors="pt").input_ids.to(device)
                print(f"\nPROMPT: {prompt}\n"+tok.decode(generate(model,ids,60,40)[0],skip_special_tokens=True))
                print("Turkish token IDs:",tok.encode("Türkiye'de yapay zeka"),"count:",len(tok.encode("Türkiye'de yapay zeka")))
        if a.mode!="all": return
    # Random initialization: uniform logits imply cross-entropy log(vocab).
    model=GPT2().to(device)
    torch.manual_seed(42)
    x=torch.randint(0,Config.vocab_size,(2,32),device=device); y=torch.randint(0,Config.vocab_size,(2,32),device=device)
    _,loss=model(x,y); print(f"Random-init loss={loss.item():.4f}; ln(50257)={math.log(50257):.4f}")
    # One-batch overfit check, intentionally tiny context/batch for CPU compatibility.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    start=time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True); _,loss=model(x,y); loss.backward(); opt.step()
        if step%50==0 or step==a.steps-1: print(f"overfit step {step:4d} loss={loss.item():.6f}")
    print(f"elapsed={time.time()-start:.1f}s")
if __name__=="__main__": main()
