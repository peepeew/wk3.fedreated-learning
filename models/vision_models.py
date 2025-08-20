# models/vision_models.py
import math, torch, torch.nn as nn
from lora.wrap import MaybeLoRALinear, freeze_except_lora

# --- MLP (MNIST) ---
class MLP(nn.Module):
    def __init__(self, num_classes=10, use_lora=False, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            MaybeLoRALinear(28*28, 200, use_lora, r, alpha, dropout), nn.ReLU(),
            MaybeLoRALinear(200, 200, use_lora, r, alpha, dropout),   nn.ReLU(),
            MaybeLoRALinear(200, num_classes, use_lora, r, alpha, dropout)
        )
        if use_lora: freeze_except_lora(self)
    def forward(self, x): return self.net(x)

# --- ViT (MNIST) ---
class _PatchEmbed(nn.Module):
    def __init__(self, img=28, patch=7, in_ch=1, dim=128):
        super().__init__()
        assert img%patch==0
        self.np = (img//patch)*(img//patch)
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)        # [B,dim,4,4]
        return x.flatten(2).transpose(1,2)  # [B,16,dim]

class _MHA(nn.Module):
    def __init__(self, dim, nhead, use_lora=False, r=8, alpha=16, dropout=0.0):
        super().__init__(); self.h=nhead; self.dk=dim//nhead
        self.q = MaybeLoRALinear(dim, dim, use_lora, r, alpha, dropout)
        self.k = MaybeLoRALinear(dim, dim, use_lora, r, alpha, dropout)
        self.v = MaybeLoRALinear(dim, dim, use_lora, r, alpha, dropout)
        self.o = MaybeLoRALinear(dim, dim, use_lora, r, alpha, dropout)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B,L,D=x.shape; H=self.h; dh=D//H
        def sp(m): return m(x).view(B,L,H,dh).transpose(1,2)
        q,k,v = sp(self.q), sp(self.k), sp(self.v)
        att = (q @ k.transpose(-2,-1))/math.sqrt(dh)
        att = self.drop(att.softmax(-1))
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,L,D)
        return self.o(out)

class _Block(nn.Module):
    def __init__(self, dim=128, nhead=4, mlp_ratio=4.0, drop=0.0, use_lora=False, r=8, alpha=16):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.att=_MHA(dim,nhead,use_lora,r,alpha,drop)
        self.n2=nn.LayerNorm(dim)
        h=int(dim*mlp_ratio)
        self.fc1=MaybeLoRALinear(dim,h,use_lora,r,alpha,drop)
        self.fc2=MaybeLoRALinear(h,dim,use_lora,r,alpha,drop)
        self.act=nn.GELU(); self.drop=nn.Dropout(drop)
    def forward(self, x):
        x = x + self.att(self.n1(x))
        x = x + self.fc2(self.drop(self.act(self.fc1(self.n2(x)))))
        return x

class ViT(nn.Module):
    def __init__(self, num_classes=10, use_lora=False, r=8, alpha=16, dropout=0.0, dim=128, depth=4, nhead=4):
        super().__init__()
        self.patch = _PatchEmbed(28,7,1,dim); np=self.patch.np
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1,np+1,dim))
        self.blocks = nn.ModuleList([_Block(dim,nhead,4.0,dropout,use_lora,r,alpha) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = MaybeLoRALinear(dim, num_classes, use_lora, r, alpha, dropout)
        nn.init.trunc_normal_(self.pos,std=0.02); nn.init.trunc_normal_(self.cls,std=0.02)
        if use_lora: freeze_except_lora(self)
    def forward(self, x):
        B=x.size(0)
        x = self.patch(x)
        x = torch.cat([self.cls.expand(B,-1,-1), x], dim=1) + self.pos
        for blk in self.blocks: x = blk(x)
        x = self.norm(x[:,0])
        return self.head(x)
