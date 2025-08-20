# lora/wrap.py
import torch.nn as nn
try:
    import loralib as lora
except Exception:
    lora = None

class MaybeLoRALinear(nn.Module):
    def __init__(self, in_f, out_f, use_lora=False, r=8, alpha=16, dropout=0.0, bias=True):
        super().__init__()
        if use_lora and lora is not None:
            self.layer = lora.Linear(in_f, out_f, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=bias)
        else:
            if use_lora and lora is None:
                print("⚠️ loralib 未安装，已回退为 nn.Linear（未启用 LoRA）")
            self.layer = nn.Linear(in_f, out_f, bias=bias)
    def forward(self, x): return self.layer(x)

def freeze_except_lora(module: nn.Module):
    has_lora = False
    for m in module.modules():
        if lora is not None and isinstance(m, lora.Linear):
            has_lora = True; break
    if not has_lora: return
    for n, p in module.named_parameters():
        p.requires_grad = ('lora_' in n)
