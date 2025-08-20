# clients/client.py
import copy
import torch
import torch.nn as nn
from utils.metrics import evaluate

__all__ = ["Client"]

class Client:
    def __init__(self, cid: int, train_loader, val_loader, n_samples: int):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.n_samples    = n_samples

    def local_train(
        self,
        global_model: nn.Module,
        device: str,
        lr: float,
        wd: float,
        epochs: int,
        use_amp: bool = False,
    ):
        model = copy.deepcopy(global_model).to(device)
        model.train()
        opt  = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        crit = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

        tot_loss, tot_correct, tot_n = 0.0, 0.0, 0
        for _ in range(epochs):
            for x, y in self.train_loader:
                if device == "cuda":
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                else:
                    x = x.to(device); y = y.to(device)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                    logits = model(x)
                    loss = crit(logits, y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                bs = y.size(0)
                tot_loss   += loss.item() * bs
                tot_correct += (logits.argmax(1) == y).float().sum().item()
                tot_n      += bs

        train_loss = tot_loss / max(1, tot_n)
        train_acc  = tot_correct / max(1, tot_n)

        val_loss, val_acc, _ = evaluate(model, self.val_loader, device)
        return model.state_dict(), self.n_samples, train_loss, train_acc, val_loss, val_acc
