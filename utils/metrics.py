import torch
import torch.nn as nn

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        acc = accuracy(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += acc * bs
        total_n    += bs
    return total_loss / total_n, total_acc / total_n, total_n
