import os, random, numpy as np, torch, argparse
import csv
from configs.config import TrainConfig
from data.dataset import get_mnist
from data.dataloader import make_mnist_clients, make_mnist_test_loader
from models.vision_models import MLP
from .client import Client
from utils.metrics import evaluate

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def fedavg(states_and_sizes, global_state):
    total = sum(n for _, n in states_and_sizes)
    new_state = {k: torch.zeros_like(v) for k, v in global_state.items()}
    for state, n in states_and_sizes:
        w = n / total
        for k, v in state.items():
            new_state[k] += v * w
    return new_state

def parse_args():
    p = argparse.ArgumentParser()
    # 训练超参
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # ↓↓↓ 新增：LoRA 开关和超参（仅作用于 MLP）
    p.add_argument("--use-lora", type=int, default=0, help="1 开启 LoRA，0 关闭（默认）")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def csv_writer(path: str, header: list[str]):
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def csv_append(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def main():
    args = parse_args()
    cfg = TrainConfig(rounds=args.rounds, batch_size=args.batch_size, lr=args.lr, local_epochs=args.local_epochs)
    device = args.device
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed(cfg.seed)

    # === 日志与产出路径 ===
    log_dir = cfg.log_dir
    art_dir = "./artifacts"
    ensure_dir(log_dir)
    ensure_dir(art_dir)
    gcsv = os.path.join(log_dir, "global_metrics.csv")
    ccsv = os.path.join(log_dir, "client_metrics.csv")
    csv_writer(gcsv, ["round", "test_loss", "test_acc"])
    csv_writer(ccsv, ["round", "client", "n_samples", "train_loss", "train_acc", "val_loss", "val_acc"])

    # 1) 数据（MNIST）
    train_ds, test_ds = get_mnist(cfg.data_root)
    clients_raw = make_mnist_clients(train_ds, cfg.batch_size, cfg.num_clients)
    test_loader = make_mnist_test_loader(test_ds, cfg.batch_size)
    print(f"Using {len(clients_raw)} clients. Training for {cfg.rounds} rounds. LoRA={bool(args.use_lora)}")

    # 2) 模型（全局）—— 现在支持 LoRA/非 LoRA 两种 MLP
    global_model = MLP(
        use_lora=bool(args.use_lora),
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    ).to(device)

    # 3) Round 0：全局评估（并记录）
    tl, ta, _ = evaluate(global_model, test_loader, device)
    print(f"[Round 0] Global test: loss={tl:.4f} acc={ta:.4f}")
    csv_append(gcsv, [0, tl, ta])

    # 4) 联邦轮次
    for r in range(1, cfg.rounds + 1):
        states_and_sizes = []
        for (cid, tr_loader, val_loader, n) in clients_raw:
            cli = Client(cid, tr_loader, val_loader, n)
            sd, ns, trl, tra, vl, va = cli.local_train(
                global_model, device, cfg.lr, cfg.weight_decay, cfg.local_epochs
            )
            states_and_sizes.append((sd, ns))
            # 记录每客户端指标
            csv_append(ccsv, [r, cid, ns, trl, tra, vl, va])
            print(f"[Round {r}] Client {cid}: train_loss={trl:.4f} train_acc={tra:.4f} | val_loss={vl:.4f} val_acc={va:.4f}")

        # FedAvg 聚合 + 全局评估（并记录 + 保存权重）
        new_state = fedavg(states_and_sizes, global_model.state_dict())
        global_model.load_state_dict(new_state)
        tl, ta, _ = evaluate(global_model, test_loader, device)
        print(f"[Round {r}] Global test: loss={tl:.4f} acc={ta:.4f}")
        csv_append(gcsv, [r, tl, ta])
        torch.save(global_model.state_dict(), os.path.join(art_dir, f"global_round_{r}.pt"))

if __name__ == "__main__":
    main()
