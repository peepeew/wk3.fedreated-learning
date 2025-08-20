from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 数据/模型
    dataset: str = "mnist"         # 先只支持 mnist
    model: str = "mlp"             # 先只支持 mlp
    num_clients: int = 10
    noniid_mode: str = "label"     # label-per-client

    # 训练
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42

    # I/O
    data_root: str = "./data"
    log_dir: str = "./logs"
