# wk3.fedreated-learning
This repository implements a minimal runnable Federated Learning framework (FedAvg). Currently, it supports: - Dataset: MNIST - Models: MLP (enabled by default); ViT (implemented in `models/vision_models.py`; toggle with a single line change) - LoRA: One-click toggle (only affects layers using `MaybeLoRALinear`)
## Project Structure (with responsibilities)
├─ clients/
│ ├─ init.py
│ ├─ client.py # Client class: local training loop (averaged metrics, optional AMP), returns weights & stats
│ └─ federated_train.py # Entry point: data/client creation, FedAvg, evaluation & saving
│
├─ configs/
│ ├─ init.py
│ └─ config.py # TrainConfig (rounds, batch size, LR, log dir, seed, etc.)
│
├─ data/
│ ├─ init.py
│ ├─ dataset.py # get_mnist(): download/load MNIST
│ └─ dataloader.py # DataLoaders:
│ # - make_mnist_clients(): Non-IID (label-per-client) partition + train/val split
│ # - make_mnist_test_loader(): global test loader
│
├─ models/
│ ├─ init.py
│ └─ vision_models.py # Models:
│ # - MLP (784→200→200→10) for MNIST (LoRA-capable)
│ # - Lightweight ViT (patch embed + transformer blocks; Q/K/V/FC are LoRA-capable)
│
├─ lora/
│ ├─ init.py # empty marker so Python treats this as a package
│ └─ wrap.py # LoRA utilities:
│ # - MaybeLoRALinear: lora.Linear when use_lora=True (else nn.Linear)
│ # - freeze_except_lora: freeze base; only LoRA params train
│
├─ utils/
│ ├─ init.py
│ └─ metrics.py # evaluate(model, loader, device) → loss/acc (+ details if needed)
│
├─ logs/ # Generated at runtime (CSV/plots if you add them)
│ ├─ global_metrics.csv # Per round: round,test_loss,test_acc
│ └─ client_metrics.csv # Per round per client: round,client,n_samples,train/val loss/acc
│
├─ artifacts/ # Generated at runtime
│ └─ global_round_{r}.pt # Aggregated global state_dict after round r
│
├─ README.md
└─ requirements.txt
## Features

- **Federated averaging (FedAvg)** with a fixed number of clients (default 10)
- **Non-IID label-per-client** partitioning for MNIST
- **LoRA** support on all linear layers inside MLP/ViT via `MaybeLoRALinear`
- **Per-round metrics**: logs global test metrics and each client’s local train/val metrics
- **Checkpointing**: saves aggregated **global** model weights per round


## (Optional) Enable GPU with CUDA
pip uninstall -y torch torchvision torchaudio
pip cache purge
# Choose one set based on your driver/toolkit
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# or (older drivers)
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
Verify:
python -c "import torch;print('cuda_available=',torch.cuda.is_available());\
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

## How to Run
# A) Non-LoRA (full-parameter training)
python -m clients.federated_train --rounds 5 --batch-size 64 --lr 1e-3 --local-epochs 1
# B) LoRA (train only LoRA adapters)
Install once:
pip install loralib

python -m clients.federated_train --use-lora 1 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --rounds 5 --batch-size 64 --lr 1e-3 --local-epochs 1


