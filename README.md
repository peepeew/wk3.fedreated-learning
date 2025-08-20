# wk3.fedreated-learning
This repository implements a minimal runnable Federated Learning framework (FedAvg). Currently, it supports: - Dataset: MNIST - Models: MLP (enabled by default); ViT (implemented in `models/vision_models.py`; toggle with a single line change) - LoRA: One-click toggle (only affects layers using `MaybeLoRALinear`)

## Module Responsibilities & Call Flow

- **clients/federated_train.py**  
  1) Parse CLI args (rounds / batch size / learning rate / LoRA on/off, etc.)  
  2) Load MNIST and build **label-per-client** partitions for 10 clients  
  3) Construct the global model `MLP(...)` (or manually switch to `ViT(...)`)  
  4) Evaluate on the global test set at Round 0  
  5) For each round:  
     - Sequentially call each `Client.local_train()` → collect local weights & metrics  
     - Aggregate with **FedAvg** to produce the new global weights  
     - Evaluate on the global test set and log results  
     - Save `artifacts/global_round_{r}.pt`  
  6) Append all metrics to `logs/*.csv`

- **clients/client.py**  
  - `Client.local_train(...)`:  
    - Clone the global model and train locally for a few epochs  
    - Optimizer uses only `requires_grad=True` params: **LoRA=1 → train LoRA adapters only; LoRA=0 → full fine-tuning**  
    - Training metrics are **averaged over the whole local epoch(s)** (not the last batch)  
    - Optional CUDA AMP supported: `torch.amp.autocast('cuda', ...)`  
    - Returns: `(state_dict, n_samples, train_loss, train_acc, val_loss, val_acc)`

- **models/vision_models.py**  
  - `MLP`: MNIST baseline (Flatten → 200 → 200 → 10); all Linear layers wrapped by `MaybeLoRALinear`  
  - `ViT` (lightweight): PatchEmbed (28×28 → 4×4 patches) + stacked Blocks (MHA + FFN; Q/K/V/O and FC1/FC2 are LoRA-capable)  
  - If `use_lora=True`, the constructor calls `freeze_except_lora(self)` to freeze the base weights and train LoRA only

- **data/dataloader.py**  
  - `make_mnist_clients(...)`: **Non-IID (label-per-client)** split; each client’s `train/val` is drawn from its single-class subset  
  - **Tip:** For more meaningful validation, have all clients share a global validation set (replace each client’s `val_loader` with the global test loader in `federated_train.py`)

- **lora/wrap.py**  
  - `MaybeLoRALinear`: uses `lora.Linear` when `use_lora=True` and `loralib` is installed; otherwise falls back to `nn.Linear`  
  - `freeze_except_lora`: keeps only LoRA parameters trainable (parameter names containing `lora_`)


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


