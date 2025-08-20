import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

def mnist_label_per_client(train_ds, num_clients=10):
    labels = np.array(train_ds.targets)
    idxs_by = {d: np.where(labels == d)[0].tolist() for d in range(10)}
    return {cid: idxs_by[cid % 10] for cid in range(num_clients)}

def make_mnist_clients(train_ds, batch_size, num_clients=10, val_ratio=0.1):
    mapping = mnist_label_per_client(train_ds, num_clients)
    clients = []
    for cid in range(num_clients):
        idxs = mapping[cid]
        sub  = Subset(train_ds, idxs)
        n_val = max(1, int(len(sub) * val_ratio))
        val, train = random_split(sub, [n_val, len(sub) - n_val])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=2)
        clients.append((cid, train_loader, val_loader, len(sub)))
    return clients

def make_mnist_test_loader(test_ds, batch_size):
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
