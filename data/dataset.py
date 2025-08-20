from torchvision import datasets, transforms

def get_mnist(data_root: str):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(root=data_root, train=True,  download=True, transform=tfm)
    test  = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    return train, test
