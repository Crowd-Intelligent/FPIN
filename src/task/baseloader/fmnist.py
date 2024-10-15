import torch
from torchvision import datasets, transforms


def fmnist_dataset(data_dir):
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                          transform=apply_transform)

    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                         transform=apply_transform)
    return train_dataset, test_dataset
