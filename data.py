# Code from https://github.com/Piyush-555/VCL-in-PyTorch/blob/main/dataset.py

import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from models import DINO

# Custom dataset with caching so we only need to run DINOv2 once per image
class mnist_with_dino(torchvision.datasets.MNIST):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.train = train
        self.dino_backbone = DINO().cuda()

    def __getitem__(self, index):
        cache_dir = 'data/mnist_dino_cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f'smallDINO_{"train" if self.train else "Test"}_{index}.npz'

        if not os.path.exists(os.path.join(cache_dir, cache_name)):
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # Pass image through DINO
            cls_token = self.dino_backbone(img.unsqueeze(0).cuda())
            cls_token = cls_token.cpu().numpy()
            np.savez(os.path.join(cache_dir, cache_name), cls_token=cls_token, target=target)
        else:
            data = np.load(os.path.join(cache_dir, cache_name))
            cls_token = data['cls_token']
            target = data['target']

        return cls_token, target

################## SplitMNIST ##################
def _extract_class_specific_idx(dataset, target_classes):
    """
    dataset: torchvision.datasets.MNIST
    target_classes: list
    """
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for target in target_classes:
        idx = idx | (dataset.targets==target)
    
    return idx


def get_split_dataloaders(class_distribution, batch_size=256, dino_transform=False):
    """
    class_distribution: list[list]
    """
    rsz = 28

    transform = transforms.Compose([
        transforms.Resize((rsz, rsz)),
        transforms.ToTensor(),
    ])

    if dino_transform:
        trainset = mnist_with_dino(root="./data", train=True, download=True, transform=transform)
        testset = mnist_with_dino(root="./data", train=False, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.MNIST
        trainset = dataset(root="./data", train=True, download=True, transform=transform)
        testset = dataset(root="./data", train=False, download=True, transform=transform)


    dataloaders = []

    for classes in class_distribution:
        train_idx = _extract_class_specific_idx(trainset, classes)
        train_idx = torch.where(train_idx)[0]
        sub_train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        sub_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=sub_train_sampler)

        test_idx = _extract_class_specific_idx(testset, classes)
        test_idx = torch.where(test_idx)[0]
        sub_test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
        sub_test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=sub_test_sampler)
        
        dataloaders.append((sub_train_loader, sub_test_loader))

    return dataloaders


################## PermutedMNIST ##################
def get_permuted_dataloaders(num_tasks, batch_size=256, dino_transform=False):
    if dino_transform:
        dataset = mnist_with_dino
    else:
        dataset = torchvision.datasets.MNIST

    dataloaders = []

    for task in range(num_tasks):
        if task == 0:
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])
        else:
            rng_permute = np.random.RandomState(task)
            idx_permute = torch.from_numpy(rng_permute.permutation(784))
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x, idx=idx_permute: x.view(-1)[idx].view(1, 28, 28))
            ])

        sub_trainset = dataset(root="./data", train=True, download=True, transform=transform)
        sub_train_sampler = torch.utils.data.RandomSampler(sub_trainset)
        sub_train_loader = torch.utils.data.DataLoader(
            sub_trainset, batch_size=batch_size, sampler=sub_train_sampler)

        sub_testset = dataset(root="./data", train=False, download=True, transform=transform)
        sub_test_sampler = torch.utils.data.RandomSampler(sub_testset)
        sub_test_loader = torch.utils.data.DataLoader(
            sub_testset, batch_size=batch_size, sampler=sub_test_sampler)

        dataloaders.append((sub_train_loader, sub_test_loader))

    return dataloaders