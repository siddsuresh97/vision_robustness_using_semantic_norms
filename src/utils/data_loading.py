from collections import Counter
import torch
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_train_test_valid_debug_dataloader(train_dataset, val_dataset, test_dataset, debug_dataset, batch_size):
    train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=batch_size)
    print('Dataloader created')
    debug_data_loader = data.DataLoader(
        debug_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=batch_size)
    print('Debug dataloader created')
        # add code to load validation data and create validation dataloader
    val_dataloader = data.DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=batch_size)
    print('Validation dataloader created')


    print('Test dataset created')
    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=batch_size)
    print('Test dataloader created')
    return train_dataloader, val_dataloader, test_dataloader, debug_data_loader


def get_train_test_valid_debug_dataset(IMAGE_DIM, train_img_dir, validation_img_dir, test_img_dir, batch_size):
    train_dataset = datasets.ImageFolder(train_img_dir, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))
    print('train created')
    debug_dataset = torch.utils.data.Subset(train_dataset, list(range(1, batch_size*10)))
    print('lenght of debug dataset is:', len(debug_dataset))
    #import ipdb;ipdb.set_trace()
    print('unique classes are:', len(dict(Counter(debug_dataset.dataset.targets))))
    print(dict(Counter(debug_dataset.dataset.targets)))
    val_dataset = datasets.ImageFolder(validation_img_dir, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))
    print('Validation dataset created')
    # add code to load test data and create test dataloader
    test_dataset = datasets.ImageFolder(test_img_dir, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))
    print('Test dataset created')
    return train_dataset, val_dataset, test_dataset, debug_dataset
