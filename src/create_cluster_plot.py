import os
import wandb
import torch
import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from src.models.alexnet import AlexNet

IMAGE_DIM = 227 
NDIM = 6
# BASE_DIR = args.base_dir
# OUTPUT_DIR = args.base_dir + '/alexnet_data_out'
# CHECKPOINT_DIR = OUTPUT_DIR + '/models/{}'.format(args.exp_name)  

def get_model_checkpoint(experiment_name, epoch, model_name, checkpoint_dir):
    """Get the path to a model checkpoint given the experiment name, epoch, and model name."""
    return os.path.join(checkpoint_dir, experiment_name, 'model_epoch_{}.pth'.format(epoch))




#load the checkpoint from a particular epoch
def load_checkpoint(model, epoch, checkpoint_dir, device_id):
    checkpoint_path = os.path.join(checkpoint_dir, 'alexnet_states_e{}.pkl'.format(epoch))
    if os.path.exists(checkpoint_path):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(device_id))
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print('Wrong checkpoint path: {}'.format(checkpoint_path))
    return model

# Create a dataloader given the dataset and batch size
def get_dataloader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
        batch_size=batch_size)

# Create a dataset given the dir
def get_dataset(dir):
    return datasets.ImageFolder(dir, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))

# Take a model and a dataset and store the pre-final layer activations of the model and the labels
def get_activations(model, dataset, batch_size=32, device='cuda'):
    """Get the activations of the model on the dataset."""
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle= False)
    model.eval()
    activations = []
    labels = []
    for data in dataloader:
        # import ipdb; ipdb.set_trace()
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        activations.append(outputs.cpu().numpy())
        labels.append(targets.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    labels = np.concatenate(labels, axis=0)
    return activations, labels

# make a df with labels as index and activations as columns
def make_df(activations, labels):
    """Make a dataframe with the activations and labels."""
    df = pd.DataFrame(activations)
    df.index = labels
    return df
