from src.create_cluster_plot import load_checkpoint
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity



def get_prediction_df(model_type, model_weights_path, epoch, img_dir, concept_number_to_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_type, epoch, model_weights_path, device)
    imageloader = get_image_loader(img_dir)
    _, pre_final_activations = get_model_activtions(model, imageloader, device)
    df = pd.DataFrame(pre_final_activations)
    df.index = [concept_number_to_name[i] for i in range(df.shape[0])]
    return df

def load_model(model_type, epoch, model_weights_path, device):
    if model_type == 'baseline_ce':
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, 86))
        
        exp_name = 'updated_ds_pre_trained_alexnet_ce_ogoptim'
        alexnet = alexnet.to(device)
    elif model_type == 'bce':
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 3000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(3000, 2500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2500, 2057))
        
        exp_name = 'updated_ds_pre_trained_alexnet_bce_adam_0.00001'
        alexnet = alexnet.to(device)
    elif model_type == 'mse':
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, 6))
        
        exp_name = 'updated_ds_pre_trained_alexnet_mse_adam_weighted_0.00001'
        alexnet = alexnet.to(device)
    elif model_type == 'bce_v2':
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, 6))
        exp_name = 'updated_ds_pre_trained_alexnet_bce_v2_adam_0.00001'
        alexnet = alexnet.to(device)
    else:
        raise ValueError('Invalid model type')
    CHECKPOINT_DIR = model_weights_path + '/models/{}'.format(exp_name)  
    alexnet = load_checkpoint(alexnet, epoch, CHECKPOINT_DIR, device_id = 1)
    return alexnet

def get_image_loader(img_dir):
    dataset = datasets.ImageFolder(img_dir, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(227),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    return loader

def get_model_activtions(model, image_loader, device):
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    pre_final_layer_hook = model.classifier[6][-2].register_forward_hook(getActivation('pre_final_layer'))
    model.eval()
    pre_final_activations = []
    final_layer_outputs = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(image_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pre_final_activations.append(activation['pre_final_layer'].cpu().numpy())
            final_layer_outputs.append(output.cpu().numpy())
    pre_final_activations = np.concatenate(pre_final_activations, axis=0)
    final_layer_outputs = np.concatenate(final_layer_outputs, axis=0)
    return final_layer_outputs, pre_final_activations


def model_prediction_on_triplets(triplets_df, model_activations_df):
    human_data = triplets_df.copy()
    triplets_df = triplets_df.loc[:, ~triplets_df.columns.isin(['count', 'total_count', 'accuracy', 'winner'])]
    # for each head, left and right in triplets_df, get the corresponding row from the baseline_ce
    # calculate the cosine similarity between head and left and head and right from baseline_ce
    # compare the cosine similarity between head and left and head and right from baseline_ce
    # if the cosine similarity between head and left is greater than the cosine similarity between head and right, then the winner is left
    # if the cosine similarity between head and left is less than the cosine similarity between head and right, then the winner is right
    # make a new column in triplets_df called 'winner' and store the winner in it
    # the embeddings are the values of the column in baseline_ce corresponding to the concept

    triplets_df['winner'] = ''
    for index, row in triplets_df.iterrows():
        head = row['head']
        left = row['left']
        right = row['right']
        head_embedding = model_activations_df.T[head].to_numpy().reshape(1, -1)
        left_embedding = model_activations_df.T[left].to_numpy().reshape(1, -1)
        right_embedding = model_activations_df.T[right].to_numpy().reshape(1, -1)
        head_left_cosine_similarity = cosine_similarity(head_embedding, left_embedding)
        head_right_cosine_similarity = cosine_similarity(head_embedding, right_embedding)
        if head_left_cosine_similarity > head_right_cosine_similarity:
            triplets_df.at[index, 'winner'] = left
        else:
            triplets_df.at[index, 'winner'] = right
    
    # make a new column in triplets_df called 'correct'. Store 1 if the winner in triplets_df is the same as the winner in human_data
    # and store 0 if the winner in triplets_df is not the same as the winner in human_data
    triplets_df['correct'] = ''
    for index, row in triplets_df.iterrows():
        if row['winner'] == human_data.loc[index, 'winner']:
            triplets_df.at[index, 'correct'] = 1
        else:
            triplets_df.at[index, 'correct'] = 0
    return triplets_df