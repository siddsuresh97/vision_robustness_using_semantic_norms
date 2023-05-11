from src.create_cluster_plot import load_checkpoint
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.vss_helper_functions import *



def get_prediction_df(model_type, model_weights_path, epoch, img_dir, concept_number_to_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_type, epoch, model_weights_path, device)
    imageloader = get_image_loader(img_dir)
    _, pre_final_activations = get_model_activtions(model, imageloader, device, model_type)

    target_names = imageloader.dataset.classes
    # Define the mapping from label indices to class names
    idx_to_class = {i: target_names[i] for i in range(len(target_names))}


    df = pd.DataFrame(pre_final_activations)
    if concept_number_to_name is not None:
        df.index = [concept_number_to_name[i] for i in range(df.shape[0])]
    else:
        # df.index = [i for i in list(imageloader.dataset.targets)]
        df.index = [idx_to_class[label] for label in list(imageloader.dataset.targets)]
    return df

def load_model(model_type, epoch, model_weights_path, device):
    if model_type == 'baseline_ce_pt':
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, 86))
        
        exp_name = 'pre_trained_ce'
        alexnet = alexnet.to(device)
    elif model_type == 'baseline_ce_scratch':
        alexnet = models.alexnet(pretrained=False)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, 86))
        
        exp_name = 'scratch_alexnet_ce_adam_og'
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
        exp_name = 'pt_alexnet_bce_0.0006621068753189949'
        alexnet = alexnet.to(device)
    elif model_type == 'bce_data_augmentation_pt':
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, 2057))
        exp_name = 'pt_alexnet_bce_0.0006621068753189949_data_augmentation'
        alexnet = alexnet.to(device)

    elif model_type == 'ce_data_augmentation_scratch':
        alexnet = models.alexnet(pretrained=False)
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, 86))
        exp_name = 'scratch_alexnet_ce_adam_og_aug'
        alexnet = alexnet.to(device)
    elif 'pt' in model_type:
        alexnet = models.alexnet(pretrained=True)
        if 'bce' in model_type:
            alexnet.classifier[-1] = nn.Sequential(nn.Linear(4096, 2000),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(2000, 500),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(500, 100),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(100, 2057))
        elif 'ce' in model_type:
            alexnet.classifier[-1] = nn.Sequential(nn.Linear(4096, 2000),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(2000, 500),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(500, 100),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(100, 86))
        elif 'hybrid' in model_type:
            alexnet.classifier[-1] = nn.Sequential(nn.Linear(4096, 2000),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(2000, 500),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(500, 100),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(100, 2057))
            alexnet = nn.Sequential(alexnet, nn.Sequential(nn.ReLU(inplace=True), nn.Linear(2057, 86)))
        else:
            raise ValueError('model_type not recognized')
        alexnet = alexnet.to(device)
    elif 'scratch' in model_type:
        alexnet = models.alexnet(pretrained=False)
        if 'bce' in model_type:
            alexnet.classifier[-1] = nn.Sequential(nn.Linear(4096, 2000),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(2000, 500),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(500, 100),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(100, 2057))
        elif 'ce' in model_type:
            alexnet.classifier[-1] = nn.Sequential(nn.Linear(4096, 2000),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(2000, 500),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(500, 100),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(100, 86))
        elif 'hybrid' in model_type:
            alexnet.classifier[-1] = nn.Sequential(nn.Linear(4096, 2000),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(2000, 500),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(500, 100),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(100, 2057))
            alexnet = nn.Sequential(alexnet, nn.Sequential(nn.ReLU(inplace=True), nn.Linear(2057, 86)))
        else:
            raise ValueError('model_type not recognized')
        alexnet = alexnet.to(device)

    else:
        raise ValueError('Invalid model type')
    # CHECKPOINT_DIR = model_weights_path + '/models/{}'.format(exp_name)  
    # alexnet = load_checkpoint(alexnet, epoch, CHECKPOINT_DIR=checkpoint_dir, device_id = 1)
    alexnet = load_checkpoint(alexnet, checkpoint_dir = model_weights_path,device_id=1, config_path = model_type)
    return alexnet