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



def get_prediction_df(model_type, model_weights_path, epoch, img_dir, concept_number_to_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_type, epoch, model_weights_path, device)
    imageloader = get_image_loader(img_dir)
    _, pre_final_activations = get_model_activtions(model, imageloader, device)
    df = pd.DataFrame(pre_final_activations)
    df.index = [concept_number_to_name[i] for i in range(df.shape[0])]
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
    else:
        raise ValueError('Invalid model type')
    CHECKPOINT_DIR = model_weights_path + '/models/{}'.format(exp_name)  
    alexnet = load_checkpoint(alexnet, epoch, CHECKPOINT_DIR, device_id = 1)
    return alexnet