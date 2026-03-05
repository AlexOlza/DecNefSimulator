#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:12:06 2025

@author: alexolza
"""

import sys
sys.path.append('..')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
###########################
from utils import load_dataset
from components.generators import VAE
from config_files.traditional_decnef_n_instances import traditional_decnef_n_instances_parser
#%%
"""
Configuration variables
"""
global_random_seed = 42
config = traditional_decnef_n_instances_parser()

device = config.device
outpath = f'../EXPERIMENTS/{config.EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{config.EXP_NAME}/weights/'

z_dim = config.z_dim 
generator_epochs = config.generator_epochs  
generator_batch_size= 64

generator_name = f'{config.generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)
print(f'generator_fname={generator_name}')
#%% 
for p in [outpath, modelpath]:
    if not os.path.exists(p):
        os.makedirs(p)
#%%
dataset = config.dataset
transform = transforms.Compose([transforms.ToTensor()]) if dataset=='FASHION' else None
tabular = True if 'fMRI' in dataset else False
trainset = load_dataset(dataset, transform, config.npz_file_path, train= True)
train_loader = DataLoader(trainset, batch_size=generator_batch_size, shuffle=True)

if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
    
class_names, class_numbers = np.array([[k,v] for k,v in class_name_dict.items()]).T
class_numbers = class_numbers.astype(int)
print(trainset[0][0].shape)
sample_size = len(trainset)
n_features = trainset[0][0].shape[-1]
print(f'Features: {n_features}, training sample size: {sample_size}')
#%%
for batch, step in train_loader:
    print(batch.shape)
    print(step.shape)
    print(step[0])
    break

if not os.path.exists(generator_fname+'.pt'):
    vae = VAE(z_dim=z_dim, tabular= tabular, n_features=n_features).to(device)
    vae.fit(train_loader, generator_epochs, verbose=1)
    vae.compute_prototypes(train_loader)
    vae_history = vae.history_to_df()
    print(f'{generator_name} TRAINING FINISHED WITH z_dim=',z_dim)
    vae.save(generator_fname+'.pt')
    print(f'Saved {generator_fname}')
else:
    print(f'Found {generator_fname}')
    vae = VAE(z_dim=z_dim,tabular=tabular, n_features=n_features, device= device).to(device)
    vae.load(generator_fname+'.pt')
    vae.device = device
    vae_history = vae.history_to_df()

print(vae)
print('\n')
print('vae.target_size= ',vae.target_size)
print('\n')
vae.eval()
latent_prototype = vae.prototypes[config.target_class_idx][0] # [1] is the variance and [0] is the mu

print(f'latent_prototype shape is {latent_prototype.shape}')
prototype = vae.decoder(torch.Tensor(latent_prototype).to(vae.device),
                                            vae.target_size).detach()

print(f'prototype shape is {prototype.shape}')
