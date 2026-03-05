#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot latent space visualizations of trained VAE models from
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models(Olza et al.)
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Thu Jul 24 17:46:35 2025

@author: alexolza
"""
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
#%%
from utils import load_dataset
from components.generators import VAE
from config_files.traditional_decnef_n_instances import traditional_decnef_n_instances_parser
from visualization.plotting import visual_eval_vae
#%%
"""
Configuration variables
"""
global_random_seed = 42
config = traditional_decnef_n_instances_parser()
ext = 'pdf'

device = 'cuda'
outpath = f'../EXPERIMENTS/{config.EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{config.EXP_NAME}/weights/'
figpath =  f'../EXPERIMENTS/{config.EXP_NAME}/figures/generator_eval'
generator_name = f'{config.generator_name}_Z{config.z_dim}_BS{config.generator_batch_size}_E{config.generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)
if not os.path.exists(figpath): os.makedirs(figpath)

z_dim = config.z_dim
generator_batch_size= config.generator_batch_size
generator_epochs = config.generator_epochs

dataset = config.dataset
transform = transforms.Compose([transforms.ToTensor()]) if dataset=='FASHION' else None
trainset = load_dataset(dataset, transform, config.npz_file_path, train= True)
testset = load_dataset(dataset, transform, config.npz_file_path, train= False)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
    
class_names, class_numbers = np.array([[k,v] for k,v in class_name_dict.items()]).T
class_numbers = class_numbers.astype(int)
tabular = True if 'fMRI' in dataset else False
n_features = trainset[0][0].shape[-1]
#%%
vae = VAE(z_dim=z_dim, tabular= tabular, n_features=n_features).to(device)
vae.load(generator_fname+'.pt')
vae_history = vae.history_to_df()
print(f'Loaded {generator_fname}')

for dl, mode in zip([train_loader, test_loader],['train','test']):
    fig_loss, rec, prot, latent_vis_umap, latent_vis_tsne, latent_trav = visual_eval_vae(vae, vae_history, z_dim,
                    dl, class_names,
                    class_numbers, device='cuda')

    
    fnames = [f'{figpath}/{figname}_{generator_name}_{mode}.{ext}'
              for figname in ['loss', 'rec', 'prototypes',
                              'latent_vis_umap', 'latent_vis_tsne', 'latent_trav']]
    
    for fig, fname in zip([fig_loss, rec, prot,
                           latent_vis_umap, latent_vis_tsne, latent_trav],
                          fnames):
        try:
            fig.tight_layout()
            fig.savefig(fname)
            print('saved: ', fname)
        except:
            print(f'Skipped {fname}')
        


