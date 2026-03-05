#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run experiments for
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Wed Jul  2 12:06:50 2025

@author: alexolza
"""

import sys
sys.path.append('..')
import torch
import os
from tqdm import tqdm
import regex as re
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
###########################
from components.generators import VAE
from protocols.decnef_loops import compute_single_trajectory
from components.classifiers import ElasticNetLinearClassification, CNNClassification, BinaryDataLoader
from components.update_rules import powsig, update_z_moving_normal_drift_adaptive_variance, update_z_moving_normal_drift_adaptive_variance_memory
from config_files.traditional_decnef_n_instances import print_config_to_fname, traditional_decnef_n_instances_parser
from utils import make_init_z_lattice, load_dataset
#%%
"""
Configuration variables
"""
global_random_seed = 42
p_scale_func = powsig
config = traditional_decnef_n_instances_parser()

EXP_NAME = config.EXP_NAME
subject = config.subject
if subject==0:
    figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/'
    outpath = f'../EXPERIMENTS/{EXP_NAME}/output/'
    modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/'
else:
    figpath = f'../EXPERIMENTS/{EXP_NAME}/{subject}/figures/'
    outpath = f'../EXPERIMENTS/{EXP_NAME}/{subject}/output/'
    modelpath = f'../EXPERIMENTS/{EXP_NAME}/{subject}/weights/'


genfigpath = figpath+'generator_eval'
disfigpath = figpath+'classifier_eval'
nfbfigpath = figpath+'nfb_eval'
update_rules = [update_z_moving_normal_drift_adaptive_variance, 
                update_z_moving_normal_drift_adaptive_variance_memory
                ]
update_rule_names = ['MNDAV', 'MNDAVMem']

device = config.device
z_dim = config.z_dim
lambda_ = 1/config.lambda_inv # Trust-in-feedback parameter. A common value could be 0.2 which is 1/5
generator_epochs = config.generator_epochs
generator_batch_size=config.generator_batch_size
n_trajectories_per_init = config.n_trajectories_per_init
classifier_epochs = config.classifier_epochs
classifier_batch_size = config.classifier_batch_size
tgt_non_tgt = [config.target_class_idx, config.non_target_class_idx]

generator_name = f'{config.generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

print(f'Shape of seed list: {np.array(config.seed_list).shape}')
flat_seedlist = [x for sublist in config.seed_list for x in sublist]
assert len(set(flat_seedlist)) == n_trajectories_per_init*config.n_trajectories
#%%
for p in [figpath, outpath, 
          modelpath]:
    if not os.path.exists(p):
        os.makedirs(p)
     
#%%
transform = transforms.Compose([transforms.ToTensor()]) if config.dataset=='FASHION' else None

trainset = load_dataset(config.dataset, transform, train=True, npz_file_path = config.npz_file_path)
testset = load_dataset(config.dataset, transform, train=False, npz_file_path = config.npz_file_path)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True
                          )

if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
class_name_dict_reverse = {v: k for k, v in class_name_dict.items()} 
combo_names = [class_name_dict_reverse[i] for i in tgt_non_tgt]
    
class_names, class_numbers = np.array([[k,v] for k,v in class_name_dict.items()]).T
class_numbers = class_numbers.astype(int)
print(trainset[0][0].shape)
sample_size = len(trainset)
n_features = trainset[0][0].shape[-1]
def check(t, name):
    if torch.isnan(t).any():
        print("NaN in", name)
        raise ValueError()

print(f'Features: {n_features}, training sample size: {sample_size}')
#%%
tabular= True if 'fMRI' in config.dataset else False
if not os.path.exists(generator_fname+'.pt'):
    print(f'Tabular: {tabular}')
    vae = VAE(z_dim=z_dim, tabular=tabular, n_features=n_features, device= device).to(device)
    vae.fit(train_loader, generator_epochs, generator_batch_size)
    vae.compute_prototypes(train_loader)
    vae_history = vae.history_to_df()
    print(f'{generator_name} TRAINING FINISHED WITH z_dim=',z_dim)
    #%%
    vae.save(generator_fname+'.pt')
else:
    vae = VAE(z_dim=z_dim,tabular=tabular, n_features=n_features, device= device).to(device)
    vae.load(generator_fname+'.pt')
    vae.device = device
    vae_history = vae.history_to_df()
    print(f'Loaded {generator_fname}')

all_class_prototypes = np.vstack([prot[0].ravel() for idx, prot in vae.prototypes.items()
                                   ])
all_class_prototypes_sigma = np.vstack([prot[1].ravel() for idx, prot in vae.prototypes.items()
                                   ])
update_rule_func, update_rule_name = update_rules[config.update_rule_idx], update_rule_names[config.update_rule_idx]

#%%

discr_str = f'{combo_names[0]} vs {combo_names[1]}'
clean_discr_str = re.sub('[^a-zA-Z0-9]','', discr_str)
classifier_type =  'ELASTICNET' if config.dataset.startswith('synth_fMRI') else 'CNN'
classifier_name = f'{classifier_type}_{clean_discr_str}__BS{classifier_batch_size}_E{classifier_epochs}'
classifier_fname = os.path.join(modelpath, classifier_name+'.pt')

trajectory_dir = os.path.join(outpath,f'TRAJS_{generator_name}_{classifier_name}',f'linv{config.lambda_inv}',f'UR{update_rule_name}',f'IGDIS{config.ignore_classifier}')
if not os.path.exists(trajectory_dir): os.makedirs(trajectory_dir)

print_config_to_fname(config, os.path.join(trajectory_dir,'config.txt'))


if classifier_type=='CNN':
    classifier = CNNClassification(torch.Size([1, 28, 28]), tgt_non_tgt, device, name=discr_str) 
    learning_rate = 1e-3
else:
    classifier = ElasticNetLinearClassification(n_features, tgt_non_tgt, device=device, name=discr_str)
    learning_rate = 1e-4


if not os.path.exists(classifier_fname):
    print('classifier TRAINING: ',discr_str)
    tl = BinaryDataLoader(trainset, tgt_non_tgt, batch_size=16)
    testl = BinaryDataLoader(testset, tgt_non_tgt, batch_size=16) 
    classifier.evaluate(testl)
    classifier.fit( epochs=classifier_epochs, learning_rate=learning_rate, train_loader=tl, val_loader = testl)
    classifier.save(classifier_fname)
    print('Performance: ')
    print(classifier.history_to_df())
    print(f'Saved {classifier_fname}')
else:
    classifier.load(classifier_fname)
    classifier.to(device)
    print(f'Loaded {classifier_fname}')

print(f'decnef iters: {config.decnef_iters}')

np.random.seed(global_random_seed)

lattice_fname, z_grid_init_fname = os.path.join(modelpath, 'lattice.npy'), os.path.join(modelpath, f'z_grid_init_{clean_discr_str}.npy')

n_trajectories = config.n_trajectories

if os.path.exists(z_grid_init_fname):
    z_grid_init = np.load(z_grid_init_fname)[:,:z_dim]
else: 
    z_grid_init = make_init_z_lattice(n_trajectories,
                                      z_dim,
                                      all_class_prototypes,
                                      all_class_prototypes_sigma,
                                      tgt_non_tgt, 
                                      lattice_fname, z_grid_init_fname)[:,:z_dim]

if n_trajectories!=len(z_grid_init): print(f'z_grid_init has {len(z_grid_init)} points, but we are taking only the {n_trajectories} first points')
#%%
    
for i in tqdm(range(n_trajectories), desc = 'Init number', total = n_trajectories):
    z0 = z_grid_init[i]
    # if len(z0.shape)==1: z0 = np.array([z0]).T
    if i==0: print(z0.shape)
    seeds = config.seed_list[i]
    for trajectory_random_seed in tqdm(seeds,
                                       desc= 'Traj: '):

        trajectory_name = f'{config.dataset}_TRAJ{trajectory_random_seed}_z0{i}_{generator_name}_{classifier_name}_UR{update_rule_name}_IGDIS{config.ignore_classifier}_linv{config.lambda_inv}'
        trajectory_fname = os.path.join(trajectory_dir, f'{trajectory_name}.npz')
        if os.path.exists(trajectory_fname): print('Found'); continue
        generated_images,\
        trajectory,\
        probabilities,\
        all_probabilities,\
        sigma =  compute_single_trajectory(vae, classifier,
                                           trajectory_random_seed,
                                           train_loader, config.target_class_idx,
                                           update_rule_func, p_scale_func,
                                           z_current= torch.Tensor(z0),
                                           trajectory_name=trajectory_name, 
                                           n_iter = config.decnef_iters, lambda_ = lambda_,
                                           device=device, 
                                           ignore_classifier=config.ignore_classifier,
                                           start_from_origin=True,
                                           )
        #%%
        np.savez_compressed(trajectory_fname, 
                            generated_images = generated_images,
                            trajectory = trajectory,
                            probabilities = probabilities,
                            all_probabilities = all_probabilities,
                            sigma = sigma
                            )
        # print('Saved ',os.path.join(trajectory_dir, f'{trajectory_name}.npz'), ', exiting.')