#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:30:15 2025

@author: alexolza
"""
import os
import argparse
import ast
import numpy as np
import torch
import random

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


update_rule_names = ['MNDAV', 'MNDAVMem']
def npz_file_paths(dataset, subject=None):
    if dataset =='synth_fMRI_FASHION':
        repeat = 5 if subject==8 else 8
        return f'../../../../data/fMRIsynth/subj0{subject}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri_with_indices.npz'
    else:
        return ''

def config_parser():
    parser = argparse.ArgumentParser(
                        prog='traditional_decnef_n_instances',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument('--read_args', type= int, default=1)
    parser.add_argument('EXP_NAME')
    parser.add_argument('--dataset', required = False, default='FASHION', type= str)
    parser.add_argument('--subject', required = False, default=0, type= int)
    parser.add_argument('--trajectory_random_seed_init', required = False, default= 0, type= int)
    parser.add_argument('--n_trajectories', required = False, default= 10, type= int)
    parser.add_argument('--target_class_idx', required = False, default= 0, type= int)
    parser.add_argument('--non_target_class_idx', required = False, default= 1, type= int)
    parser.add_argument('--lambda_inv', required = False, default= 5, type= int)
    parser.add_argument('--gamma_inv', required = False, default= 5, type= int)
    parser.add_argument('--decnef_iters', required = False, default= 500, type= int)
    parser.add_argument('--ignore_classifier', required = False, default= 0, type= int)
    parser.add_argument('--update_rule_idx', required = False, default= 0, type= int)
    # parser.add_argument('--generator_name', type= str, required = False, default='VAE')
    parser.add_argument('--generator_batch_size', required = False, default=64, type= int)
    parser.add_argument('--classifier_epochs', required = False, default=10, type= int)
    parser.add_argument('--classifier_batch_size', required = False, default=16, type= int)
    parser.add_argument('--n_trajectories_per_init', required = False, default=10, type= int)
    parser.add_argument('--device', required = False, default='cuda:0', type= str)
    parser.add_argument('--z_dim', type= int, required = False, default=2)
    parser.add_argument('--generator_epochs', required = False, default=25, type= int)
    
    c0 = parser.parse_args()
    parser.add_argument('--update_rule_name', type= str, required = False, default=update_rule_names[c0.update_rule_idx])
    npz_fname = npz_file_paths(c0.dataset, c0.subject)
    parser.add_argument('--npz_file_path', type= str, required = False, default=npz_fname)
    if 'PCA' in c0.EXP_NAME:
        c0.generator_name = 'PCA'
        print(f'Setting generator name to {c0.generator_name}')
        parser.add_argument('--generator_name', type= str, required = False, default='PCA')
    else:
        parser.add_argument('--generator_name', type= str, required = False, default='VAE')

    seed_list = [] 
    for i in range(c0.n_trajectories):
        seeds = [ s for s in range((i+1)*c0.trajectory_random_seed_init,
        (i+1)*c0.trajectory_random_seed_init + c0.n_trajectories_per_init)]
        seed_list.append(seeds)
    parser.add_argument('--seed_list', type=list_of_ints, required = False, default=seed_list)
    
    
    config = parser.parse_args()
    for arg in vars(config):
        if arg=='seed_list': continue
        print(f'{arg} =  {getattr(config, arg)}')
    return config

def print_config_to_fname(config, fname):
    with open(fname, 'w') as f:
        f.write('CONFIGURATION DETAILS: \n')
        for arg in vars(config):
            f.write(f'{arg} =  {getattr(config, arg)}\n')

def print_config_to_screen(config):
        print('CONFIGURATION DETAILS: \n')
        for arg in vars(config):
            print(f'{arg} =  {getattr(config, arg)}')
            

def load_config(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # skip empty lines and header
            if not line or line.startswith("CONFIGURATION"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # handle empty values
            if value == "":
                data[key] = None
                continue

            # try safe parsing (numbers, lists, etc.)
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                # fallback: keep as string
                data[key] = value

    return argparse.Namespace(**data)

def seed_everything(seed=42, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')
    
