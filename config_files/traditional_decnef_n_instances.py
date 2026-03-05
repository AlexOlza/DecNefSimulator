#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:30:15 2025

@author: alexolza
"""

import argparse

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


update_rule_names = ['MNDAV', 'MNDAVMem']
def npz_file_paths(dataset, subject=None):
    if dataset=='FASHION': return ''
    elif dataset =='synth_fMRI_FASHION':
        repeat = 5 if subject==8 else 8
        return f'../data/fMRIsynth/subj0{subject}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri.npz'


def traditional_decnef_n_instances_parser():
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
    parser.add_argument('--ignore_discriminator', required = False, default= 0, type= int)
    parser.add_argument('--update_rule_idx', required = False, default= 0, type= int)
    #parser.add_argument('--production', required = False, default= 1, type= int)
    parser.add_argument('--generator_name', type= str, required = False, default='VAE')
    #parser.add_argument('--discriminator_type', type= str, required = False, default='CNN')
    parser.add_argument('--generator_batch_size', required = False, default=64, type= int)
    parser.add_argument('--discriminator_epochs', required = False, default=10, type= int)
    parser.add_argument('--discriminator_batch_size', required = False, default=16, type= int)
    parser.add_argument('--n_trajectories_per_init', required = False, default=10, type= int)
    parser.add_argument('--device', required = False, default='cuda:0', type= str)
    
    c0 = parser.parse_args()
    parser.add_argument('--update_rule_name', type= str, required = False, default=update_rule_names[c0.update_rule_idx])
    npz_fname = npz_file_paths(c0.dataset, c0.subject)
    parser.add_argument('--npz_file_path', type= str, required = False, default=npz_fname)
    
    default_generator_epochs = 25 if 'fMRI' in c0.dataset else 25 
    default_z_dim = 256 if 'fMRI' in c0.dataset else 2 
    default_z_dim = 2 if 'z2' in c0.EXP_NAME else default_z_dim 
    parser.add_argument('--generator_epochs', required = False, default=default_generator_epochs, type= int)
    parser.add_argument('--z_dim', type= int, required = False, default=default_z_dim)

    seed_list = [] 
    for i in range(c0.n_trajectories):
        seeds = [ s for s in range((i+1)*c0.trajectory_random_seed_init,
        (i+1)*c0.trajectory_random_seed_init + c0.n_trajectories_per_init)]
        seed_list.append(seeds)
    parser.add_argument('--seed_list', type=list_of_ints, required = False, default=seed_list)
    
    
    # parser.parse_args(args=['--update_rule_name', update_rule_names[c0.update_rule_idx]], namespace=c0)
    config = parser.parse_args()
    for arg in vars(config):
        if arg=='seed_list': continue
        print(f'{arg} =  {getattr(config, arg)}')
    return config

def print_config_to_fname(config, fname):
    with open(fname, 'w') as f:
        f.write('CONFIGURATION DETAILS: \n')
        for arg in vars(config):
            f.write(f'{arg} =  {getattr(config, arg)}')
