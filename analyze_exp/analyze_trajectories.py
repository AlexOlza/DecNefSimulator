#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze results of DecNef simulations from
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.
Created on Wed Jul  2 17:05:25 2025

@author: alexolza
"""
import matplotlib.cm as cm
# matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import os
import sys
sys.path.append('../')
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from glob import glob
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

###########################################
from components.generators import VAE
from components.discriminators import CNNClassification, ElasticNetLinearClassification
from visualization.plotting import plot_discriminator_violins, plot_probability_map_grid, heatmap_mean_X_over_time
from analysis.utils import vae_probability_map, trajectory_properties_as_df, get_probabilities
############################################
#%%
# FONT STYLE IS DEFINED IN visualization/plotting.py, WHICH WE IMPORT ABOVE HERE

# mpl.rcParams["font.family"] = "DejaVu Serif"   # or another installed system font you like
# # Use Computer Modern for math
# mpl.rcParams["mathtext.fontset"] = "cm"
# mpl.rcParams["mathtext.default"] = "bf"

# for k, v in mpl.rcParams.items():
#     if 'mathtext' in k: print(f'{k} = {v}')

#%%
"""
##############################################
CONFIGURATION VARIABLES
##############################################
"""
ext = 'pdf'
EXP_NAME  = sys.argv[1]
dataset = 'synth_fMRI_FASHION' if 'synth' in EXP_NAME else 'FASHION'

target_class_idx = int(eval(sys.argv[2])) 
non_target_class_idx = int(eval(sys.argv[3])) 
subj = int(eval(sys.argv[4])) 
decnef_iters =  500
n_trajs = 100
linv = 5
z_dim = 2 if (dataset=='FASHION' or 'z2' in EXP_NAME) else 256
lambda_ = 1/linv 
generator_epochs = 25 if dataset=='FASHION' else 25#300
space_radius=2.5 if dataset=='FASHION' else 3.5
n_samples=75
device='cuda:0'
generator_batch_size=64
discriminator_type = 'CNN' if dataset=='FASHION' else 'ELASTICNET'
generator_name = 'VAE'
discriminator_epochs = 10
discriminator_batch_size = 16
tgt_non_tgt = [target_class_idx, non_target_class_idx]
outpath = f'../EXPERIMENTS/{EXP_NAME}/output/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/weights/'
repeat = 5 if subj==8 else 8
npz_file_paths = {'FASHION':'',
                  'synth_fMRI_FASHION':'../data/fMRIsynth/subj0{subj}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri.npz',
                  'synth_fMRI_COCO':''}

npz_file_path = npz_file_paths[dataset]

class_name_dict =  {'TSHIRTTOP': 0,
  'TROUSER': 1,
  'PULLOVER': 2,
  'DRESS': 3,
  'COAT': 4,
  'SANDAL': 5,
  'SHIRT': 6,
  'SNEAKER': 7,
  'BAG': 8,
  'ANKLEBOOT': 9}
class_name_dict_reverse = {v: k for k, v in class_name_dict.items()} 

combo_names = [class_name_dict_reverse[i] for i in tgt_non_tgt]
clean_discr_str = re.sub('[^a-zA-Z0-9]','', f'{combo_names[0]} vs {combo_names[1]}')
figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/{clean_discr_str}/nfb_eval/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/figures/{clean_discr_str}/nfb_eval/'

if not os.path.exists(figpath): os.makedirs(figpath)

discriminator_name = f'{discriminator_type}_{clean_discr_str}__BS{discriminator_batch_size}_E{discriminator_epochs}'
discriminator_fname = os.path.join(modelpath, discriminator_name+'.pt')
generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

keys = {f'With memory ({discriminator_type})': ['MNDAVMem', 0],
        'With memory (Random)': ['MNDAVMem', 1],
        f'MNDAV ({discriminator_type})': ['MNDAV', 0],
        'MNDAV (Random)': ['MNDAV', 1]
        }

seed_list = [] 
for i in range(100):
    seeds = [ s for s in range((i+1)*42,
    (i+1)*42 + 10)]
    seed_list.append(seeds)

z_grid_init_fname = os.path.join(modelpath, f'z_grid_init_{clean_discr_str}.npy')
z_grid_init = np.load(z_grid_init_fname)
original_gaussian = z_grid_init[:,-1].astype(int)
seed_list = np.array(seed_list).flatten()
map_traj_orig = {s: o for s, o in zip(seed_list, original_gaussian.repeat(10))}

#%%
img_size = 14386 # TODO: assign programatically# trainset[0][0].shape[-1]
tabular= True if 'fMRI' in dataset else None
n_features = img_size if 'fMRI' in dataset else None
if dataset=='FASHION':
    discriminator = CNNClassification(torch.Size([1, 28, 28]), tgt_non_tgt, device='cuda:0') 
else:
    discriminator = ElasticNetLinearClassification(img_size, tgt_non_tgt, device='cuda:0')

discriminator.load(discriminator_fname)

vae = VAE(z_dim=z_dim, tabular=tabular, n_features=img_size, device= device).to(device)
vae.target_size = 2 if dataset=='FASHION' else 256
vae.load(generator_fname+'.pt')
#%%
"""
##############################################
LOADING RESULTS
##############################################
"""

vae.eval()
latent_prototype = vae.prototypes[target_class_idx][0] # [1] is the variance and [0] is the mu
alt_latent_prototype = vae.prototypes[non_target_class_idx][0]
prototype = vae.decoder(torch.Tensor(latent_prototype).to(vae.device),
                                            vae.target_size).detach()
all_class_prototypes = {class_name_dict_reverse[k]: v for k,v in vae.prototypes.items()}


probability_dfs, sigma_dfs, pixcorr_dfs, ssim_dfs, trajectory_matrices = {},{},{},{},{}
dist_dfs = {}
names_dfs = {}

print('processing results...')
for UR, URname in tqdm(zip(['MNDAV', 'MNDAVMem'], ['MNDAV', 'With memory'])):
    for IGDIS, IGDIS_label in zip([0,1], ['CNN', 'Random']):
            trajectory_dir = os.path.join(outpath,f'TRAJS_{generator_name}_{discriminator_name}', f'linv{linv}',f'UR{UR}',f'IGDIS{IGDIS}')
            
            trajectory_names = glob(f'{dataset}_TRAJ*_z0*_{generator_name}_{discriminator_name}_UR{UR}_IGDIS{IGDIS}_linv{linv}.npz',
                                    root_dir=trajectory_dir)
            
            trajectory_paths = [os.path.join(trajectory_dir, f'{trajectory_name}')
                                for trajectory_name in trajectory_names]
            print(f'N={len(trajectory_names)}; {UR}, {IGDIS_label}')
            label = f'{URname} ({IGDIS_label})'    
            probability_dfs[label], sigma_dfs[label],\
                    trajectory_matrices[label], names_dfs[label] =trajectory_properties_as_df(trajectory_paths,
                                                                                             decnef_iters, 
                                                                                                 prototype,
                                                                                                 latent_prototype)
for key in trajectory_matrices.keys():
    try:
        names_dfs[key]["init"] = names_dfs[key]["traj_name"].str.extract(r"z0*(\d+)_").astype(int)
        names_dfs[key]["rep"] = names_dfs[key]["traj_name"].str.extract(r"TRAJ*(\d+)_").astype(int)
        names_dfs[key]["linv"] = names_dfs[key]["traj_name"].str.extract(r"linv*(\d+).npz").astype(int)
    except:
        print('pass: ', key)
all_dfs = {'p': probability_dfs, 
           'dist': dist_dfs,
           'sigma': sigma_dfs, 
           }
#%%
vae = vae.to(vae.device)
probability_map, coordinates, generated_samples = vae_probability_map(vae, discriminator, target_class_idx,  n_samples=75)
#%%
"""
##############################################
GENERATING PLOTS
##############################################
"""
tgt, non_tgt= tgt_non_tgt
seed=7

tgt, non_tgt = tgt_non_tgt
random_probability_dfs = {}
for metric in ['p', 'sigma']:
    metric_dfs = all_dfs[metric]
    for i, key in enumerate(trajectory_matrices.keys()):
        if not 'Random' in key: continue
        ALL_TRAJS = trajectory_matrices[key] # shape (501, 7781, 2)
        tnames = names_dfs[key]
        if ALL_TRAJS.shape[1]== 0: continue
    
        title = key + f' - {metric}(t)'
        clean_title = re.sub('[^a-zA-Z0-9]','', title)
        fname = os.path.join(figpath, f'heatmap_{clean_title}_NA')
        offset=2 if metric=='p' else 0
        if ('Random' in key) and (metric=='p'):
            random_probability_dfs[key] = get_probabilities(ALL_TRAJS, tgt, vae, discriminator, batch_size=1024)
            heatmap_mean_X_over_time(random_probability_dfs[key], ALL_TRAJS, tnames, latent_prototype, 
                                     title, save=True, fname=fname, offset=offset, ext=ext, N_A=10)
        else:
            heatmap_mean_X_over_time(metric_dfs[key], ALL_TRAJS, tnames, latent_prototype,
                                     title, save=True, fname=fname, offset=offset, ext=ext, N_A=10)


figsize=(8,6)
key = 'With memory (CNN)'
tnames = names_dfs[key]
ALL_TRAJS = trajectory_matrices[key]
TRAJS_groupbyR = {
    init: ALL_TRAJS[:, idx, :].mean(axis=1)   # shape (501, 2)
    for init, idx in tnames.groupby("init").groups.items()
    if idx.max() < ALL_TRAJS.shape[1]
}

#%%
for label, trajectory_matrix in tqdm(trajectory_matrices.items(),
                                     desc='plotting trajectories',
                                     total=len(list(trajectory_matrices.keys()))):
    urname, igdis = keys[label]
    if 'memory' not in label: continue
    n_time, n_samples, _ = trajectory_matrix.shape
    
    # Plot all trajectories in 2D space
    fig, ax = plt.subplots(figsize=(8, 8))
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    h, w = probability_map.shape
    for ax_ in [ax, ax2, ax3]:
        im = ax_.imshow(
            probability_map,
            extent=[-space_radius, space_radius, -space_radius, space_radius],
            origin="lower",
            cmap="viridis",
            vmin=0, vmax=1,
            aspect=(w/h),
            alpha=1
        )
        
        
    data = trajectory_matrix
    for i in range(n_samples):
        ax.plot(data[:, i, 0], data[:, i, 1], alpha=0.07, color='white')  # trajectory of sample i
        ax2.plot(data[:, i, 0], data[:, i, 1], alpha=0.05, color='white')  # trajectory of sample i

    # Overlay mean trajectories
    ALL_TRAJS = trajectory_matrices[label]
    TRAJS_groupbyR = {
        init: ALL_TRAJS[:, idx, :].mean(axis=1)   # shape (501, 2)
        for init, idx in names_dfs[label].groupby("init").groups.items()
        if idx.max() < ALL_TRAJS.shape[1]
    }
    
    # below: needed for colorbars
    colors = cm.inferno(np.linspace(0, 1, n_time))
    norm = mcolors.Normalize(vmin=0, vmax=n_time)
    sm = cm.ScalarMappable(cmap="inferno", norm=norm)
    sm.set_array([]) 
    
    for mean_traj in tqdm(TRAJS_groupbyR.values(),
                          total = len(list(TRAJS_groupbyR.values())),
                          desc = 'plotting mean trajs'):
        colors = cm.inferno(np.linspace(0, 1, n_time))
        for t in range(n_time - 1):
            ax.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2,  1], color=colors[t], linewidth=1.5) 
            ax2.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2,  1], color=colors[t], linewidth=1.5) 
            ax3.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2,  1], color=colors[t], linewidth=1.5) 
    
    # Attach colorbars without shrinking
    for fig_,ax_ in zip([fig, fig2, fig3], [ax, ax2, ax3]):
        divider = make_axes_locatable(ax_)
        cax_right = divider.append_axes("right", size="5%", pad=0.1)
        cax_bottom = divider.append_axes("bottom", size="5%", pad=0.55)
        fig_.colorbar(sm, cax=cax_right, orientation="vertical", label='Time')
        fig_.colorbar(im, cax=cax_bottom, label=r" $p(y=y^\ast \mid x)$;  $x$: projection of $z$ to the native space of the data", location='bottom')
    for ax_ in [ax, ax2, ax3]:
        ax_.plot(latent_prototype[:, 0], latent_prototype[:, 1], markersize=20, marker = 'X', color='c', markeredgecolor='white', label=r"Target: $z : y=y^\ast$")
        ax_.plot(alt_latent_prototype[:, 0], alt_latent_prototype[:, 1], markersize=20, marker = 'o', color='c', markeredgecolor='white', label=r"Alternative: $z : y=y^{alt}$")
        ax_.set_xlabel(r'$z_x$')
        ax_.set_ylabel(r'$z_y$')
        ax_.legend(loc='upper right')
        

        ax_.set_xlim(-space_radius, space_radius)
        ax_.set_ylim(-space_radius, space_radius)
        ax_.set_aspect("equal")
    ax.grid(True)
    ax3.grid(True)   
    # plt.show()
    fig.savefig(os.path.join(figpath, f'trajs_{urname}_igdis{igdis}.{ext}'))
    fig2.savefig(os.path.join(figpath, f'trajs2_{urname}_igdis{igdis}.{ext}'))
    fig3.savefig(os.path.join(figpath, f'trajs3_{urname}_igdis{igdis}.{ext}'))
    
    plt.show()
    break
