#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze results of DecNef simulations from
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models (Olza et al., 2026)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.
Created on Wed Jul  2 17:05:25 2025

@author: alexolza
"""
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import os
import sys
sys.path.append('../')
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
###########################
from components.generators import VAE
from components.classifiers import CNNClassification, ElasticNetLinearClassification
from visualization.plotting import plot_probability_map_grid, heatmap_mean_X_over_time
from analysis.utils import generator_probability_map, trajectory_properties_as_df, get_probabilities
############################################
#%%
# FONT STYLE IS DEFINED IN visualization/plotting.py, WHICH WE IMPORT ABOVE HERE
#%%
"""
##############################################
CONFIGURATION VARIABLES
##############################################
"""
ext = 'pdf'
EXP_NAME  = sys.argv[1]
print(EXP_NAME)
dataset = 'synth_fMRI_FASHION' if 'synth' in EXP_NAME else EXP_NAME.split('_')[0]

target_class_idx = int(eval(sys.argv[2])) 
non_target_class_idx = int(eval(sys.argv[3])) 
subject = int(eval(sys.argv[4])) 
z_dim = int(eval(sys.argv[5]))
linv = int(eval(sys.argv[6])) 
decnef_iters =  500
n_trajs = 100
lambda_ = 1/linv 
generator_epochs = 25
space_radius = (-2.75, 2.75, -2.75, 2.75)
n_samples=75
device=sys.argv[7]
generator_batch_size=64
classifier_type = 'CNN' if not 'fMRI' in dataset else 'ELASTICNET'
generator_name = 'VAE' if not 'PCA' in EXP_NAME else 'PCA'
classifier_epochs = 10
classifier_batch_size = 16
tgt_non_tgt = [target_class_idx, non_target_class_idx]
tgt, non_tgt= tgt_non_tgt

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

combo_names = [class_name_dict_reverse[i].upper() for i in tgt_non_tgt]
clean_discr_str = re.sub('[^a-zA-Z0-9]','', f'{combo_names[0]} vs {combo_names[1]}')

figpath = f'../EXPERIMENTS/{EXP_NAME}/subj{subject}/figures/'
outpath = f'../EXPERIMENTS/{EXP_NAME}/subj{subject}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/subj{subject}/weights/'
logspath = f'../EXPERIMENTS/{EXP_NAME}/subj{subject}/logs/'


if not os.path.exists(figpath): os.makedirs(figpath)

classifier_name = f'{classifier_type}_{clean_discr_str}__BS{classifier_batch_size}_E{classifier_epochs}'
classifier_fname = os.path.join(modelpath, classifier_name+'.pt')
generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

keys = {f'With memory ({classifier_type})': ['MNDAVMem', 0],
        'With memory (Random)': ['MNDAVMem', 1],
        f'MNDAV ({classifier_type})': ['MNDAV', 0],
        'MNDAV (Random)': ['MNDAV', 1]
        }

#%%
img_size = 14386 # TODO: assign programatically# trainset[0][0].shape[-1]
tabular= True if 'fMRI' in dataset else None
n_features = img_size if 'fMRI' in dataset else None
if 'fMRI' not in dataset:
    classifier = CNNClassification(torch.Size([1, 28, 28]), tgt_non_tgt, device='cuda:0') 
else:
    classifier = ElasticNetLinearClassification(img_size, tgt_non_tgt, device='cuda:0')

classifier.load(classifier_fname)
classifier.device = device
classifier.to(device)
tabular= True if 'fMRI' in dataset else False
generator = VAE(z_dim=z_dim,tabular=tabular, n_features=n_features, device= device).to(device)
generator.load(generator_fname+'.pt')
generator.device = device
# generator_history = generator.history_to_df()
print(f'Loaded {generator_fname}')
#%%
"""
##############################################
LOADING RESULTS
##############################################
"""

generator.eval()
latent_prototype = generator.prototypes[target_class_idx] 
alt_latent_prototype = generator.prototypes[non_target_class_idx]
if 'VAE' in generator_name: # [1] is the variance and [0] is the mu
    latent_prototype = latent_prototype[0]
    alt_latent_prototype = alt_latent_prototype[0]
else:
    latent_prototype = latent_prototype.reshape(1,-1)
    alt_latent_prototype = alt_latent_prototype.reshape(1,-1)
    
prototype = generator.decoder(torch.Tensor(latent_prototype).to(device),
                                            generator.target_size).detach()
all_class_prototypes = {class_name_dict_reverse[k]: v for k,v in generator.prototypes.items()}


probability_dfs, sigma_dfs, pixcorr_dfs, ssim_dfs, trajectory_matrices = {},{},{},{},{}
dist_dfs = {}
names_dfs = {}
random_probability_dfs = {}

print('processing results...')
for UR, URname in tqdm(zip([ 'MNDAVMem', 'MNDAV'], ['With memory', 'MNDAV'])):
    for IGDIS, IGDIS_label in zip([0,1], ['CNN', 'Random']):
            trajectory_dir = os.path.join(outpath,f'TRAJS_{generator_name}_{classifier_name}', f'linv{linv}',f'UR{UR}',f'IGDIS{IGDIS}')
            
            trajectory_names = glob(f'{dataset}_TRAJ*_z0*_{generator_name}_{classifier_name}_UR{UR}_IGDIS{IGDIS}_linv{linv}.npz',
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
            if label=='With memory (Random)': random_probability_dfs[label] = get_probabilities(trajectory_matrices[label], tgt, generator, classifier, batch_size=1024)

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
generator = generator.to(device)
probability_map, coordinates, generated_samples, pca_pipe_proto, pca_df_proto = generator_probability_map(generator, z_dim, classifier, target_class_idx, n_samples, space_radius = space_radius)
#%%
"""
##############################################
GENERATING PLOTS
##############################################
"""
tgt, non_tgt= tgt_non_tgt
seed=7

plot_probability_map_grid(probability_map, coordinates, pca_df_proto,
                                 generator, z_dim, classifier,
                                 target_class_idx, non_target_class_idx, class_name_dict, clean_discr_str,
                                 n_samples, space_radius,
                                 ext = ext,
                                 seed=seed,
                                
                                   maxmin=True,
                                   fname = os.path.join(figpath, f'pmap_{clean_discr_str}'),
                                   accuracy= f'{classifier.history_to_df()["val_acc"].values[-1]:.2f}'
                                   )
#%%
random_probability_dfs = {}
metric='p'
metric_dfs = all_dfs[metric]
for i, key in enumerate(trajectory_matrices.keys()):
        # if not 'Random' in key: continue
        ALL_TRAJS = trajectory_matrices[key] # shape (T, Ntraj, 2)
        tnames = names_dfs[key]
        if ALL_TRAJS.shape[1]== 0: continue
    
        title = key + f' - {metric}(t)'
        clean_title = re.sub('[^a-zA-Z0-9]','', title)
        fname = os.path.join(figpath, f'{clean_discr_str}_{clean_title}_withpmean')
        offset=2 if metric=='p' else 0
        metric_dfs[key].plot(legend = False)
        if ('Random' in key) and (metric=='p'):
            random_probability_dfs[key] = get_probabilities(ALL_TRAJS, tgt, generator, classifier, batch_size=1024)
            heatmap_mean_X_over_time(random_probability_dfs[key], ALL_TRAJS, tnames, latent_prototype, 
                                     class_name_dict,
                                     title, save=True, fname=fname, offset=offset, ext=ext, N_A=10,
                                     )
        else:
            heatmap_mean_X_over_time(metric_dfs[key], ALL_TRAJS, tnames, latent_prototype,
                                     class_name_dict,
                                     title, save=True, fname=fname, offset=offset, ext=ext, N_A=10,
                                     )


figsize=(8,6)
key = 'With memory (CNN)'
tnames = names_dfs[key]
ALL_TRAJS = trajectory_matrices[key]
TRAJS_groupbyR = {
    init: ALL_TRAJS[:, idx, :].mean(axis=1)
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
    
    # Plot all trajectories in the latent space (thin white lines)
    fig, ax = plt.subplots(figsize=(8, 8))
    h, w = probability_map.shape
    for ax_ in [ax]:
        im = ax.imshow(
            probability_map,
            extent= space_radius, 
            origin="lower",
            cmap="viridis",
            vmin=0, vmax=1,
            aspect=(w/h),
            alpha=1
        )
        
        
    data = trajectory_matrix
    for i in range(n_samples):
        ax.plot(data[:, i, 0], data[:, i, 1], alpha=0.07, color='white')  # trajectory of sample i
    
    # Plot mean trajectories starting from each z0 (bold lines, time-colored)
    ALL_TRAJS = trajectory_matrices[label]
    TRAJS_groupbyR = {
        init: ALL_TRAJS[:, idx, :].mean(axis=1) 
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
    
    # Attach colorbars without shrinking
    divider = make_axes_locatable(ax_)
    cax_right = divider.append_axes("right", size="5%", pad=0.1)
    cax_bottom = divider.append_axes("bottom", size="5%", pad=0.55)
    fig.colorbar(sm, cax=cax_right, orientation="vertical", label='Time')
    fig.colorbar(im, cax=cax_bottom, label=r" $p(y=y^\ast \mid x)$;  $x$: projection of $z$ to the native space of the data", location='bottom')
    
    
    ax.plot(latent_prototype[:, 0], latent_prototype[:, 1], markersize=18, marker = 'X', color='c', markeredgecolor='white', label=r"Target: $z : y=y^\ast$")
    ax.plot(alt_latent_prototype[:, 0], alt_latent_prototype[:, 1], markersize=18, marker = 'o', color='c', markeredgecolor='white', label=r"Alternative: $z : y=y^{alt}$")
    ax.set_xlabel(r'$z_x$')
    ax.set_ylabel(r'$z_y$')
    # ax.legend(loc='upper right')   
    ax.set_xlim(space_radius[0], space_radius[1])
    ax.set_ylim(space_radius[2], space_radius[3])
    ax.set_aspect("equal")
    ax.grid(True)  
    plt.show()
    fig.savefig(os.path.join(figpath, f'trajs_{clean_discr_str}_{urname}_igdis{igdis}.{ext}'),
                transparent=True,
                dpi=600
                )
    print(os.path.join(figpath, f'trajs_{clean_discr_str}_{urname}_igdis{igdis}.{ext}'))    
    plt.show()

