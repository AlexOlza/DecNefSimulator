#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze results of DecNef simulations from
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Wed Dec 10 17:13:44 2025

@author: alexolza
"""
# Third party imports (and seeding)
import sys
sys.path.append('../')
import re
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import os
###########################################
#	Seeding before torch import       #
from utils.configuration import seed_everything
global_random_seed = 42
seed_everything(global_random_seed)
###########################################
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from glob import glob
from tqdm import tqdm
# from pathlib import Path
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
###########################################
# DecNefSimulator imports
from utils.utils import load_dataset, make_init_z_lattice
from components.generators import VAE
from components.classifiers import ElasticNetLinearClassification
from visualization.plotting import heatmap_mean_X_over_time, plot_probability_map_grid
from analysis.utils import trajectory_properties_as_df, get_probabilities, generator_probability_map
############################################
#%%
"""
##############################################
CONFIGURATION VARIABLES
##############################################
"""
mpl.rcParams["font.family"] = "DejaVu Serif"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.default"] = "bf"
ext = 'pdf'
EXP_NAME  = sys.argv[1]
dataset = 'synth_fMRI_FASHION'

target_class_idx = int(eval(sys.argv[2])) 
non_target_class_idx = int(eval(sys.argv[3])) 
subj = int(eval(sys.argv[4])) 
z_dim = int(eval(sys.argv[5]))
linv = int(eval(sys.argv[6]))
device = sys.argv[7]
reduction = sys.argv[8]
decnef_iters =  500
n_trajs = 100

z_dim = 256
lambda_ = 1/linv 
generator_epochs = 25
n_samples=75
generator_batch_size=64
classifier_type = 'ELASTICNET'
generator_name = 'VAE'
classifier_epochs = 10
classifier_batch_size = 16
tgt_non_tgt = [target_class_idx, non_target_class_idx]
tgt, non_tgt= tgt_non_tgt
seed=7
outpath = f'../EXPERIMENTS/{EXP_NAME}/subj{subj}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/subj{subj}/weights/'
repeat = 5 if subj==8 else 8
npz_file_paths = {'FASHION':'',
                  'synth_fMRI_FASHION':f'../../../../data/fMRIsynth/subj0{subj}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri_with_indices.npz',
                  'synth_fMRI_COCO':''}

npz_file_path = npz_file_paths[dataset]

transform = transforms.Compose([transforms.ToTensor()]) if dataset=='FASHION' else None
trainset = load_dataset(dataset, transform, npz_file_path=npz_file_path, train=True)
testset = load_dataset(dataset, transform, npz_file_path=npz_file_path, train=False)
train_loader = DataLoader(trainset, batch_size=64)
test_loader = DataLoader(testset, batch_size=64)
if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
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
clean_clf_str = re.sub('[^a-zA-Z0-9]','', f'{combo_names[0]} vs {combo_names[1]}')
figpath = f'../EXPERIMENTS/{EXP_NAME}/subj{subj}/figures/'

if not os.path.exists(figpath): os.makedirs(figpath)

classifier_name = f'{classifier_type}_{clean_clf_str}__BS{classifier_batch_size}_E{classifier_epochs}'
classifier_fname = os.path.join(modelpath, classifier_name+'.pt')
generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

keys = {f'With memory ({classifier_type})': ['MNDAVMem', 0],
        'With memory (Random)': ['MNDAVMem', 1],
        f'MNDAV ({classifier_type})': ['MNDAV', 0],
        'MNDAV (Random)': ['MNDAV', 1]
        }

seed_list = [] 
for i in range(100):
    seeds = [ s for s in range((i+1)*42,
    (i+1)*42 + 10)]
    seed_list.append(seeds)

#%%
"""
LOAD MODELS
"""
img_size = 14386 # TODO: assign programatically# trainset[0][0].shape[-1]
tabular= True
n_features = img_size
print(generator_fname)

classifier = ElasticNetLinearClassification(img_size, tgt_non_tgt, device='cuda:0')

classifier.load(classifier_fname)
classifier.eval()
generator = VAE(z_dim=z_dim, tabular=tabular, n_features=img_size, device= device).to(device)
generator.load(generator_fname+'.pt')
generator.eval()
latent_prototype = generator.prototypes[target_class_idx][0] # [1] is the variance and [0] is the mu
alt_latent_prototype = generator.prototypes[non_target_class_idx][0]
prototype = generator.decoder(torch.Tensor(latent_prototype).to(generator.device),
                                            generator.target_size).detach()
all_class_prototypes = {class_name_dict_reverse[k]: v for k,v in generator.prototypes.items()}

#%%
"""
##############################################
LOADING RESULTS
##############################################
"""

probability_dfs, sigma_dfs, trajectory_matrices = {},{},{}
names_dfs = {}
random_probability_dfs = {}
print('processing results...')
for UR, URname in tqdm(zip(['MNDAV', 'MNDAVMem'], ['MNDAV', 'With memory'])):
    for IGDIS, IGDIS_label in zip([0,1], ['ELASTICNET', 'Random']):
            print(f'{dataset}_TRAJ*_z0*_{generator_name}_{classifier_name}_UR{UR}_IGDIS{IGDIS}_linv{linv}.npz')    
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
           'sigma': sigma_dfs, 
          }
#%%
"""
##############################################
GENERATING PLOTS
##############################################
"""
"""
PROBABILITY MAP EQUIVALENT
"""
pca_fname = os.path.join(modelpath, "pca_latent_visualization.pkl")
pmap_fname = os.path.join(modelpath, f"pmap_{clean_clf_str}.npz")
n_samples=75
label='With memory (ELASTICNET)'
generator = generator.to(generator.device)
all_class_prototypes = np.vstack([prot[0].ravel() for idx, prot in generator.prototypes.items()
                                   ])
all_class_prototypes_sigma = np.vstack([prot[1].ravel() for idx, prot in generator.prototypes.items()
                                   ])
points = make_init_z_lattice(100000, z_dim, all_class_prototypes, all_class_prototypes_sigma, tgt_non_tgt, 
                            lattice_fname='aux', z_grid_init_fname='aux.npy')
with torch.no_grad():
        probability_map, coordinates, generated_samples, pca_pipe_proto, pca_df_proto = generator_probability_map(generator, z_dim, 
                                                                                                              classifier, target_class_idx,
                                                                                                              n_samples,
                                                                                                              
                                                                                                              init_targets = points[:,-1],#np.repeat(range(10),10),
                                                                                                              
                                                                                                              init_points = points[:,:z_dim],
                                                                                                              )

    
prots = pca_pipe_proto.transform(np.array(list(generator.prototypes.values()))[:,0,:,:].squeeze(1))
z0s = pca_pipe_proto.transform(np.unique(trajectory_matrices[label][0, :, :], axis=0))
space_radius= (pca_df_proto.PC1.min(),pca_df_proto.PC1.max(), pca_df_proto.PC2.min(), pca_df_proto.PC1.max())
  
plot_probability_map_grid(probability_map, coordinates, pca_pipe_proto,
                                     generator, z_dim, classifier,
                                     target_class_idx, non_target_class_idx, class_name_dict, clean_clf_str,
                                     n_samples,
                                     space_radius= (pca_df_proto.PC1.min(),pca_df_proto.PC1.max(), pca_df_proto.PC2.min(), pca_df_proto.PC1.max()), 
                                     ext = ext,
                                     seed=seed,
                                     maxmin=True,
                                     fname = os.path.join(figpath, f'pmap_{clean_clf_str}_{reduction}'),
                                     accuracy= f'{classifier.history_to_df()["val_acc"].values[-1]:.2f}'
                          )

"""
PROBABILITY LINE PLOTS
"""
transform = None
subject=8
npz_file_path = f'../../../../data/fMRIsynth/subj0{subject}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri_with_indices.npz'
trainset = load_dataset(dataset, transform, npz_file_path, train=True)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
idx = 0 if target_class_idx==min(classifier.classes) else 1
classifier.to(device)

tgt, non_tgt = tgt_non_tgt
generator.target_size = 256
metric = 'p'
metric_dfs = all_dfs[metric]
for i, key in enumerate(trajectory_matrices.keys()):
        ALL_TRAJS = trajectory_matrices[key] # shape (T, Ntraj, zdim)
        tnames = names_dfs[key]
        if ALL_TRAJS.shape[1]== 0: continue    
        title = key + f' - {metric}(t)'
        clean_title = re.sub('[^a-zA-Z0-9]','', title)
        fname = os.path.join(figpath, f'{clean_clf_str}_{clean_title}')
        offset=2 if metric=='p' else 0
        if ('Random' in key) and (metric=='p'):
            heatmap_mean_X_over_time(random_probability_dfs[key], ALL_TRAJS, tnames, latent_prototype, 
                                     class_name_dict, title, save=True, fname=fname, offset=offset,
                                     ext=ext, N_A=10)
        else:
            heatmap_mean_X_over_time(metric_dfs[key], ALL_TRAJS, tnames, latent_prototype,
                                      class_name_dict, title, save=True, fname=fname, offset=offset,
                                      ext=ext, N_A=10)
#%%
"""
TRAJECTORIES
"""
for label, trajectory_matrix in tqdm(trajectory_matrices.items(),
                                     desc='plotting trajectories',
                                     total=len(list(trajectory_matrices.keys()))):
    if 'memory' not in label: continue
    if trajectory_matrix.shape[1] == 0: print(label, ' empty'); break
    urname, igdis = keys[label]
    
    if 'Random' in label: continue
    n_time, n_samples, _ = trajectory_matrix.shape
    
    # Plot all trajectories in 2D space
    fig, ax = plt.subplots(figsize=(8, 8))
    h, w = probability_map.shape
    for ax_ in [ax]:
        im = ax_.imshow(
            probability_map,
            extent=space_radius,
            origin="lower",
            cmap="viridis",
            vmin=0, vmax=1,
            alpha=1
        )
                
    data = np.transpose(np.array([pca_pipe_proto.transform(t) for t in
                     np.transpose(trajectory_matrix, axes = (1,0,2))]),
                        axes = (1,0,2))
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
    
    for mean_traj_ in tqdm(TRAJS_groupbyR.values(),
                          total = len(list(TRAJS_groupbyR.values())),
                          desc = 'plotting mean trajs'):
        mean_traj = pca_pipe_proto.transform(mean_traj_)
        colors = cm.inferno(np.linspace(0, 1, n_time))
        for t in range(n_time - 1):
            ax.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2,  1], color=colors[t], linewidth=1.5)
    # Attach colorbars without shrinking
    for fig_,ax_ in zip([fig], [ax]):
        divider = make_axes_locatable(ax_)
        cax_right = divider.append_axes("right", size="5%", pad=0.1)
        cax_bottom = divider.append_axes("bottom", size="5%", pad=0.55)
        fig_.colorbar(sm, cax=cax_right, orientation="vertical", label='Time')
        fig_.colorbar(im, cax=cax_bottom, label=r"$p(y=y^\ast \mid x)$;  $x$: projection of $z$ to the native space of the data"+"\n"+r"(PCA visualization of $z$ coordinates)", location='bottom')
    for ax_ in [ax]:
        ax_.plot(prots[tgt][0], prots[tgt][1], markersize=18, marker = 'X', color='c', markeredgecolor='white', label=r"Target: $z : y=y^\ast$")
        ax_.plot(prots[non_tgt][0], prots[non_tgt][1], markersize=18, marker = 'o', color='c', markeredgecolor='white', label=r"Alternative: $z : y=y^{alt}$")
        ax_.set_xlabel(r'$PC_1$')
        ax_.set_ylabel(r'$PC_2$')
        
        ax_.set_xlim(space_radius[0], space_radius[1])
        ax_.set_ylim(space_radius[2], space_radius[3])
        ax_.set_aspect("equal")

    ax.grid(True)
    
    fig.savefig(os.path.join(figpath, f'trajs_{clean_clf_str}_{urname}_igdis{igdis}_{reduction}.{ext}'),
                # dpi=600,
                # transparent=True
                )
    
    plt.show()

